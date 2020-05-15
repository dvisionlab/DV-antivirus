import subprocess
from lungmask import mask
import SimpleITK as sitk
import os
import numpy as np
import csv
import torch
import argparse
import sys

from utils import getImageSeriesId


def do_prediction(input_image, force_cpu):
    # Run segmentation
    print('Running segmentation...')
    segmentation = mask.apply(input_image, force_cpu=force_cpu)
    # free memory
    torch.cuda.empty_cache()

    # Convert to itk image
    # isVector=True to get a 2D vector image instead of 3D image
    out_img = sitk.GetImageFromArray(segmentation)

    spacing = input_image.GetSpacing()
    direction = input_image.GetDirection()
    origin = input_image.GetOrigin()
    out_img.SetSpacing(spacing)
    out_img.SetDirection(direction)
    out_img.SetOrigin(origin)

    # Write output
    print('Writing output...')
    sitk.WriteImage(out_img, 'lung_segmentation.nrrd')

    return segmentation

# Generate a csv file in which for each ijk is reported the value of image
# with cm, value without cm label of lung (1 or 2) and label of perfusion
# low/high (10 or 20)


def maskToCSV(mask, image, folder_path):
    t1_low = -983
    t2_low = -870
    t1_high = -870
    t2_high = -740

    print('clean segmentation and write csv')

    pxs = mask.shape[0]*mask.shape[1]*mask.shape[2]

    perfusion_mask = np.copy(mask)

    file_path = os.path.join(folder_path, 'output.csv')
    with open(file_path, mode='w+') as csv_file:
        writer = csv.writer(csv_file, delimiter=';',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['valore con mdc', 'polmone', 'perfusion'])

        for k in range(mask.shape[0]):
            print('>>', k, ' // ', mask.shape[0])
            for j in range(mask.shape[1]):
                for i in range(mask.shape[2]):
                    pix = image.GetPixel(i, j, k)
                    if (mask[k][j][i] == 0):
                        continue
                    elif (mask[k][j][i] > 0 and pix < t1_low):
                        perfusion_mask[k][j][i] = 0
                    elif (mask[k][j][i] > 0 and pix <= t2_low):
                        perfusion_mask[k][j][i] = 10
                        writer.writerow(
                            [pix, mask[k][j][i], perfusion_mask[k][j][i]])
                    elif (mask[k][j][i] > 0 and pix <= t2_high):
                        perfusion_mask[k][j][i] = 20
                        writer.writerow(
                            [pix, mask[k][j][i], perfusion_mask[k][j][i]])
                    elif (mask[k][j][i] > 0 and pix > t2_high):
                        perfusion_mask[k][j][i] = 0

    # Convert to itk image
    out_img = sitk.GetImageFromArray(perfusion_mask)

    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()
    out_img.SetSpacing(spacing)
    out_img.SetDirection(direction)
    out_img.SetOrigin(origin)

    # Write output
    print('Writing output cleaned...')
    sitk.WriteImage(out_img, 'lung_mask_palette.nrrd')

    return perfusion_mask


def readImage(series_folder):
    for (root, dirs, files) in os.walk(series_folder):
        series_id = getImageSeriesId(os.path.join(root, files[0]), [], [])

    sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        series_folder, series_id)

    # Read the bulk pixel data
    input_image = sitk.ReadImage(sorted_file_names)
    return input_image


def dicom2nrrd(dcm_path, nrrd_path):
    print('Dicom to nrrd...')
    image = readImage(dcm_path)
    img_basename = os.path.basename(dcm_path)
    sitk.WriteImage(image, nrrd_path)
    return image


def register(image_fixed, image_move, folder_out):
    print('Run registration...')
    subprocess.call("./runRegistration.sh -f %s -m %s -o %s" %
                    (image_fixed, image_move, folder_out), shell=True)


if __name__ == "__main__":
    # init arg parser

    parser = argparse.ArgumentParser(
        description='Extract lung values from given images and store output in target csv file')

    parser.add_argument('--force_cpu', action='store_true',
                        help='force using cpu for segmentation')

    parser.add_argument('--use_mask', action='store',
                        help='use passed mask instead of running segmentation (intended for dev)')

    parser.add_argument('--dicomdir', action='store',
                        default=None, help='the dicom folder of target image')

    parser.add_argument('--outfolder', action='store',
                        default="output/", help='the output folder')

    args = parser.parse_args()
    if (not args.dicomdir):
        sys.exit("Please provide dicom series directory")

    path_image = args.dicomdir

    # create temp folder
    temp_path = os.path.join(os.getcwd(), "temp/")
    os.makedirs(temp_path, exist_ok=True)

    # create path to temp files
    nrrd_image_path = os.path.join(temp_path, 'image.nrrd')

    # convert input dicom to nrrd
    image = dicom2nrrd(path_image, nrrd_image_path)

    # run segmentation (or load a mask)
    if (args.use_mask):
        mask = sitk.ReadImage(args.use_mask)
        segmentation_arr = sitk.GetArrayFromImage(mask)
    else:
        segmentation_arr = do_prediction(image, args.force_cpu)

    # extract only values inside the target palette
    maskToCSV(segmentation_arr, image, args.outfolder)

    print('DONE, output in:', args.outfolder)
