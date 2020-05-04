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


def generatePseudoMask(image, mask):
    # Return segmentation and input img as np array
    input_image_array = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()

    # create a mask of lungs with original values
    indices = mask > 0
    out_arr = input_image_array * indices

    # isVector=True to get a 2D vector image instead of 3D image
    out_img = sitk.GetImageFromArray(out_arr)
    out_img.SetSpacing(spacing)
    out_img.SetDirection(direction)
    out_img.SetOrigin(origin)

    sitk.WriteImage(out_img, 'lung_extraction.nrrd')

    return out_img


def extractValuesInsidePalette(image, mask):
    t1_low = -983
    t2_low = -860
    t1_high = -860
    t2_high = -740

    image_arr = sitk.GetArrayFromImage(image)

    print('clean segmentation')
    # mask[image < t1_low & image > t2_high] = 0

    # # Generate a mask that is 1 in the lower perfusion
    # # zone and 1 in the higher (0 in background)
    # perfusion_mask = mask
    # perfusion_mask[image > t1_low & image < t2_low] = 1
    # perfusion_mask[image > t1_high & image < t2_high] = 2

    pxs = mask.shape[0]*mask.shape[1]*mask.shape[2]

    perfusion_mask = mask

    for k in range(mask.shape[0]):
        print('cleaning', k, ' // ', mask.shape[0])
        for j in range(mask.shape[1]):
            for i in range(mask.shape[2]):
                if (mask[k][j][i] == 0):
                    continue
                elif (mask[k][j][i] > 0 and image_arr[k][j][i] < t1_low):
                    perfusion_mask[k][j][i] = 0
                elif (mask[k][j][i] > 0 and image_arr[k][j][i] <= t2_low):
                    perfusion_mask[k][j][i] = 10
                elif (mask[k][j][i] > 0 and image_arr[k][j][i] <= t2_high):
                    perfusion_mask[k][j][i] = 20
                elif (mask[k][j][i] > 0 and image_arr[k][j][i] > t2_high):
                    perfusion_mask[k][j][i] = 0

    # Convert to itk image
    out_img = sitk.GetImageFromArray(mask)

    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()
    out_img.SetSpacing(spacing)
    out_img.SetDirection(direction)
    out_img.SetOrigin(origin)

    # Write output
    print('Writing output cleaned...')
    sitk.WriteImage(out_img, 'lung_mask_palette.nrrd')

    return mask


def generateCSV(image_f, image_m, lung_mask, perfusion_mask, folder_path):
    print('Writing csv...')

    file_path = os.path.join(folder_path, 'output.csv')
    with open(file_path, mode='w+') as csv_file:
        writer = csv.writer(csv_file, delimiter=';',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['i', 'j', 'k', 'valore senza mdc',
                         'valore con mdc', 'polmone', 'perfusion'])
        for k in range(lung_mask.shape[0]):
            print('writing csv', k, ' // ', lung_mask.shape[0])
            for j in range(lung_mask.shape[1]):
                for i in range(lung_mask.shape[2]):
                    f_pix = image_f.GetPixel(i, j, k)
                    m_pix = image_m.GetPixel(i, j, k)
                    if (lung_mask[k][j][i] > 0):
                        writer.writerow(
                            [i, j, k, image_f.GetPixel(i, j, k), image_m.GetPixel(i, j, k), lung_mask[k][j][i], perfusion_mask[k][j][i]])


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
    # HACK force z spacing to 1.0
    # sp = image.GetSpacing()
    # sp = (sp[0], sp[1], 1.0)
    # image.SetSpacing(sp)
    img_basename = os.path.basename(dcm_path)
    sitk.WriteImage(image, nrrd_path)


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

    parser.add_argument('--dicomdir_fixed', action='store',
                        default=None, help='the dicom folder of target image')

    parser.add_argument('--dicomdir_moving', action='store',
                        default=None, help='the dicom folder of source image')

    parser.add_argument('--outfolder', action='store',
                        default="output/", help='the output folder')

    args = parser.parse_args()
    if (not args.dicomdir_fixed):
        sys.exit("Please provide dicom series directory")
    if (not args.dicomdir_moving):
        sys.exit("Please provide dicom series directory")

    path_fixed = args.dicomdir_fixed
    path_moving = args.dicomdir_moving

    # create temp folder
    temp_path = os.path.join(os.getcwd(), "temp/")
    os.makedirs(temp_path, exist_ok=True)

    # create path to temp files
    nrrd_fixed = os.path.join(temp_path, 'fixed.nrrd')
    nrrd_moving = os.path.join(temp_path, 'moving.nrrd')
    nrrd_moved = os.path.join(temp_path, 'result.nrrd')

    # convert input dicom to nrrd
    dicom2nrrd(path_fixed, nrrd_fixed)
    dicom2nrrd(path_moving, nrrd_moving)

    # register moving nrrd on fixed nrrd
    register(nrrd_fixed, nrrd_moving, temp_path)

    # read registration results
    image_senzamdc = sitk.ReadImage(nrrd_fixed)
    image_conmdc = sitk.ReadImage(nrrd_moved)

    # run segmentation (or load a mask)
    if (args.use_mask):
        mask = sitk.ReadImage(args.use_mask)
        segmentation_arr = sitk.GetArrayFromImage(mask)
    else:
        segmentation_arr = do_prediction(image_senzamdc, args.force_cpu)

    # extract only values inside the target palette
    segmentation_arr_cleaned = extractValuesInsidePalette(
        image_conmdc, segmentation_arr)

    # write pseudo-mask (for dev)
    # pseudo_mask = generatePseudoMask(image_fixed, segmentation_arr)
    # pseudo_mask = generatePseudoMask(image_move, segmentation_arr)

    # generate csv file
    generateCSV(
        image_senzamdc, image_conmdc, segmentation_arr, segmentation_arr_cleaned, args.outfolder)

    print('DONE, output in:', args.outfolder)
