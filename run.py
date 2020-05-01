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


def generateCSV(image_f, image_m, mask, filepath):
    print('Writing csv...')

    with open(filepath, mode='w+') as csv_file:
        writer = csv.writer(csv_file, delimiter=';',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['i', 'j', 'k', 'valore senza mdc', 'valore con mdc'])
        for k in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                for i in range(mask.shape[2]):
                    if mask[k][j][i]:
                        writer.writerow(
                            [i, j, k, image_f.GetPixel(i, j, k), image_m.GetPixel(i, j, k)])


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

    parser.add_argument('--outfile', action='store',
                        default="./output.csv", help='the output csv file')

    args = parser.parse_args()
    if (not args.dicomdir_fixed):
        sys.exit("Please provide dicom series directory")
    if (not args.dicomdir_moving):
        sys.exit("Please provide dicom series directory")

    path_fixed = args.dicomdir_fixed
    path_moving = args.dicomdir_moving

    # create temp folder
    temp_path = os.path.join(os.getcwd(), "./temp/")
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

    # write pseudo-mask (for dev)
    # pseudo_mask = generatePseudoMask(image_fixed, segmentation_arr)
    # pseudo_mask = generatePseudoMask(image_move, segmentation_arr)

    # generate csv file
    generateCSV(
        image_senzamdc, image_conmdc, segmentation_arr, args.outfile)

    print('DONE', args.outfile)
