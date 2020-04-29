from lungmask import mask
import SimpleITK as sitk
import os
import numpy as np
import csv
import torch
import argparse
import sys


def moveIntoFolder(f, name, wdir):
    name = name.strip()
    name = name.replace(" ", "_")
    name = name.replace(",", "_")
    name = name.replace(".", "_")
    dest_folder = os.path.join(wdir, name)
    wdir_path = os.path.join(wdir, f)
    dest_path = os.path.join(dest_folder, f)
    os.makedirs(dest_folder, exist_ok=True)
    print(wdir_path, ' >> ', dest_path)
    os.rename(wdir_path, dest_path)


def getImageSeriesId(file_name, series_list):
    print('Reading image...')
    # A file name that belongs to the series we want to read

    # Read the file's meta-information without reading bulk pixel data
    # print('Reading image...')
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(file_name)
    file_reader.ReadImageInformation()

    # Get the sorted file names, opens all files in the directory and reads the meta-information
    # without reading the bulk pixel data
    series_ID = file_reader.GetMetaData('0020|000e')
    description = file_reader.GetMetaData('0008|103e')
    print('seriesId', series_ID)
    print('description', description)

    if series_ID not in series_list:
        series_list.append((series_ID, description))

    return series_ID


def organize_series(files, data_directory):
    series_list = []

    for file in files:
        getImageSeriesId(file, series_list)

    for (series_ID, description) in series_list:
        sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            data_directory, series_ID)
        for file_path in sorted_file_names:
            f = os.path.basename(file_path)
            moveIntoFolder(f, description, './DICOM/')


def do_prediction(input_image):
    # Run segmentation
    print('Running segmentation...')
    segmentation = mask.apply(input_image)
    # free memory
    torch.cuda.empty_cache()

    # Convert to itk image
    # isVector=True to get a 2D vector image instead of 3D image
    out_img = sitk.GetImageFromArray(segmentation)

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
        series_id = getImageSeriesId(os.path.join(root, files[0]), [])

    sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        series_folder, series_id)

    # Read the bulk pixel data
    input_image = sitk.ReadImage(sorted_file_names)
    return input_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract lung values from given images and store output in target csv file')

    parser.add_argument('--use_mask', action='store',
                        help='use passed mask instead of running segmentation (intended for dev)')

    parser.add_argument('--dicomdir_fixed', action='store',
                        default=None, help='the dicom folder of target image')

    parser.add_argument('--dicomdir_move', action='store',
                        default=None, help='the dicom folder of source image')

    parser.add_argument('--outfile', action='store',
                        default="./output.csv", help='the output csv file')

    args = parser.parse_args()
    if (not args.dicomdir_fixed):
        sys.exit("Please provide dicom series directory")

    # TODO registration !
    path_fixed = args.dicomdir_fixed
    path_move = args.dicomdir_move
    image_fixed = readImage(path_fixed)
    image_move = readImage(path_move)
    if (args.use_mask):
        mask = sitk.ReadImage(args.use_mask)
        segmentation_arr = sitk.GetArrayFromImage(mask)
    else:
        segmentation_arr = do_prediction(image_fixed)
    # pseudo_mask = generatePseudoMask(image_fixed, segmentation_arr)
    # pseudo_mask = generatePseudoMask(image_move, segmentation_arr)
    generateCSV(image_fixed, image_move, segmentation_arr, args.outfile)
    print('DONE', args.outfile)
