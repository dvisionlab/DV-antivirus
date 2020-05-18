import SimpleITK as sitk
import os
import numpy as np
import csv
import argparse
import sys

# from utils import getImageSeriesId

if __name__ == "__main__":
    # init arg parser

    parser = argparse.ArgumentParser(
        description='Extract lung values from given images and store output in target csv file')

    parser.add_argument('--base_nrrd', action='store',
                        help='path to nrrd base image')
    parser.add_argument('--contrast_nrrd', action='store',
                        help='path to nrrd image with contrast medium')

    args = parser.parse_args()
    if (not args.base_nrrd or not args.contrast_nrrd):
        sys.exit("Please provide both nrrd files")

    print('Reading...')
    base_img = sitk.ReadImage(args.base_nrrd)
    contrast_img = sitk.ReadImage(args.contrast_nrrd)

    base_arr = sitk.GetArrayFromImage(base_img)
    contrast_arr = sitk.GetArrayFromImage(contrast_img)

    print('Subtracting...')
    diff_arr = contrast_arr - base_arr

    print(base_arr.shape)
    print(contrast_arr.shape)
    print(diff_arr.shape)

    diff_img = sitk.GetImageFromArray(diff_arr)

    spacing = base_img.GetSpacing()
    direction = base_img.GetDirection()
    origin = base_img.GetOrigin()
    diff_img.SetSpacing(spacing)
    diff_img.SetDirection(direction)
    diff_img.SetOrigin(origin)

    # put output in the nrrds folder
    outfolder = os.path.dirname(args.base_nrrd)
    outfile = os.path.join(outfolder, 'diff_image.nrrd')

    # Write output
    print('Writing output...')
    sitk.WriteImage(diff_img, outfile)
