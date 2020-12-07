import subprocess
from lungmask import mask
import SimpleITK as sitk
import os
import numpy as np
import csv
import torch
import argparse
import sys

from utils import getImageSeriesId, readImage, dicom2nrrd


def register(image_fixed, image_move, folder_out):
    print("Run registration...")
    return subprocess.call(
        "./runRegistration.sh -f %s -m %s -o %s"
        % (image_fixed, image_move, folder_out),
        shell=True,
    )


if __name__ == "__main__":
    # init arg parser

    parser = argparse.ArgumentParser(
        description="Register dicom umages using elastix lib"
    )

    parser.add_argument(
        "--dicomdir_fixed",
        action="store",
        default=None,
        help="the dicom folder of target image",
    )

    parser.add_argument(
        "--dicomdir_moving",
        action="store",
        default=None,
        help="the dicom folder of target image",
    )

    parser.add_argument(
        "--nrrd_fixed",
        action="store",
        default=None,
        help="the path to the fixed image",
    )

    parser.add_argument(
        "--nrrd_moving",
        action="store",
        default=None,
        help="the path to the moving image",
    )

    parser.add_argument("--outpath", action="store", help="the output file path")

    parser.add_argument(
        "--subtract",
        action="store_true",
        default=None,
        help="generate subtracted image",
    )

    args = parser.parse_args()

    # create temp folder
    temp_path = os.path.join(os.getcwd(), "temp/")
    os.makedirs(temp_path, exist_ok=True)

    if args.dicomdir_fixed:
        dicom_fixed_path = args.dicomdir_fixed

        # create path to temp files
        nrrd_fixed_path = os.path.join(temp_path, "fixed.nrrd")

        # convert input dicom to nrrd
        dicom2nrrd(dicom_fixed_path, nrrd_fixed_path)
        image_fixed = nrrd_fixed_path

    elif args.nrrd_fixed:
        image_fixed = args.nrrd_fixed

    if args.dicomdir_moving:
        dicom_moving_path = args.dicomdir_moving

        # create path to temp files
        nrrd_moving_path = os.path.join(temp_path, "fixed.nrrd")

        # convert input dicom to nrrd
        dicom2nrrd(dicom_moving_path, nrrd_moving_path)
        image_moving = nrrd_fixed_path

    elif args.nrrd_moving:
        image_moving = args.nrrd_moving

    print(image_fixed, image_moving)

    if image_moving is None or image_fixed is None:
        sys.exit("Please provide both input images")

    if args.outpath is None:
        sys.exit("Please provide output path")

    res = register(image_fixed, image_moving, temp_path)

    if res == 0:
        # move to output path
        os.rename(os.path.join(temp_path, "result.nrrd"), args.outpath)

        # remove temp dir
        # os.rmdir(temp_path)
        print("\nDONE registration, output at:", args.outpath)

        if args.subtract:
            im1 = sitk.ReadImage(image_fixed)
            im2 = sitk.ReadImage(args.outpath)
            arr1 = sitk.GetArrayFromImage(im1)
            arr2 = sitk.GetArrayFromImage(im2)
            diff_arr = arr1 - arr2
            diff_im = sitk.GetImageFromArray(diff_arr)
            diff_im.SetOrigin(im1.GetOrigin())
            diff_im.SetSpacing(im1.GetSpacing())
            diff_im.SetDirection(im1.GetDirection())
            # outdir = os.path.dirname(args.outpath)
            filename = os.path.splitext(args.outpath)[0] + "_sub.nrrd"
            sitk.WriteImage(diff_im, filename)
            print("\nSub image:", filename)
    else:
        print("Error in registration process")
