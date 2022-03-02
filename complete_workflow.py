from lungmask import mask as lung_mask
import SimpleITK as sitk
import os
import numpy as np
import csv
import torch
import argparse
import time

from utils import dicom2nrrd


def label_mask(image, mask, thresholds):
    t1_low = thresholds[0]
    t2_low = thresholds[1]
    t1_high = thresholds[1]
    t2_high = thresholds[2]

    pxs = mask.shape[0] * mask.shape[1] * mask.shape[2]
    perfusion_mask = np.zeros(mask.shape)

    image_arr = sitk.GetArrayFromImage(image)

    # assign a label based on original value & thresholds & side
    # first digit is the purfusion zone, second digit (units) is the lung
    # !!! don't touch the order !!! highest threshold first
    perfusion_mask[(mask == 1) & (image_arr > t2_high)] = 31
    perfusion_mask[(mask == 1) & (image_arr <= t2_high)] = 21
    perfusion_mask[(mask == 1) & (image_arr <= t2_low)] = 11
    perfusion_mask[(mask == 2) & (image_arr > t2_high)] = 32
    perfusion_mask[(mask == 2) & (image_arr <= t2_high)] = 22
    perfusion_mask[(mask == 2) & (image_arr <= t2_low)] = 12
    perfusion_mask[(mask > 0) & (image_arr < t1_low)] = 0

    return perfusion_mask


def do_prediction(input_image, force_cpu, dev=False):
    # Run segmentation
    print("Running segmentation...")
    if dev:
        # only for dev: load mask from file
        print("WARNING: DEV MODE - loading segmentation from file")
        out_img = sitk.ReadImage("./lung_segmentation_original.nrrd")
    else:
        segmentation = lung_mask.apply(input_image, force_cpu=force_cpu)
        # free memory
        torch.cuda.empty_cache()
        # Convert to itk image
        # isVector=True to get a 2D vector image instead of 3D image
        out_img = sitk.GetImageFromArray(segmentation)

    print("Erosion...")
    # correct with erosion filter
    tic = time.time()
    eroder = sitk.BinaryErodeImageFilter()
    eroder.SetKernelType(sitk.sitkBall)
    eroder.SetKernelRadius(1)
    # lung 1
    eroder.SetForegroundValue(1)
    out_img = eroder.Execute(out_img)
    # lung 2
    eroder.SetForegroundValue(2)
    out_img = eroder.Execute(out_img)
    toc = time.time()
    print("erosion:", toc - tic)

    # prepare output image
    spacing = input_image.GetSpacing()
    direction = input_image.GetDirection()
    origin = input_image.GetOrigin()
    out_img.SetSpacing(spacing)
    out_img.SetDirection(direction)
    out_img.SetOrigin(origin)

    # Write output
    print("Writing output...")
    sitk.WriteImage(out_img, "lung_segmentation.nrrd")

    return out_img


def label_image(mask, image, tresholds, folder_path):
    print("labeling image with thresholds: ", tresholds)

    perfusion_mask = label_mask(image, mask, tresholds)

    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()

    # Write perfusion mask
    print("Writing perfusion mask...")
    out_img = sitk.GetImageFromArray(perfusion_mask)
    out_img.SetSpacing(spacing)
    out_img.SetDirection(direction)
    out_img.SetOrigin(origin)

    sitk.WriteImage(out_img, os.path.join(folder_path, "lung_mask_palette.nrrd"))

    # write pulmonary-only image for debug
    # apply lung mask to original image and set background to -1000
    image_arr = sitk.GetArrayFromImage(image)
    mask[mask == 2] = 1
    lungs_arr = image_arr * mask
    lungs_arr[mask == 0] = -1000
    lungs = sitk.GetImageFromArray(lungs_arr)
    print("Writing lungs extraction...")
    # Write lungs extraction
    lungs.SetSpacing(spacing)
    lungs.SetDirection(direction)
    lungs.SetOrigin(origin)
    sitk.WriteImage(lungs, os.path.join(folder_path, "lungs.nrrd"))

    return perfusion_mask


def compute_stats(perf_arr, ignoreHighThreshold, spacing, dims, outdir):

    # sum by label
    # first digit is the purfusion zone, second digit (units) is the lung
    label_11 = np.count_nonzero(perf_arr == 11)  # low perf left lung
    label_21 = np.count_nonzero(perf_arr == 21)
    label_31 = np.count_nonzero(perf_arr == 31)
    label_12 = np.count_nonzero(perf_arr == 12)  # low perf right lung
    label_22 = np.count_nonzero(perf_arr == 22)
    label_32 = np.count_nonzero(perf_arr == 32)

    # compute total volume for each side
    if ignoreHighThreshold:
        tot_vol_left = label_11 + label_21
        tot_vol_right = label_12 + label_22
    else:
        tot_vol_left = label_11 + label_21 + label_31
        tot_vol_right = label_12 + label_22 + label_32

    print(tot_vol_left, tot_vol_right)

    # assign low perfusion volume
    low_perf_vol_left = label_11
    low_perf_vol_right = label_12

    print(low_perf_vol_left, low_perf_vol_right)

    # compute low perfusion volume percentage
    perc_low_perf_left = (low_perf_vol_left / tot_vol_left) * 100
    perc_low_perf_right = (low_perf_vol_right / tot_vol_right) * 100

    print(perc_low_perf_left, perc_low_perf_right)

    # compute conversion factor n_of_px -> volume
    conversion_factor = spacing[0] * spacing[1] * spacing[2]

    # write output csv
    file_path = os.path.join(outdir, "./output.csv")
    with open(file_path, mode="w+") as csv_file:
        writer = csv.writer(
            csv_file, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow(["desc", "n_of_px", "vol [mm^3]" "percentage"])
        writer.writerow(
            ["tot_vol_left", tot_vol_left, tot_vol_left * conversion_factor, ""]
        )
        writer.writerow(
            [
                "low_perf_left",
                low_perf_vol_left,
                low_perf_vol_left * conversion_factor,
                perc_low_perf_left,
            ]
        )
        writer.writerow(
            ["tot_vol_right", tot_vol_right, tot_vol_right * conversion_factor, ""]
        )
        writer.writerow(
            [
                "low_perf_right",
                low_perf_vol_right,
                low_perf_vol_right * conversion_factor,
                perc_low_perf_right,
            ]
        )

    return


if __name__ == "__main__":

    # init arg parser
    parser = argparse.ArgumentParser(
        description="Extract lungs from a given image and store output stats in a csv file"
    )

    parser.add_argument(
        "--ignore_high_threshold",
        action="store_true",
        help="do not consider label 3 in total",
    )

    parser.add_argument(
        "--force_cpu", action="store_true", help="force using cpu for segmentation"
    )

    parser.add_argument(
        "--dicomdir",
        action="store",
        required=True,
        help="the dicom folder of target image",
    )

    parser.add_argument(
        "--outdir", action="store", help="the output folder", required=True
    )

    parser.add_argument(
        "--thresholds",
        action="store",
        nargs="+",
        type=float,
        help="array of tresholds",
        default=[-1000, -930, -770],
    )

    parser.add_argument(
        "--load_mask", action="store_true", help="use pre-computed mask FOR DEV"
    )

    args = parser.parse_args()

    path_image = args.dicomdir

    tic = time.perf_counter()

    # create temporary folder
    temp_path = os.path.join(os.getcwd(), "temp/")
    os.makedirs(temp_path, exist_ok=True)

    # create path to temp files
    nrrd_image_path = os.path.join(temp_path, "image.nrrd")

    # convert input dicom to nrrd
    image = dicom2nrrd(path_image, nrrd_image_path)

    # run segmentation with lungmask (or load a mask)
    segmentation = do_prediction(image, args.force_cpu, args.load_mask)
    segmentation_arr = sitk.GetArrayFromImage(segmentation)

    # extract only values inside the target palette (thresholds)
    perfusion_mask = label_image(segmentation_arr, image, args.thresholds, args.outdir)

    # compute volumes
    compute_stats(
        perfusion_mask,
        args.ignore_high_threshold,
        image.GetSpacing(),
        image.GetSize(),
        args.outdir,
    )

    toc = time.perf_counter()

    print(f"DONE in {toc - tic:0.1f} seconds")
    print("Output in:", args.outdir)
