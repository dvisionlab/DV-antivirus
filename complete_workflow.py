from lungmask import mask
import SimpleITK as sitk
import os
import numpy as np
import csv
import torch
import argparse
import time

from utils import getImageSeriesId, readImage, dicom2nrrd

# OLD, SLOW METHOD
# def label_mask_1(image, mask, thresholds):
#     t1_low = thresholds[0]
#     t2_low = thresholds[1]
#     t1_high = thresholds[1]
#     t2_high = thresholds[2]

#     pxs = mask.shape[0] * mask.shape[1] * mask.shape[2]
#     perfusion_mask = np.zeros(mask.shape)

#     # for k in range(mask.shape[0]): #TODO restore
#     for k in range(100, 110):
#         print("slice >>", k, " // ", mask.shape[0])  # TODO restore
#         sum_slice = np.sum(mask[k])
#         if sum_slice == 0:
#             continue
#         for j in range(mask.shape[1]):
#             for i in range(mask.shape[2]):
#                 pix = image.GetPixel(i, j, k)
#                 mask_val = mask[k, j, i]
#                 if mask_val == 0:
#                     continue
#                 elif mask_val > 0 and pix < t1_low:
#                     perfusion_mask[k, j, i] = 0
#                 elif mask_val == 1 and pix <= t2_low:
#                     perfusion_mask[k, j, i] = 11
#                 elif mask_val == 1 and pix <= t2_high:
#                     perfusion_mask[k, j, i] = 21
#                 elif mask_val == 1 and pix > t2_high:
#                     perfusion_mask[k, j, i] = 31
#                 elif mask_val == 2 and pix <= t2_low:
#                     perfusion_mask[k, j, i] = 12
#                 elif mask_val == 2 and pix <= t2_high:
#                     perfusion_mask[k, j, i] = 22
#                 elif mask_val == 2 and pix > t2_high:
#                     perfusion_mask[k, j, i] = 32
#                 else:
#                     print(k, j, i)

#     return perfusion_mask


def label_mask(image, mask, thresholds):
    t1_low = thresholds[0]
    t2_low = thresholds[1]
    t1_high = thresholds[1]
    t2_high = thresholds[2]

    pxs = mask.shape[0] * mask.shape[1] * mask.shape[2]
    perfusion_mask = np.zeros(mask.shape)

    image_arr = sitk.GetArrayFromImage(image)

    # !!! don't touch the order !!! highest threshold first
    perfusion_mask[(mask == 1) & (image_arr > t2_high)] = 31
    perfusion_mask[(mask == 1) & (image_arr <= t2_high)] = 21
    perfusion_mask[(mask == 1) & (image_arr <= t2_low)] = 11
    perfusion_mask[(mask == 2) & (image_arr > t2_high)] = 32
    perfusion_mask[(mask == 2) & (image_arr <= t2_high)] = 22
    perfusion_mask[(mask == 2) & (image_arr <= t2_low)] = 12
    perfusion_mask[(mask > 0) & (image_arr < t1_low)] = 0

    return perfusion_mask


def do_prediction(input_image, force_cpu):
    # Run segmentation
    print("Running segmentation...")
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
    print("Writing output...")
    sitk.WriteImage(out_img, "lung_segmentation.nrrd")

    return segmentation


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


def compute_stats(perf_arr, thresholds):
    # first digit is the purfusion zone, second digit (units) is the lung

    label_00 = np.count_nonzero(perf_arr == 0)
    label_11 = np.count_nonzero(perf_arr == 11)  # low perf left lung
    label_21 = np.count_nonzero(perf_arr == 21)
    label_31 = np.count_nonzero(perf_arr == 31)
    label_12 = np.count_nonzero(perf_arr == 12)  # low perf right lung
    label_22 = np.count_nonzero(perf_arr == 22)
    label_32 = np.count_nonzero(perf_arr == 32)

    tot_vol_left = label_11 + label_21 + label_31
    tot_vol_right = label_12 + label_22 + label_32

    print(tot_vol_left, tot_vol_right)

    low_perf_vol_left = label_11
    low_perf_vol_right = label_12

    print(low_perf_vol_left, low_perf_vol_right)

    perc_low_perf_left = (low_perf_vol_left / tot_vol_left) * 100
    perc_low_perf_right = (low_perf_vol_right / tot_vol_right) * 100

    print(perc_low_perf_left, perc_low_perf_right)

    file_path = os.path.join("./new_output.csv")
    with open(file_path, mode="w+") as csv_file:
        writer = csv.writer(
            csv_file, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow(["desc", "n_of_px", "percentage"])
        writer.writerow(["tot_vol_left", tot_vol_left, ""])
        writer.writerow(["low_perf_left", low_perf_vol_left, perc_low_perf_left])
        writer.writerow(["tot_vol_right", tot_vol_right, ""])
        writer.writerow(["low_perf_right", low_perf_vol_right, perc_low_perf_right])

    return


if __name__ == "__main__":
    # init arg parser

    parser = argparse.ArgumentParser(
        description="Extract lungs from a given image and store output stats in a csv file"
    )

    parser.add_argument(
        "--force_cpu", action="store_true", help="force using cpu for segmentation"
    )

    parser.add_argument(
        "--dicomdir",
        action="store",
        default=None,
        help="the dicom folder of target image",
    )

    parser.add_argument(
        "--outfolder", action="store", default="output/", help="the output folder"
    )

    parser.add_argument(
        "--thresholds",
        action="store",
        nargs="+",
        type=float,
        help="array of tresholds",
        default=[-1000, -930, -770],
    )

    args = parser.parse_args()

    path_image = args.dicomdir

    tic = time.perf_counter()

    # create temp folder
    temp_path = os.path.join(os.getcwd(), "temp/")
    os.makedirs(temp_path, exist_ok=True)

    # create path to temp files
    nrrd_image_path = os.path.join(temp_path, "image.nrrd")

    # convert input dicom to nrrd
    image = dicom2nrrd(path_image, nrrd_image_path)

    # run segmentation (or load a mask)
    segmentation_arr = do_prediction(image, args.force_cpu)

    # only for dev: load mask from file
    # mask = sitk.ReadImage("./lung_segmentation.nrrd")
    # segmentation_arr = sitk.GetArrayFromImage(mask)

    # extract only values inside the target palette
    perfusion_mask = label_image(
        segmentation_arr, image, args.thresholds, args.outfolder
    )

    # compute volumes
    compute_stats(perfusion_mask, args.thresholds)

    toc = time.perf_counter()

    print(f"DONE in {toc - tic:0.1f} seconds")
    print("Output in:", args.outfolder)