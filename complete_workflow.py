from lungmask import mask as lung_mask
import SimpleITK as sitk
import os
import numpy as np
import csv
import torch
import argparse
import time
from matplotlib import pyplot as plt
import jinja2
from weasyprint import HTML

from utils import dicom2nrrd


def label_mask(image, mask, thresholds):
    t1_low = thresholds[0]  # -1000
    t2_low = thresholds[1]  # -920
    t1_high = thresholds[1]  # -920
    t2_high = thresholds[2]  # -770

    pxs = mask.shape[0] * mask.shape[1] * mask.shape[2]
    perfusion_mask = np.zeros(mask.shape)

    image_arr = sitk.GetArrayFromImage(image)

    # assign a label based on original value & thresholds & side
    # first digit is the purfusion zone, second digit (units) is the lung
    # !!! don't touch the order !!! highest threshold first
    perfusion_mask[(mask == 1) & (image_arr > t2_high)] = 31  # less than -770
    perfusion_mask[(mask == 1) & (image_arr <= t2_high)] = 21  # less than/equal to -770
    perfusion_mask[(mask == 1) & (image_arr <= t2_low)] = 11  # less than/equal to -920
    perfusion_mask[(mask == 2) & (image_arr > t2_high)] = 32  # greater than -770
    perfusion_mask[(mask == 2) & (image_arr <= t2_high)] = 22  # less than/equal to -770
    perfusion_mask[(mask == 2) & (image_arr <= t2_low)] = 12  # less than/equal to -920
    perfusion_mask[(mask > 0) & (image_arr < t1_low)] = 0

    # perfusion zone that ranges from -1000 to -770 and from -1000 to -920
    perf_mask2 = np.zeros(mask.shape)
    perf_mask2[
        (mask == 1) & (t1_low <= image_arr) & (image_arr <= t2_high)
    ] = 41  # -1000/-770 right
    labels_41 = np.count_nonzero(perf_mask2 == 41)

    perf_mask2 = np.zeros(mask.shape)
    perf_mask2[
        (mask == 1) & (t1_low <= image_arr) & (image_arr <= t2_low)
    ] = 51  # -1000/-920 right
    labels_51 = np.count_nonzero(perf_mask2 == 51)

    perf_mask2 = np.zeros(mask.shape)
    perf_mask2[
        (mask == 2) & (t1_low <= image_arr) & (image_arr <= t2_high)
    ] = 42  # -1000/-770 left
    labels_42 = np.count_nonzero(perf_mask2 == 42)

    perf_mask2 = np.zeros(mask.shape)
    perf_mask2[
        (mask == 2) & (t1_low <= image_arr) & (image_arr <= t2_low)
    ] = 52  # -1000/-920 left
    labels_52 = np.count_nonzero(perf_mask2 == 52)

    perf_zones = [labels_41, labels_51, labels_42, labels_52]

    return perfusion_mask, perf_zones


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
    # lung 1 (right)
    eroder.SetForegroundValue(1)
    out_img = eroder.Execute(out_img)
    # lung 2 (left)
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

    perfusion_mask, perf_zones = label_mask(image, mask, tresholds)

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

    return perfusion_mask, perf_zones


def compute_stats(
    perf_arr, perf_zones_list, ignoreHighThreshold, spacing, dims, outdir
):
    # sum by label
    # first digit is the purfusion zone, second digit (units) is the lung
    label_11 = np.count_nonzero(perf_arr == 11)  # low perf right lung
    label_21 = np.count_nonzero(perf_arr == 21)
    label_31 = np.count_nonzero(perf_arr == 31)
    label_12 = np.count_nonzero(perf_arr == 12)  # low perf left lung
    label_22 = np.count_nonzero(perf_arr == 22)
    label_32 = np.count_nonzero(perf_arr == 32)

    # compute total volume for each side
    if ignoreHighThreshold:
        tot_vol_right = label_11 + label_21
        tot_vol_left = label_12 + label_22
    else:
        tot_vol_right = label_11 + label_21 + label_31
        tot_vol_left = label_12 + label_22 + label_32

    print(tot_vol_left, tot_vol_right)

    # assign low perfusion volume
    low_perf_vol_right = label_11
    low_perf_vol_left = label_12

    # assign perfusion range (-1000, -770) and (-1000,-920)
    right_1000_770 = perf_zones[0]
    right_1000_920 = perf_zones[1]
    left_1000_770 = perf_zones[2]
    left_1000_920 = perf_zones[3]

    print(low_perf_vol_left, low_perf_vol_right)

    # compute low perfusion volume percentage
    perc_low_perf_left = (low_perf_vol_left / tot_vol_left) * 100
    perc_low_perf_right = (low_perf_vol_right / tot_vol_right) * 100

    # compute volume percentage range (-1000, -770) and (-1000,-920)
    perc_left_1000_770 = (left_1000_770 / tot_vol_left) * 100
    perc_left_1000_920 = (left_1000_920 / tot_vol_left) * 100
    perc_right_1000_770 = (right_1000_770 / tot_vol_right) * 100
    perc_right_1000_920 = (right_1000_920 / tot_vol_right) * 100

    print(perc_low_perf_left, perc_low_perf_right)

    # compute conversion factor n_of_px -> volume
    conversion_factor = spacing[0] * spacing[1] * spacing[2]

    # write output csv
    file_path = os.path.join(outdir, "./output.csv")
    with open(file_path, mode="w+") as csv_file:
        writer = csv.writer(
            csv_file, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow(["desc", "n_of_px", "vol [mm^3]", "percentage"])
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
        writer.writerow(
            [
                "left_1000_770",
                left_1000_770,
                left_1000_770 * conversion_factor,
                perc_left_1000_770,
            ]
        )
        writer.writerow(
            [
                "left_1000_770",
                left_1000_920,
                left_1000_920 * conversion_factor,
                perc_left_1000_920,
            ]
        )
        writer.writerow(
            [
                "right_1000_770",
                right_1000_770,
                right_1000_770 * conversion_factor,
                perc_right_1000_770,
            ]
        )
        writer.writerow(
            [
                "right_1000_920",
                right_1000_920,
                right_1000_920 * conversion_factor,
                perc_right_1000_920,
            ]
        )

    # write a pdf file
    Titolo_tabella = "Summary"
    src = "temp/histogram.png"

    context = {
        "src": src,
        "Titolo_tabella": Titolo_tabella,
        "n_pix_1": tot_vol_left,
        "n_pix_2": low_perf_vol_left,
        "n_pix_3": tot_vol_right,
        "n_pix_4": low_perf_vol_right,
        "n_pix_5": left_1000_770,
        "n_pix_6": left_1000_920,
        "n_pix_7": right_1000_770,
        "n_pix_8": right_1000_920,
        "vol_1": tot_vol_left * conversion_factor,
        "vol_2": low_perf_vol_left * conversion_factor,
        "vol_3": tot_vol_right * conversion_factor,
        "vol_4": low_perf_vol_right * conversion_factor,
        "vol_5": left_1000_770 * conversion_factor,
        "vol_6": left_1000_920 * conversion_factor,
        "vol_7": right_1000_770 * conversion_factor,
        "vol_8": right_1000_920 * conversion_factor,
        "perc_1": "",
        "perc_2": perc_low_perf_left,
        "perc_3": "",
        "perc_4": perc_low_perf_right,
        "perc_5": perc_left_1000_770,
        "perc_6": perc_left_1000_920,
        "perc_7": perc_right_1000_770,
        "perc_8": perc_right_1000_920,
    }

    template_loader = jinja2.FileSystemLoader("./")
    template_env = jinja2.Environment(loader=template_loader)

    template = template_env.get_template("template_pdf.html")
    output_text = template.render(context).replace("^3", "³")

    with open("html_generated.html", "w") as f:
        f.write(output_text)

    HTML("html_generated.html").write_pdf(outdir + "/" + "summary.pdf")

    return


def maskToCSV(mask, image, tresholds, folder_path):
    mask = sitk.GetArrayFromImage(mask)

    t1_low = tresholds[0]
    t2_low = tresholds[1]
    t1_high = tresholds[1]
    t2_high = tresholds[2]

    print("clean segmentation and write csv")

    perfusion_mask = np.copy(mask)

    file_path = os.path.join(folder_path, "hist_output.csv")
    with open(file_path, mode="w+") as csv_file:
        writer = csv.writer(
            csv_file, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow(["valore con mdc", "polmone", "perfusion"])

        for k in range(mask.shape[0]):
            print(">>", k + 1, " // ", mask.shape[0])
            for j in range(mask.shape[1]):
                for i in range(mask.shape[2]):
                    pix = image.GetPixel(i, j, k)
                    if mask[k][j][i] == 0:
                        continue
                    elif mask[k][j][i] > 0 and pix < t1_low:
                        perfusion_mask[k][j][i] = 0
                        writer.writerow([pix, mask[k][j][i], perfusion_mask[k][j][i]])
                    elif mask[k][j][i] > 0 and pix <= t2_low:
                        perfusion_mask[k][j][i] = 10
                        writer.writerow([pix, mask[k][j][i], perfusion_mask[k][j][i]])
                    elif mask[k][j][i] > 0 and pix <= t2_high:
                        perfusion_mask[k][j][i] = 20
                        writer.writerow([pix, mask[k][j][i], perfusion_mask[k][j][i]])
                    elif mask[k][j][i] > 0 and pix > t2_high:
                        perfusion_mask[k][j][i] = 0
                        writer.writerow([pix, mask[k][j][i], perfusion_mask[k][j][i]])


def examine_threshold(csv_path, thresholds):
    """
    Examine a csv file.

    Generate histogram (saved as png in the temp folder)

    Parameters:
    csv_path (string): path to .csv to analyze
    thresholds (array): array in the form [lower_bound_value, threshold_value, high_bound_value]
    """

    print("Loading data...")
    data = np.genfromtxt(csv_path, delimiter=";", dtype=int, names=True)
    print("Data loaded.")
    # define which is left/right
    right_data = data[data["polmone"] == 1]
    left_data = data[data["polmone"] == 2]
    print("RIGHT LUNG\tLEFT LUNG")
    print(len(right_data), "\t", len(left_data))

    # select mdc column
    left_data_mdc = left_data["valore_con_mdc"]
    right_data_mdc = right_data["valore_con_mdc"]
    # select data inside the thresholds
    t1_low = thresholds[0]
    t2_low = thresholds[1]
    t1_high = thresholds[1]
    t2_high = thresholds[2]
    left_data_low = left_data_mdc[(t1_low <= left_data_mdc) & (left_data_mdc < t2_low)]
    right_data_low = right_data_mdc[
        (t1_low <= right_data_mdc) & (right_data_mdc < t2_low)
    ]
    left_data_high = left_data_mdc[
        (t1_high <= left_data_mdc) & (left_data_mdc < t2_high)
    ]
    right_data_high = right_data_mdc[
        (t1_high <= right_data_mdc) & (right_data_mdc < t2_high)
    ]

    print(len(left_data_low), "\t", len(left_data_high))
    print(len(right_data_low), "\t", len(right_data_high))

    print(left_data_low)

    # compute ratio on total volume
    left_ratio = len(left_data_low) / (len(left_data_low) + len(left_data_high))
    right_ratio = len(right_data_low) / (len(right_data_low) + len(right_data_high))
    print(right_ratio, "\t", left_ratio)

    # define bins
    start = t1_low
    stop = t2_high
    step = 1
    bins = range(int(start), int(stop) + 2, int(step))
    print("Hist bins input:", start, stop, step, bins)
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(25, 10))

    # LEFT
    nl, binsl, patchesl = axes[0].hist(
        left_data_mdc, bins, facecolor="blue", alpha=0.3, label="high perfusion"
    )
    axes[0].hist(
        left_data_low, bins, facecolor="blue", alpha=0.9, label="low perfusion"
    )
    # RIGHT
    nr, binsr, patchesr = axes[1].hist(
        right_data_mdc, bins, facecolor="red", alpha=0.3, label="high perfusion"
    )
    axes[1].hist(
        right_data_low, bins, facecolor="red", alpha=0.9, label="low perfusion"
    )

    # CHARTS SETUP
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    fig.suptitle(
        "Image with mc, Arterial 1mm \n Range: {} / {} \n Threshold: {}".format(
            thresholds[0], thresholds[2], thresholds[1]
        )
    )

    y_max = int(max(nl.max(), nr.max()))

    yticks = range(0, y_max, 100000)
    axes[0].set_title("Left lung low perfusion volume = %s" % left_ratio)
    axes[0].set_xticks(bins)
    axes[0].set_yticks(yticks)
    axes[1].set_title("Right lung low perfusion volume = %s" % right_ratio)
    axes[1].set_xticks(bins)
    axes[1].set_yticks(yticks)

    # save hist as png
    plt.savefig("temp/histogram.png")
    plt.show()


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
        default=[-1000, -920, -770],
    )

    parser.add_argument(
        "--load_mask", action="store_true", help="use pre-computed mask FOR DEV"
    )

    args = parser.parse_args()

    path_image = args.dicomdir

    tic = time.perf_counter()

    # create the output folder (if it does not exit)
    os.makedirs(args.outdir, exist_ok=True)

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
    perfusion_mask, perf_zones = label_image(
        segmentation_arr, image, args.thresholds, args.outdir
    )

    # generate the histogram
    maskToCSV(segmentation, image, args.thresholds, temp_path)
    examine_threshold(temp_path + "hist_output.csv", args.thresholds)

    # compute volumes
    compute_stats(
        perfusion_mask,
        perf_zones,
        args.ignore_high_threshold,
        image.GetSpacing(),
        image.GetSize(),
        args.outdir,
    )

    toc = time.perf_counter()

    print(f"DONE in {toc - tic:0.1f} seconds")
    print("Output in:", args.outdir)
