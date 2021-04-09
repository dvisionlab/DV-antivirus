# segment vascular tree and airways

import SimpleITK as sitk
import os
import numpy as np
import argparse
import vtk

# import itk


def connected_threshold(nrrd_image, seeds, tresh):
    print("reading image")
    segmentationfilter = sitk.ConnectedThresholdImageFilter()
    segmentationfilter.SetUpper(tresh[0])
    segmentationfilter.SetLower(tresh[1])
    segmentationfilter.SetReplaceValue(1)
    for i in range(0, len(seeds), 3):
        seed = seeds[i : i + 3]
        print(seed)
        segmentationfilter.AddSeed(seed)
    print("run filter")
    result = segmentationfilter.Execute(nrrd_image)
    return result


if __name__ == "__main__":
    # init arg parser

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--image",
        action="store",
        default=None,
        help="the image",
    )

    parser.add_argument(
        "--mask",
        action="store",
        default=None,
        help="the mask",
    )

    parser.add_argument(
        "--seeds",
        action="store",
        nargs="+",
        type=int,
        help="array of xyz",
        default=[],
    )

    parser.add_argument(
        "--thresholds",
        action="store",
        nargs="+",
        type=int,
        help="array of upper and lower tr",
        default=[],
    )

    args = parser.parse_args()

print(args.seeds)
print(args.thresholds)

img = sitk.ReadImage(args.image)
out_img_path = "./airw.nrrd"
# tree = connected_threshold(img, args.seeds, args.thresholds)

seg = sitk.ConfidenceConnected(
    img,
    seedList=[args.seeds],
    numberOfIterations=10,
    multiplier=2.0,
    initialNeighborhoodRadius=10,
    replaceValue=1,
)

# initial sharpening ?
# erosion, then conn tr ?

# mask = sitk.ReadImage(args.mask)
# mask = sitk.GetArrayFromImage(mask)
# mask[mask > 1] = 1
# tree_arr = sitk.GetArrayFromImage(tree)
# tree_arr[mask == 1] = 0

# outttt = sitk.GetImageFromArray(tree_arr)

# thresholder = sitk.BinaryThresholdImageFilter()
# thresholder.SetLowerThreshold(0)
# # thresholder.SetUpperThreshold()
# thresholder.SetInsideValue(1)
# thresholder.SetOutsideValue(0)
# tree_mask = thresholder.Execute(img)

# # closing
# closing = sitk.BinaryMorphologicalClosingImageFilter()
# closing.SetKernelRadius(1)
# closing.SetForegroundValue(1)
# closed = closing.Execute(seg)

# # opening
opening = sitk.BinaryMorphologicalOpeningImageFilter()
opening.SetKernelRadius(2)
opening.SetForegroundValue(1)
opened = opening.Execute(seg)

# # closing
closing = sitk.BinaryMorphologicalClosingImageFilter()
closing.SetKernelRadius(1)
closing.SetForegroundValue(1)
closed = closing.Execute(opened)

# smooth
# smoother = sitk.MedianImageFilter()
# smoother.SetRadius([0, 0, 1])
# smooth = smoother.Execute(opened)

# gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
# gaussian.SetSigma(1)
# smooth = gaussian.Execute(opened)

# surface extraction


out = closed
out.SetSpacing(img.GetSpacing())
out.SetDirection(img.GetDirection())
out.SetOrigin(img.GetOrigin())
print("writing image")
sitk.WriteImage(out, out_img_path, True)
