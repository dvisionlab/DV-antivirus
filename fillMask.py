import SimpleITK as sitk
import numpy as np
import argparse

if __name__ == "__main__":
    # init arg parser

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--mask",
        action="store",
        default=None,
        help="the binary lung mask",
    )

    parser.add_argument(
        "--image",
        action="store",
        default=None,
        help="the original image",
    )

    args = parser.parse_args()

    mask = sitk.ReadImage(args.mask)
    mask_arr = sitk.GetArrayFromImage(mask)

    out_mask = mask

    dims = mask.GetSize()

    def fill_line(k, j, arr_in, arr_out):
        fill = False
        for i in range(arr_in.shape[2]):
            px0 = arr_in[k, j, i]
            px1 = arr_in[k, j, i + 1]
            if px0 == 1 and px1 == 0:
                fill = True
            if fill:
                arr_out[k, j, i] = 3
            if fill and px0 == 0 and px1 == 2:  # right lung value is 2
                break

    llv = 1
    rlv = 2

    out_arr = np.copy(mask_arr)

    for k in range(mask_arr.shape[0]):
        for j in range(mask_arr.shape[1]):
            row = mask_arr[k, j, :]
            if (llv in row) and (rlv in row):
                print(k, j)
                fill_line(k, j, mask_arr, out_arr)

    # binarize!
    out_arr[out_arr == 2] = 1
    out_arr[out_arr == 3] = 1

    spacing = mask.GetSpacing()
    direction = mask.GetDirection()
    origin = mask.GetOrigin()

    out_mask = sitk.GetImageFromArray(out_arr)
    out_mask.SetSpacing(spacing)
    out_mask.SetDirection(direction)
    out_mask.SetOrigin(origin)

    # close little holes
    filler = sitk.BinaryFillholeImageFilter()
    filler.SetForegroundValue(1)
    filled_mask = filler.Execute(out_mask)

    sitk.WriteImage(filled_mask, "./filled_mask.nrrd")

    if args.image:
        img = sitk.ReadImage(args.image)
        img_arr = sitk.GetArrayFromImage(img)
        o_arr = np.zeros(img_arr.shape)
        o_arr = o_arr - 1500  # background value
        o_arr[out_arr == 1] = img_arr[out_arr == 1]
        out_img = sitk.GetImageFromArray(o_arr)
        out_img.SetDirection(img.GetDirection())
        out_img.SetSpacing(img.GetSpacing())
        out_img.SetOrigin(img.GetOrigin())
        sitk.WriteImage(out_img, "./filled_lungs.nrrd")
