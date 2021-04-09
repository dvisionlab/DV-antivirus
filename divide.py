import SimpleITK as sitk
import numpy as np
import argparse
import os

# divide image with same serie id but different acquisition number in the same folder
# write img_1 and img_2 in output

if __name__ == "__main__":
    # init arg parser

    parser = argparse.ArgumentParser(
        description=""
    )

    parser.add_argument(
        "--image",
        action="store",
        default=None,
        help="the target image",
    )

    parser.add_argument(
        "--dicomdir",
        action="store",
        default=None,
        help="the dicom folder of target image",
    )

    parser.add_argument(
        "--mask",
        action="store",
        default=None,
        help="the mask image",
    )

    args = parser.parse_args()

    # image = sitk.ReadImage(args.image)
    # mask = sitk.ReadImage(args.mask)

    # crop = sitk.CropImageFilter()
    # crop.SetUpperBoundaryCropSize([0,0,23])
    # crop.SetLowerBoundaryCropSize([0,0,0])
    # res = crop.Execute(image)


    # res = image[:, :, 200:423]

    # sitk.WriteImage(res, "./res.nrrd", True)

    # img = sitk.GetImageFromArray(image)
    # msk = sitk.GetImageFromArray(mask)
    
    # print('done conversion')

    # out_img = img[msk]

    # print(out_img)

    def getImageSeriesId(file_name, series_list, desc_list):
        print("Reading image...")
        # A file name that belongs to the series we want to read

        # Read the file's meta-information without reading bulk pixel data
        # print('Reading image...')
        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(file_name)

        try:
            file_reader.ReadImageInformation()
        except:
            print("ERROR while reading: ", file_name)
            print("SKIP file")
            return

        # Get the sorted file names, opens all files in the directory and reads the meta-information
        # without reading the bulk pixel data
        series_ID = file_reader.GetMetaData("0020|000e")
        description = file_reader.GetMetaData("0008|103e")
        # print('seriesId', series_ID, '\t\t descr', description)

        if series_ID not in series_list:
            series_list.append(series_ID)
            desc_list.append(description)

        return series_ID

    def divideByAcqNumber(file_names):
        seriesObject = [[],[]]

        for file_name in file_names:
            file_reader = sitk.ImageFileReader()
            file_reader.SetFileName(file_name)

            try:
                file_reader.ReadImageInformation()
            except:
                print("ERROR while reading: ", file_name)
                print("SKIP file")
                return

            acqNumber = file_reader.GetMetaData("0020|0012")
            print(acqNumber, file_name)
            seriesObject[int(acqNumber)-1].append(file_name)

        return seriesObject

    path_image = args.dicomdir

    for (root, dirs, files) in os.walk(path_image):
        series_id = getImageSeriesId(os.path.join(root, files[0]), [], [])

        sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            path_image, series_id
        )

    obj = divideByAcqNumber(sorted_file_names)

    image1 = sitk.ReadImage(obj[0])
    image2 = sitk.ReadImage(obj[1])

    sitk.WriteImage(image1, "./image1.nrrd", True)
    sitk.WriteImage(image2, "./image2.nrrd", True)


