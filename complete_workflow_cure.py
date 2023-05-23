import os
import SimpleITK as sitk
from io import BytesIO
from pydicom import dcmread
from complete_workflow import (
    do_prediction,
    label_mask,
    maskToCSV,
    examine_threshold,
    compute_stats,
)


def complete_workflow(image, thresholds=[-1000, -920, -770]):
    # workflow for integration with cure

    # create temporary folder
    temp_path = os.path.join(os.getcwd(), "temp/")
    os.makedirs(temp_path, exist_ok=True)

    # run segmentation with lungmask
    segmentation = do_prediction(image, force_cpu=True, write_image=True)
    segmentation_arr = sitk.GetArrayFromImage(segmentation)

    # extract only values inside the target palette (thresholds)
    perfusion_mask, perf_zones = label_mask(image, segmentation_arr, thresholds)

    # generate the histogram
    maskToCSV(segmentation, image, thresholds, temp_path)
    examine_threshold(temp_path + "hist_output.csv", thresholds)

    # compute volumes
    pdf_file = compute_stats(
        perfusion_mask,
        perf_zones,
        False,
        image.GetSpacing(),
        image.GetSize(),
        temp_path,
    )

    return pdf_file


# run example
# dicom2nrrd not needed
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames("./N2784744/DICOM/MEDIASTINO")
reader.SetFileNames(dicom_names)
img = reader.Execute()

complete_workflow(img)

# estrarre l'ID della serie per scriverlo nel pdf
