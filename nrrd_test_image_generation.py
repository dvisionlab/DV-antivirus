import numpy as np
import SimpleITK as sitk

# creazione dell'immagine
test_image_array = np.zeros((10, 10, 10))

for i in range(5):
    test_image_array[i] = -500

for i in range(5, 10):
    test_image_array[i] = -950

test_image = sitk.GetImageFromArray(test_image_array)

sitk.WriteImage(test_image, "test_image.nrrd")

print(test_image_array)


# creazione della segmentazione per distinguire polmone destro e sinistro
test_segm_array = np.zeros((10, 10, 10))

for i in range(10):
    test_segm_array[i][0:5] = 1
    test_segm_array[i][5:] = 2

test_segm = sitk.GetImageFromArray(test_segm_array)

sitk.WriteImage(test_segm, "test_segm.nrrd")

print(test_segm_array)
