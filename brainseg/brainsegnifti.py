import os
import numpy as np
import tensorflow as tf
import SimpleITK as sitk

from .brainsegnifti_helper import data_model_preprocess, resize_256


class BrainSegNifti:

    def __init__(self, filepath: str, precision: int):
        dtype = sitk.sitkFloat32 if precision == 32 else sitk.sitkFloat64

        sitk_image = sitk.ReadImage(filepath, dtype)

        self.nifti_array = sitk.GetArrayFromImage(sitk_image)
        self.file_name = os.path.basename(filepath)
        self.filepath = filepath
        self.reader = sitk.ImageFileReader()

    def information(self):
        print(f'--{self.file_name}--')
        print(f'Shape: {self.nifti_array.shape}')

        self.reader.SetFileName(self.filepath)
        self.reader.LoadPrivateTagsOn()

        self.reader.ReadImageInformation()

        for k in self.reader.GetMetaDataKeys():
            v = self.reader.GetMetaData(k)
            if not (v == '0' or v == ''):
                print(f"({k}) == \"{v}\"")

    def segment_nifti(self, model_path):
        model = tf.keras.models.load_model(model_path)

        data = data_model_preprocess(self.nifti_array, self.reader)
        self.segmented_array = []

        counter = 1
        for slice in data:
            pred_image = model.predict(slice[np.newaxis, :])

            pred_image = np.squeeze(pred_image, axis=0)
            pred_image = np.squeeze(pred_image, axis=-1)

            pred_image[pred_image > 0] = 1
            pred_image[pred_image < 0] = 0

            self.segmented_array.append(pred_image)

            counter += 1
            if counter % 50 == 0:
                print(counter, ' slices segmented')

        reshaped_array = resize_256(self.nifti_array, [0, 256, 256])

        for z in range(len(self.nifti_array)):
            for y in range(len(self.nifti_array[z])):
                for x in range(len(self.nifti_array[z][y])):
                    if self.segmented_array[z][y][x] == 1:
                        self.segmented_array[z][y][x] = reshaped_array[z][y][x]

    def write_nifti(self, out_filepath):
        if self.segmented_array:
            output = sitk.GetImageFromArray(self.segmented_array)
        else:
            output = sitk.GetImageFromArray(self.nifti_array)

        output_path = os.path.join(os.getcwd(), out_filepath)

        sitk.WriteImage(output, output_path)

