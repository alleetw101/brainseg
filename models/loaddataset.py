import tensorflow as tf
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os

def load_structure_seg_data():
    dataset_path = 'dataverse_files/NKI-RS-22_volumes/NKI-RS-22-2/t1weighted.nii.gz'
    data_image = sitk.ReadImage(dataset_path, sitk.sitkFloat32)
    x = sitk.GetArrayFromImage(data_image)

    # x = np.where(1000.0 <= x < 2000.0, x, x - 1000.0)
    x = np.where(2000.0 > x, x, x - 2000.0)
    x = np.where(1000.0 > x, x, x - 1000.0)

    plt.figure(figsize=(9,9))

    for position in range(4 * 4):
        plt.subplot(4, 4, position + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x[50 + (position * 10)])
        plt.title(f'Slice index: {50 + (position * 10)}')
    plt.show()

    print(x.shape)


def load_whole_seg_data(datadirpath, dataset: str, num: int = None):
    datasetpath_dic = {
        'NKI-RS': 'NKI-RS-22_volumes',
        'OASIS-TRT': 'OASIS-TRT-20_volumes',
        'MMRR': 'MMRR-21_volumes',
        'NKI-TRT': 'NKI-TRT-20_volumes',
        'Extra': 'Extra-18_volumes'
    }
    dataset_path = os.path.join(datadirpath, datasetpath_dic[dataset])
    dir_list = [x for x in os.listdir(dataset_path) if dataset in x]

    if dataset == 'NKI-RS' and 'NKI-RS-22-16' in dataset:
        dir_list.remove('NKI-RS-22-16')

    if num:
        dir_list.sort()
        dir_list = dir_list[:num]

    def oasistrt_resize(image):
        image = np.transpose(image, [2, 1, 0])
        image = np.flip(image, axis=1)
        image = np.pad(image, [(0, 0), (0, 0), (16, 16)], mode='constant', constant_values=0.0)

        return image

    first = True
    for directory in dir_list:
        raw_scan_file = os.path.join(dataset_path, directory, 't1weighted.nii.gz')
        brain_scan_file = os.path.join(dataset_path, directory, 't1weighted_brain.nii.gz')

        scan_image = sitk.ReadImage(raw_scan_file, sitk.sitkFloat32)
        scan_image = sitk.GetArrayFromImage(scan_image)
        scan_image /= np.amax(scan_image)

        mask_image = sitk.ReadImage(brain_scan_file, sitk.sitkFloat32)
        mask_image = sitk.GetArrayFromImage(mask_image)
        mask_image = np.where(0 == mask_image, mask_image, 1)

        if dataset == 'OASIS-TRT':
            scan_image = oasistrt_resize(scan_image)
            mask_image = oasistrt_resize(mask_image)

        scan_image = np.expand_dims(scan_image, axis=-1)
        mask_image = np.expand_dims(mask_image, axis=-1)
        print(scan_image.shape, mask_image.shape)

        if first:
            ds = tf.data.Dataset.from_tensor_slices((scan_image, mask_image))
        else:
            ds = ds.concatenate(tf.data.Dataset.from_tensor_slices((scan_image, mask_image)))
        first = False
        print(directory)

    return ds




def testing():
    dataset_path = 'dataverse_files/OASIS-TRT-20_volumes/OASIS-TRT-20-1/t1weighted_brain.nii.gz'
    data_image = sitk.ReadImage(dataset_path, sitk.sitkFloat32)
    x = sitk.GetArrayFromImage(data_image)

    # x = np.where(0 == x, x, 1)
    x = np.transpose(x, [2, 1, 0])
    x = np.flip(x, axis=1)
    x = np.pad(x, [(0, 0), (0, 0), (16, 16)], mode='constant', constant_values=0.0)

    plt.figure(figsize=(9, 9))

    if True:
        for position in range(3 * 4):
            plt.subplot(3, 4, position + 1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x[20 + (position * 10)])
            plt.title(f'Slice index: {20 + (position * 10)}')
    else:
        plt.imshow(x[100])

    plt.show()
    print(x.shape)

# testing()


dir_path = 'dataverse_files'
load_whole_seg_data(dir_path, 'OASIS-TRT', 1)