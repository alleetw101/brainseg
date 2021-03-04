import tensorflow as tf
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os


def load_whole_brain_seg_data(set_start: int, set_end: bool, structures: bool, base_raw: bool,
                              mni152: bool, structures_cortex: bool,
                              resize: bool, resize_shape: [int, int, int],
                              augment: bool, augment_only_images: bool):
    datadirpath = 'dataverse_files'
    datasetpath_dic = {
        'NKI-RS': 'NKI-RS-22_volumes',
        'OASIS-TRT': 'OASIS-TRT-20_volumes',
        'MMRR': 'MMRR-21_volumes',
        'NKI-TRT': 'NKI-TRT-20_volumes',
        'Extra': 'Extra-18_volumes'
    }

    if set_start == 0 and set_end == 0:
        dataset_list = list(datasetpath_dic.values())
    else:
        dataset_list = list(datasetpath_dic.values())[set_start:set_end]

    scan_file_name = 't1weighted'
    if not base_raw:
        scan_file_name = scan_file_name + '_brain'

    if structures:
        mask_file_name = 'labels.DKT31.manual+aseg'
    else:
        mask_file_name = 't1weighted_brain'

    if mni152:
        scan_file_name = scan_file_name + '.MNI152'
        mask_file_name = mask_file_name + '.MNI152'

    scan_file_name = scan_file_name + '.nii.gz'
    mask_file_name = mask_file_name + '.nii.gz'

    for dataset in dataset_list:
        dataset_path = os.path.join(datadirpath, dataset)
        dir_list = [x for x in os.listdir(dataset_path) if '-' in x]

        for directory in dir_list:
            raw_scan_file = os.path.join(dataset_path, directory, scan_file_name)
            brain_scan_file = os.path.join(dataset_path, directory, mask_file_name)

            scan_image = sitk.GetArrayFromImage(sitk.ReadImage(raw_scan_file, sitk.sitkFloat32))
            scan_image /= np.amax(scan_image)

            mask_image = sitk.GetArrayFromImage(sitk.ReadImage(brain_scan_file, sitk.sitkFloat32))

            if not structures:
                mask_image = np.where(0 == mask_image, mask_image, 1)

            if structures and not structures_cortex:
                mask_image = np.where(2000.0 > mask_image, mask_image, mask_image - 1000.0)
                mask_image = np.where(1000.0 > mask_image, mask_image, mask_image - 1000.0)

            if resize:
                scan_image = resize_256(scan_image, resize_shape)
                mask_image = resize_256(mask_image, resize_shape)

            if augment:
                scan_image, mask_image = image_augmentation(scan_image, mask_image, augment_only_images)

            scan_image = np.expand_dims(scan_image, axis=-1)
            mask_image = np.expand_dims(mask_image, axis=-1)

            for slices in range(len(scan_image)):
                yield scan_image[slices], mask_image[slices]


def resize_256(array, resize_shape: [int, int, int]) -> np.array:
    for dim in range(len(resize_shape)):
        if resize_shape[dim] == 0:
            resize_shape[dim] = array.shape[dim]
    shape_dif = np.subtract(resize_shape, array.shape)

    # Padding
    pad_list = []
    for dif in shape_dif:
        if dif <= 0:
            pad_list.append([0, 0])
        elif dif % 2 != 0:
            pad = int((dif - 1) / 2)
            pad_list.append([pad, pad + 1])
        else:
            pad = int(dif / 2)
            pad_list.append([pad, pad])

    output_array = np.pad(array, pad_list, mode='constant', constant_values=0.0)

    # Cropping
    cl = []  # crop_list
    for dif in shape_dif:
        if dif >= 0:
            cl.append([0, 256])
        elif dif % 2 != 0:
            crop = abs(int((dif + 1) / 2))
            cl.append([crop, crop + 256])
        else:
            crop = abs(int(dif / 2))
            cl.append([crop, crop + 256])

    output_array = output_array[cl[0][0]:cl[0][1], cl[1][0]:cl[1][1], cl[2][0]:cl[2][1]]

    return output_array


def image_augmentation(scan_image, mask_image, augment_only_images=True):
    rng = np.random.default_rng()

    if augment_only_images:
        scan = scan_image
        mask = mask_image
        for index in range(len(scan_image)):
            if bool(rng.integers(2, size=1)[0]):
                scan[index] = np.transpose(scan[index], [1, 0])
                mask[index] = np.transpose(mask[index], [1, 0])
    else:
        tranpose_array = [0, 1, 2]
        rng.shuffle(tranpose_array)

        scan = np.transpose(scan_image, tranpose_array)
        mask = np.transpose(mask_image, tranpose_array)

    for index in range(len(scan_image)):
        if bool(rng.integers(2, size=1)[0]):
            scan[index] = np.flip(scan[index], axis=0)
            mask[index] = np.flip(mask[index], axis=0)

        if bool(rng.integers(2, size=1)[0]):
            scan[index] = np.flip(scan[index], axis=1)
            mask[index] = np.flip(mask[index], axis=1)

    return scan, mask


def display_test(dataset, samples: int = 5):
    plt.figure(figsize=(9, 9))
    images = []
    for scan, mask in dataset.take(samples):
        images.append(np.squeeze(scan[0], axis=-1))
        images.append(np.squeeze(mask[0], axis=-1))

    for i in range(samples * 2):
        plt.subplot(samples, 2, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()


# Sample Dataset from Generator
# args = [set_start, set_end, structures, base_raw, mni152, structures_cortex, resize, resize_shape, augment, augment_only_images]
# ds = tf.data.Dataset.from_generator(load_whole_brain_seg_data,
#                                     args=[0, 5, True, False, True, False, True, [0, 256, 256], True, True],
#                                     output_signature=(
#                                         tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32),
#                                         tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32)))
# ds = ds.batch(16)
# display_test(ds)

# Structures, structures_cortex(false) = Highest 59 labels (Common 57-58)
