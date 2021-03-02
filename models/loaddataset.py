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

    plt.figure(figsize=(9, 9))

    for position in range(4 * 4):
        plt.subplot(4, 4, position + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x[50 + (position * 10)])
        plt.title(f'Slice index: {50 + (position * 10)}')
    plt.show()

    print(x.shape)


def load_whole_brain_seg_data(): # , ds_list: [str, ], num: int = None):
    datadirpath = 'dataverse_files'
    datasetpath_dic = {
        'NKI-RS': 'NKI-RS-22_volumes',
        'OASIS-TRT': 'OASIS-TRT-20_volumes',
        'MMRR': 'MMRR-21_volumes',
        'NKI-TRT': 'NKI-TRT-20_volumes'
    }

    # if ds_list[0].lower() == 'all':
    dataset_list = list(datasetpath_dic.values())
    # else:
    #     dataset_list = list(map(lambda x: datasetpath_dic[x], ds_list))
    print(dataset_list)

    for dataset in dataset_list:
        dataset_path = os.path.join(datadirpath, dataset)
        dir_list = [x for x in os.listdir(dataset_path) if dataset[:4] in x]

        # if num:
        #     dir_list.sort()
        #     dir_list = dir_list[:num]

        for directory in dir_list:
            raw_scan_file = os.path.join(dataset_path, directory, 't1weighted.MNI152.nii.gz')
            brain_scan_file = os.path.join(dataset_path, directory, 't1weighted_brain.MNI152.nii.gz')

            scan_image = sitk.GetArrayFromImage(sitk.ReadImage(raw_scan_file, sitk.sitkFloat32))
            scan_image /= np.amax(scan_image)

            mask_image = sitk.GetArrayFromImage(sitk.ReadImage(brain_scan_file, sitk.sitkFloat32))
            mask_image = np.where(0 == mask_image, mask_image, 1)

            # scan_image = resize_256(scan_image)
            # mask_image = resize_256(mask_image)

            # scan_image, mask_image = image_augmentation(scan_image, mask_image)

            scan_image = np.expand_dims(scan_image, axis=-1)
            mask_image = np.expand_dims(mask_image, axis=-1)

            for slices in range(len(scan_image)):
                yield scan_image[slices], mask_image[slices]


def load_whole_brain_seg_extra(datadirpath, num: int = None):
    # TF dataset from Extra-18_volumes subjects for evaluation

    dataset_path = os.path.join(datadirpath, 'Extra-18_volumes')
    dir_list = [x for x in os.listdir(dataset_path) if '-' in x]

    if num:
        dir_list.sort()
        dir_list = dir_list[:num]

    for directory in dir_list:
        raw_scan_file = os.path.join(dataset_path, directory, 't1weighted.nii.gz')
        brain_scan_file = os.path.join(dataset_path, directory, 't1weighted_brain.nii.gz')

        scan_image = sitk.GetArrayFromImage(sitk.ReadImage(raw_scan_file, sitk.sitkFloat32))
        scan_image /= np.amax(scan_image)

        mask_image = sitk.GetArrayFromImage(sitk.ReadImage(brain_scan_file, sitk.sitkFloat32))
        mask_image = np.where(0 == mask_image, mask_image, 1)

        scan_image = resize_256(scan_image)
        mask_image = resize_256(mask_image)

        scan_image = np.expand_dims(scan_image, axis=-1)
        mask_image = np.expand_dims(mask_image, axis=-1)

        for slices in range(len(scan_image)):
            yield scan_image[slices], mask_image[slices]


def resize_256(array) -> np.array:
    shape_dif = np.subtract([256, 256, 256], array.shape)

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


def image_augmentation(scan_image, mask_image):
    tranpose_array = [0, 1, 2]
    rng = np.random.default_rng()
    rng.shuffle(tranpose_array)

    scan = np.transpose(scan_image, tranpose_array)
    mask = np.transpose(mask_image, tranpose_array)

    if bool(rng.integers(1, size=1)[0]):
        scan = np.flip(scan, axis=1)
        mask = np.flip(mask, axis=1)

    if bool(rng.integers(1, size=1)[0]):
        scan = np.flip(scan, axis=2)
        mask = np.flip(mask, axis=2)

    return scan, mask


def testing():
    dataset_path = 'dataverse_files/MMRR-21_volumes/MMRR-21-1/t1weighted.MNI152.nii.gz'
    data_image = sitk.ReadImage(dataset_path, sitk.sitkFloat32)
    x = sitk.GetArrayFromImage(data_image)

    # x = np.where(0 == x, x, 1)
    # x = np.transpose(x, [2, 1, 0])
    # x = np.flip(x, axis=1)
    # x = np.pad(x, [(0, 0), (0, 0), (16, 16)], mode='constant', constant_values=0.0)
    # x = resize_256(x)
    # x, _ = image_augmentation(x, x)

    # x = np.transpose(x, [2,0,1])

    plt.figure(figsize=(9, 9))

    if True:
        for position in range(3 * 4):
            plt.subplot(3, 4, position + 1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x[20 + (position * 10)], cmap='gray')
            plt.title(f'Slice index: {20 + (position * 10)}')
    else:
        plt.imshow(x[100])

    plt.show()
    print(x.shape)


def display_test(dataset, samples: int = 5, start: int = 500):
    plt.figure(figsize=(9, 9))
    images = []
    for index in range(samples):
        for scan, mask in dataset.skip(start + index).take(1):
            images.append(np.squeeze(scan.numpy(), axis=-1))
            images.append(np.squeeze(mask.numpy(), axis=-1))

    for i in range(samples * 2):
        plt.subplot(samples, 2, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()


# testing()

dir_path = 'dataverse_files'
ds_gen = load_whole_brain_seg_data() # , ds_list=['all'])
# ds_gen = load_whole_brain_seg_extra(dir_path)
gen = lambda: (pair for pair in ds_gen)
# ds = load_whole_brain_seg_extra(dir_path).shuffle(2000)
#
ds = tf.data.Dataset.from_generator(load_whole_brain_seg_data, output_signature=(
    tf.TensorSpec(shape=(218, 182, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(218, 182, 1), dtype=tf.float32)))

counter = 0
for test, maks in ds:
    print(counter)
    counter += 1

# display_test(ds, samples=10)
# print(ds.cardinality())

# image_augmentation([], [])r

print('end')
# test_array = np.zeros((20, 25, 4005))
# resize_256(test_array)
