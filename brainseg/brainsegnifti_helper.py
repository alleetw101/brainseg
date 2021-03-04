import numpy as np
from niftiprocessing import niftihelper


def data_model_preprocess(data_array, reader):
    output_array = data_array
    output_array /= np.amax(output_array)

    try:
        output_array = niftihelper.align(output_array, reader)
    except RuntimeError:
        print('Unable to align due to invalid orientation xyz in metadata')

    output_array = resize_256(output_array, [0, 256, 256])

    return output_array


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