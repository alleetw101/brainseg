import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

import loaddataset


def display(display_list):
    plt.figure(figsize=(9, 9))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def create_mask(pred_mask):
    pred_mask[pred_mask > 0] = 1
    pred_mask[pred_mask < 0] = 0
    return pred_mask[0]


def show_predictions(dataset, num=1, skip=0):
    for image, mask in dataset.skip(skip).take(num):
        pred_mask = model.predict(image)
        display([image[0], mask[0], create_mask(pred_mask)])


def test_ds003434(test_model, slice):
    ds00343_path = 'ds003434/003434-1-1.nii.gz'
    test_array = sitk.GetArrayFromImage(sitk.ReadImage(ds00343_path, sitk.sitkFloat32))

    # Axial
    test_array = np.transpose(test_array, [1, 2, 0])
    test_array = np.flip(test_array, axis=0)

    test_array /= np.amax(test_array)

    test_image = loaddataset.resize_256(test_array, [0, 256, 256])
    test_image = test_image[slice]

    pred_image = test_image[np.newaxis, :]

    pred_image = test_model.predict(pred_image)

    pred_image = np.squeeze(pred_image, axis=0)
    pred_image = np.squeeze(pred_image, axis=-1)

    pred_image[pred_image > 0] = 1
    pred_image[pred_image < 0] = 0

    plt.figure(figsize=(9, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(test_image)
    plt.subplot(1, 2, 2)
    plt.imshow(pred_image)
    plt.axis('off')
    plt.show()


model = tf.keras.models.load_model('SavedModels/epochmodels_210303_5/15-20210303-235240')

# args = [set_start, set_end, structures, base_raw, mni152, structures_cortex, resize, resize_shape,
# augment, augment_only_images]
evaluation_ds = tf.data.Dataset.from_generator(loaddataset.load_whole_brain_seg_data,
                                               args=[4, 5, False, True, True, False, True, [0, 256, 256], False, False],
                                               output_signature=(
                                                   tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32),
                                                   tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32)))

evaluation_ds = evaluation_ds.batch(16)

show_predictions(evaluation_ds, skip=30, num=10)

# test_ds003434(model, 140)
# model.evaluate(evaluation_ds.take(5))