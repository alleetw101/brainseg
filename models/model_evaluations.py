import tensorflow as tf
from loaddataset import load_whole_seg_data

model = tf.keras.models.load_model('SavedModels/SavedModelunet_210301_1')

evaluation_ds = load_whole_seg_data('dataverse_files', 'OASIS-TRT', 1).batch(1)
model.evaluate(evaluation_ds)

