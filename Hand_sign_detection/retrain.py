import coremltools as ct
import tensorflow as tf

keras_model = tf.keras.models.load_model('path_to_model.h5')
coreml_model = ct.convert(keras_model)
coreml_model.save('model.mlmodel')
