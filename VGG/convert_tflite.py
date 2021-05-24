from models.tensorflow.mobilenet_tf import MobileNetv1 as MBN_tf
from models.tensorflow.vgg_tf import VGG as VGG_tf
import tensorflow as tf

model = MBN_tf()
model.load_weights("mbnv1_tf.ckpt")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(f'mbnv1_tf.tflite', 'wb') as f:
	f.write(tflite_model)

model = VGG_tf()
model.load_weights("vgg_tf.ckpt")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(f'vgg_tf.tflite', 'wb') as f:
	f.write(tflite_model)