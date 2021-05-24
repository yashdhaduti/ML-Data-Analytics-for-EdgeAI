from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np_utils
import tensorflow_model_optimization as tfmot
from mobilenet_rm_filt_tf import MobileNetv1
from mobilenet_rm_filt_tf import remove_channel
import tensorflow as tf
import numpy as np
import time



def convert_tflite(model, name="", optim=0):
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	if optim:
		converter.optimizations = [tf.lite.Optimize.DEFAULT]
		converter.representative_dataset = representative_data_gen
		# Ensure that if any ops can't be quantized, the converter throws an error
		converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
		# Set the input and output tensors to uint8 (APIs added in r2.3)
		converter.inference_input_type = tf.uint8
		converter.inference_output_type = tf.uint8
	tflite_model = converter.convert()
	"""interpreter = tf.lite.Interpreter(model_content=tflite_model)
	input_type = interpreter.get_input_details()[0]['dtype']
	print('input: ', input_type)
	output_type = interpreter.get_output_details()[0]['dtype']
	print('output: ', output_type)"""


	with open(f'{name}.tflite', 'wb') as f:
		f.write(tflite_model)


#Prune channels by fraction amount
def channel_fraction_pruning(model, fraction):
	for layer in model.layers:
		if isinstance(layer, tf.keras.layers.Conv2D) and not isinstance(layer, tf.keras.layers.DepthwiseConv2D):
			#Calculate L1 Norm
			norms = np.sum(np.abs(layer.get_weights()[0]), axis=(0,1,2))
			#Find number of channels to prune
			number_zero = int(np.round((fraction)*norms.shape[0]))
			if number_zero > norms.shape[0]-3:
				number_zero = norms.shape[0]-3
			#find indices of channels to prune
			zero_indices = np.argpartition(norms, number_zero)[:number_zero]
			new_weights = np.array(layer.get_weights())
			#prune
			new_weights[..., zero_indices] = 0
			#set pruned weights in layer
			layer.set_weights(new_weights)

model = MobileNetv1()
print("*******************ORIGINAL MODEL**********************")
model.summary()
model.load_weights('mbnv1_tf.ckpt')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 4
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# convert from integers to floats
train_norm = X_train.astype('float32')
test_norm = X_test.astype('float32')

# normalize to range 0-1
train_norm = train_norm / 255.0
test_norm = test_norm / 255.0

# one hot encode target values
trainY = to_categorical(y_train)
testY = to_categorical(y_test)

def representative_data_gen():
	for input_value in tf.data.Dataset.from_tensor_slices(train_norm).batch(1).take(100):
    	# Model has only one input so each data point has one element.
		yield [input_value]

_, baseline_model_accuracy = model.evaluate(X_test, testY, verbose=0)
print("Baseline model accuracy " + str(baseline_model_accuracy))
convert_tflite(model, "baseline_quant", optim=1)

epochs = [15, 20, 30]
fractions = [0.85]
for fraction in fractions:
	for epoch in epochs:
		model = MobileNetv1()
		model.load_weights('mbnv1_tf.ckpt')
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		channel_fraction_pruning(model, fraction)
		removed_filters_model = remove_channel(model)
		removed_filters_model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
		print(f"\n\n\n\n\n\n\n\n *********************REMOVED_FILTERS {fraction}*****************")
		removed_filters_model.summary()
		start_time = time.time()
		removed_filters_model.fit(train_norm, trainY, batch_size=batch_size, epochs=epoch, validation_data=(test_norm, testY))
		finetune_time = time.time()-start_time
		_, pruned_model_accuracy = removed_filters_model.evaluate(test_norm, testY, verbose=0)
		print("Pruned model accuracy " + str(pruned_model_accuracy))
		convert_tflite(removed_filters_model, f"pruned_filters_{fraction}_{epoch}_quant_new1", optim=1)
		file = open("accuracy.txt","a")
		file.write(f"fraction: {fraction}, epoch: {epoch}, accuracy {pruned_model_accuracy}, training time {finetune_time} batch_size {batch_size}\n")
		file.close()