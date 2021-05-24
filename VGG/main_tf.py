import tensorflow as tf
import argparse
import time
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from models.tensorflow.vgg_tf import VGG
from models.tensorflow.mobilenet_tf import MobileNetv1
from tensorflow.keras.optimizers import Adam

# Argument parser
parser = argparse.ArgumentParser(description='EE379K HW3 - Starter TensorFlow code')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=100, help='Number of epoch to train')
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size

random_seed = 1
tf.random.set_seed(random_seed)

model = VGG()
model.summary()

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

opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
train_start = time.time()
history = model.fit(train_norm, trainY, epochs=epochs, batch_size=batch_size, validation_data=(test_norm, testY), verbose=2)
train_time = time.time() - train_start

print("tensorflow train time " + str(train_time))
print("tensorflow epochs " + str(history.history))
print("tensorflow evaluate" + str(model.evaluate(test_norm, testY)))

file = open("out_tf.txt","w")
file.write("train time " + str(train_time) + '\n')
file.write(str(history.history) + '\n')
file.write(str(model.evaluate(test_norm, testY)))
file.close()
model.save_weights('vgg_tf.ckpt')
