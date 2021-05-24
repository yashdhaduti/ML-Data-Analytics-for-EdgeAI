import numpy as np
import onnxruntime
import argparse
from tqdm import tqdm
import os
from PIL import Image
import time
import re

parser = argparse.ArgumentParser(description='EE379K HW3 - Deploy ONNX')
parser.add_argument('--impl', default="pt", help='Implementation: PyTorch or TensorFlow')
parser.add_argument('--model', default="VGG", help='Model: VGG or MobileNet')
args = parser.parse_args()

impl = args.impl
model = args.model
pt, tf, vgg, mobilenet = False, False, False, False

if impl == "tf":
    tf = True
elif impl == "pt":
    pt = True
else:
    raise Exception("Invalid imput for impl")

if model == "VGG":
    vgg = True
elif model == "mobilenet":
    mobilenet = True
else:
    raise Exception("Invalid imput for model")


onnx_model_name = "" # TODO: insert ONNX model name
if tf:
    if vgg:
        onnx_model_name = "vgg_tf.onnx"
    elif mobilenet:
        onnx_model_name = "mbnv1_tf.onnx"
elif pt:
    if vgg:
        onnx_model_name = "vgg_pt.onnx"
    elif mobilenet:
        onnx_model_name = "mbnv1_pt.onnx"



# Create Inference session using ONNX runtime
sess = onnxruntime.InferenceSession(onnx_model_name)

# Get the input name for the ONNX model
input_name = sess.get_inputs()[0].name
print("Input name  :", input_name)

# Get the shape of the input
input_shape = sess.get_inputs()[0].shape
print("Input shape :", input_shape)

# Mean and standard deviation used for PyTorch models
mean = np.array((0.4914, 0.4822, 0.4465))
std = np.array((0.2023, 0.1994, 0.2010))

# Label names for CIFAR10 Dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

inf_time = 0;
corr = 0
total = 0
# The test_deployment folder contains all 10.000 images from the testing dataset of CIFAR10 in .png format
for filename in tqdm(os.listdir("test_deployment")):
    # Take each image, one by one, and make inference
    with Image.open(os.path.join("test_deployment", filename)).resize((32, 32)) as img:
        print("Image shape:", np.float32(img).shape)

        # For PyTorch models ONLY: normalize image
        if pt:
            input_image = (np.float32(img) / 255. - mean) / std
        # For PyTorch models ONLY: Add the Batch axis in the data Tensor (C, H, W)
            input_image = np.expand_dims(np.float32(input_image), axis=0)

        # For TensorFlow models ONLY: Add the Batch axis in the data Tensor (H, W, C)
        if tf:
            input_image = np.expand_dims(np.float32(img) / 255., axis=0)
            print("Image shape after expanding size:", input_image.shape)

        # For PyTorch models ONLY: change the order from (B, H, W, C) to (B, C, H, W)
        if pt:
            input_image = input_image.transpose([0, 3, 1, 2])

        # Run inference and get the prediction for the input image
        inf_start = time.time()
        pred_onnx = sess.run(None, {input_name: input_image})[0]
        inf_time += time.time()-inf_start

        # Find the prediction with the highest probability
        top_prediction = np.argmax(pred_onnx[0])

        # Get the label of the predicted class
        pred_class = label_names[top_prediction]

        actual_class = re.sub(r'(^(.*)_)|(\.(.*))','',filename)
        if pred_class == actual_class:
            corr += 1
        total += 1
test_accuracy = corr/total
print("Test accuracy " + str(test_accuracy))
print("Inf time " + str(inf_time))
