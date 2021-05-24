from PIL import Image
import numpy as np
import os
import argparse
from tqdm import tqdm
import tflite_runtime.interpreter as tflite
import time
import re

parser = argparse.ArgumentParser(description='EE379K HW3 - Deploy TfLite')
parser.add_argument('--model', default="VGG", help='Model: VGG or MobileNet')
args = parser.parse_args()

model = args.model
vgg, mobilenet = False, False

if model == "VGG":
    vgg = True
elif model == "mobilenet":
    mobilenet = True
else:
    raise Exception("Invalid imput for model")

if vgg:
    tflite_model_name = "vgg_tf.tflite"
elif mobilenet:
    tflite_model_name = "mbnv1_tf.tflite"


# Get the interpreter for TensorFlow Lite model
interpreter = tflite.Interpreter(model_path=tflite_model_name)

# Very important: allocate tensor memory
interpreter.allocate_tensors()

# Get the position for inserting the input Tensor
input_details = interpreter.get_input_details()
# Get the position for collecting the output prediction
output_details = interpreter.get_output_details()

# Label names for CIFAR10 Dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

inf_time = 0;
corr = 0
total = 0

for filename in tqdm(os.listdir("HW3_files/test_deployment")):
  with Image.open(os.path.join("HW3_files/test_deployment", filename)).resize((32, 32)) as img:
    input_image = np.expand_dims(np.float32(img), axis=0)

    # Set the input tensor as the image
    interpreter.set_tensor(input_details[0]['index'], input_image)

    # Run the actual inference
    inf_start = time.time()
    interpreter.invoke()
    inf_time += time.time()-inf_start

    # Get the output tensor
    pred_tflite = interpreter.get_tensor(output_details[0]['index'])

    # Find the prediction with the highest probability
    top_prediction = np.argmax(pred_tflite[0])

    # Get the label of the predicted class
    pred_class = label_names[top_prediction]
    actual_class = re.sub(r'(^(.*)_)|(\.(.*))','',filename)
    if pred_class == actual_class:
        corr += 1
    total += 1
test_accuracy = corr/total
print("Test accuracy " + str(test_accuracy))
print("Inf time " + str(inf_time))
