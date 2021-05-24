from PIL import Image
import numpy as np
import os
import argparse
from tqdm import tqdm
import tflite_runtime.interpreter as tflite
import time
import re

parser = argparse.ArgumentParser(description='EE379K HW3 - Deploy TfLite')
parser.add_argument('--frac', default=0.05)
parser.add_argument('--epoch', default=0)
args = parser.parse_args()

frac = args.frac
epoch = args.epoch
tflite_model_name = f"pruned_filters_{frac}_{epoch}_quant.tflite"

print(f"Fraction: {frac} Epoch: {epoch} model: {tflite_model_name}")

# Get the interpreter for TensorFlow Lite model
interpreter = tflite.Interpreter(model_path=tflite_model_name)

# Very important: allocate tensor memory
interpreter.allocate_tensors()

# Get the position for inserting the input Tensor
input_details = interpreter.get_input_details()[0]
# Get the position for collecting the output prediction
output_details = interpreter.get_output_details()[0]

# Label names for CIFAR10 Dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

inf_time = 0;
corr = 0
total = 0

for filename in tqdm(os.listdir("../HW3_files/test_deployment")):
  with Image.open(os.path.join("../HW3_files/test_deployment", filename)).resize((32, 32)) as img:
    input_scale, input_zero_point = input_details["quantization"]
    img = np.float32(img) / input_scale + input_zero_point
    input_image = np.expand_dims(img / 255., axis=0).astype(input_details["dtype"])

    # Set the input tensor as the image
    interpreter.set_tensor(input_details['index'], input_image)

    # Run the actual inference
    inf_start = time.time()
    interpreter.invoke()
    inf_time += time.time()-inf_start

    # Get the output tensor
    pred_tflite = interpreter.get_tensor(output_details['index'])
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
