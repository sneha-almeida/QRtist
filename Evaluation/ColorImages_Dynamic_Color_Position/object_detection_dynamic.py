# -*- coding: utf-8 -*-
import random
import string
import qrcode

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Concatenate, Conv2D, UpSampling2D, Reshape, Dense, LeakyReLU, ZeroPadding2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.cluster import KMeans

import cv2
import os
from PIL import Image


import tensorflow_datasets as tfds
from matplotlib import pyplot

"""### Function to Load Cats Vs Dogs Dataset"""

n = 10
def load_real_samples(target_shape, num_samples_to_take = n):
    constant_label = 1
    caltech_builder = tfds.builder('cats_vs_dogs')
    caltech_builder.download_and_prepare()
    dataset = caltech_builder.as_dataset(split='train')

    shuffled_dataset = dataset.shuffle(buffer_size=num_samples_to_take)
    subset_dataset = shuffled_dataset.take(num_samples_to_take)

    num_images_in_subset = tf.data.experimental.cardinality(subset_dataset).numpy()
    #print(f"Number of images in subset_dataset_with_labels: {num_images_in_subset}")
    images, labels = [], []
    for example in subset_dataset:
        example['image'] = tf.image.resize(example['image'], target_shape)
        images.append(example['image'].numpy())
        labels.append(constant_label)

    images_array = np.array(images)
    labels_array = np.array(labels, dtype=np.int64)

    num_images_in_subset = len(images_array)
    print(f"Number of images in subset_dataset_with_labels: {num_images_in_subset}")

    return images_array, labels_array

"""### Create QR Codes"""

LETTERS = string.ascii_lowercase + ' ' # letters and space
EOI = '#' # end of input
MAX_SIZE = 11 # max input/output size
ALL_CHAR_CLASSES = LETTERS + EOI # all available classes



def find_most_textured_region(image, region_size=(128, 128), n_colors=4):
    # Check the dimensions of the input image
    assert image.shape == (256, 256, 3), "Image must be of shape (256, 256, 3)"

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure the grayscale image is in 8-bit format (CV_8U)
    gray_image = gray_image.astype(np.uint8)

    # Apply the Laplacian filter to find edges (high texture regions)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

    # Take the absolute value of the Laplacian (since it can be negative)
    abs_laplacian = np.abs(laplacian)

    # Use a sliding window to find the region with the most texture
    window_size = region_size[0]
    max_texture_value = -1
    best_position = (0, 0)

    # Scan through the image in (128, 128) windows to find the area with the most texture
    for i in range(image.shape[0] - window_size + 1):
        for j in range(image.shape[1] - window_size + 1):
            window = abs_laplacian[i:i+window_size, j:j+window_size]
            texture_value = np.sum(window)

            if texture_value > max_texture_value:
                max_texture_value = texture_value
                best_position = (i, j)

    # Get the coordinates of the most textured region
    top_left_y, top_left_x = best_position
    textured_region = image[top_left_y:top_left_y+window_size, top_left_x:top_left_x+window_size]

    # Detect the 4 most prominent colors in the textured region using KMeans
    def detect_prominent_colors(region, n_colors):
        # Reshape the image into (num_pixels, 3) for clustering
        pixels = region.reshape(-1, 3)

        # Apply KMeans to find the most prominent colors
        kmeans = KMeans(n_clusters=n_colors)
        kmeans.fit(pixels)

        # Get the cluster centers (the colors) and labels (which cluster each pixel belongs to)
        prominent_colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_

        # Count the number of pixels in each cluster to determine the largest color region
        unique, counts = np.unique(labels, return_counts=True)
        largest_color_index = unique[np.argmax(counts)]
        largest_color = prominent_colors[largest_color_index]

        return prominent_colors, largest_color

    # Get the 4 most prominent colors and the color that covers the maximum area
    prominent_colors, largest_color = detect_prominent_colors(textured_region, n_colors)

    # Convert RGB colors to LAB space
    def rgb_to_lab(color):
        color_rgb = np.uint8([[color]])  # Convert color to the right shape for cv2
        color_lab = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2LAB)[0][0]
        return color_lab

    # Convert all prominent colors and the largest color to LAB
    largest_color_lab = rgb_to_lab(largest_color)
    prominent_colors_lab = [rgb_to_lab(color) for color in prominent_colors]

    # Calculate the Euclidean distance between the largest color and other colors in LAB space
    def euclidean_distance(color1, color2):
        return np.linalg.norm(color1 - color2)

    max_distance = -1
    furthest_color_lab = None
    furthest_color_rgb = None

    # Find the color that is at the maximum distance from the largest color in LAB space
    for color_lab, color_rgb in zip(prominent_colors_lab, prominent_colors):
        distance = euclidean_distance(largest_color_lab, color_lab)
        if distance > max_distance:
            max_distance = distance
            furthest_color_lab = color_lab
            furthest_color_rgb = color_rgb

    # Convert the furthest LAB color back to RGB
    def lab_to_rgb(color_lab):
        color_lab = np.uint8([[color_lab]])  # Convert color to the right shape for cv2
        color_rgb = cv2.cvtColor(color_lab, cv2.COLOR_LAB2RGB)[0][0]
        return color_rgb

    # Convert furthest color from LAB to RGB
    furthest_color_rgb_converted = lab_to_rgb(furthest_color_lab)

    # Ensure colors are returned as tuples
    largest_color_rgb = tuple(largest_color)
    furthest_color_rgb_converted = tuple(furthest_color_rgb_converted)

    # Return the coordinates of the most textured region along with prominent colors and furthest color
    return (top_left_y, top_left_x), prominent_colors, largest_color_rgb, furthest_color_rgb_converted



# define a function for a QR code generation
# box_size is an option to manage size of an output image
def make_qr(custom_fill_color, custom_back_color, text, box_size=12, target_size=(128, 128)):
    qr = qrcode.QRCode(
        version=1,
        box_size=box_size,
        border=0
    )
    qr.add_data(text)
    qr.make(fit=True)
    qr_image = qr.make_image(fill_color=custom_fill_color, back_color=custom_back_color).convert('RGB')

    # Resize the QR code image to the target size
    resized_image = qr_image.resize(target_size)

    # Convert RGBA to RGB
    #rgb_image = resized_image.convert('RGB')

    return np.asarray(resized_image, dtype='uint8')



def pad_qr_code(qr_code, coordinates):
    # Unpack the coordinates tuple
    top_left_y, top_left_x = coordinates

    # Check that the QR code has the correct shape
    assert qr_code.shape == (128, 128, 3), "QR code must be of shape (128, 128, 3)"

    # Create a blank image of size (256, 256, 3) filled with zeros
    padded_image = np.zeros((256, 256, 3), dtype=np.uint8)

    # Calculate the bottom-right corner where the QR code will end
    bottom_right_y = top_left_y + qr_code.shape[0]
    bottom_right_x = top_left_x + qr_code.shape[1]

    # Ensure that the QR code fits within the bounds of the (256, 256) image
    assert bottom_right_y <= 256 and bottom_right_x <= 256, "QR code exceeds image boundaries."

    # Place the QR code in the specified region
    padded_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = qr_code

    return padded_image

# Example usage:
# Assuming 'qr_code' is the QR code image of shape (128, 128, 3)
# and coordinates is a tuple (top_left_y, top_left_x) from the previous function
# new_image_with_qr = pad_qr_code(qr_code, (top_left_y, top_left_x))

# 'new_image_with_qr' will now contain the (256, 256, 3) image with the QR code placed at the textured region.


# a function to generate a train data set
# output: (numpy array of images, list of corresponding texts)
def generate_dataset(host_images, n_of_samples, min_size = 1, max_size = 11):
    data = []
    labels = []
    report_step = int(n_of_samples * .1)
    report = report_step
    print("Generating")
    for i in range(n_of_samples):
        if i == report:
            #print("Done:", report / report_step * 10, "%")
            report = report + report_step
        #select size of the string to be embedded in a QR Code randonly between 1 and 11
        size = random.randint(min_size, max_size)
        #generate the string using random letters and space
        s = ''.join(random.choice(LETTERS) for i in range(size))
        #find the most textured region in the host image and the custom fill color and custom back color for the QR code
        (top_left_y, top_left_x), prominent_colors, largest_color, furthest_color_rgb_converted = find_most_textured_region(host_images[i])
        #make the QR Code
        img = make_qr(largest_color, furthest_color_rgb_converted, s)
        #ensure the size is 256 x 256 pixels
        #assert img.size == (256,256)
        qr = np.asarray(img, dtype='float')
        #print('qr.shape: ', qr.shape)
        #resize QR to match the host image
        padded_qr = pad_qr_code(qr, (top_left_y, top_left_x))
        #Append the QR code to the data list
        data.append(padded_qr)
        #give adding to the QR
        #Append the String of random size generated using random letters to the label list
        labels.append(s)
    print("Done:", "100", "%")
    return (np.asarray(data), labels)

"""### Define YOLO function"""

# Load YOLOv3 model and class names
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function to perform object detection using YOLOv3
def detect_objects_yolo(image, net, output_layers):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                box = detection[0:4] * np.array([width, height, width, height])
                center_x, center_y, w, h = box.astype("int")
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids, len(class_ids)

"""### Calculate n_host"""

n_host = 0
# Load images
target_shape = (256, 256)
images, _ = load_real_samples(target_shape) ######################### Give Composite Images Here

# Get the output layers from YOLO
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Perform object detection on the loaded images
for i, image in enumerate(images):
    boxes, confidences, class_ids, count = detect_objects_yolo(image, net, output_layers)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(f"Image {i+1} has {len(indices)} objects detected.")
    n_host = n_host + len(indices)
    if len(indices) > 0:
        for j in indices.flatten():
            x, y, w, h = boxes[j]
            label = str(classes[class_ids[j]])
            confidence = confidences[j]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image with bounding boxes
    cv2_imshow(image)
print('n_host = ', n_host)

"""### Calculate n_composite"""

n_composite = 0
# Get the output layers from YOLO
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Perform object detection on the loaded images
for i, image in enumerate(composite_images):
    boxes, confidences, class_ids, count = detect_objects_yolo(image, net, output_layers)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(f"Image {i+1} has {len(indices)} objects detected.")
    n_composite = n_composite + len(indices)
    if len(indices) > 0:
        for j in indices.flatten():
            x, y, w, h = boxes[j]
            label = str(classes[class_ids[j]])
            confidence = confidences[j]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image with bounding boxes
    cv2_imshow(image)
print('n_composite = ', n_composite)

"""### Calculate ODR"""

def compute_odr(N_host, N_composite):
    numerator = np.sum(N_host)
    denominator = np.sum(N_composite)
    odr = numerator / denominator
    return odr

# Calculate and print ODR
odr = compute_odr(N_host, N_composite)
print(f'Object Detection Ratio (ODR): {odr} for {n} images')

"""### Create inputs for generator"""

def make_labels_for_position(labels, pos):
    chars = list(map(lambda x: x[pos] if pos < len(x) else EOI, labels)) # either a letter or EOI
    return list(map(lambda x: [ALL_CHAR_CLASSES.index(x)], chars)) # all classes are indexed [0..len(ALL_CHAR_CLASSES))

def get_training_label_sizes(training_labels):
  training_label_sizes = list(map(lambda x: [len(x) - 1], training_labels))
  return training_label_sizes

#Host Images
host_images_array, discriminator_1_labels_array = load_real_samples(target_shape = (256, 256), num_samples_to_take = 10)

#QR Codes
qr_codes, qr_code_labels = generate_dataset(host_images_array, n_of_samples = 10)
qr_code_labels = [list(string) for string in qr_code_labels]
evaluation_label_sizes = get_training_label_sizes(qr_code_labels)
evaluation_labels_char0 = make_labels_for_position(qr_code_labels, 0)
evaluation_labels_char1 = make_labels_for_position(qr_code_labels, 1)
evaluation_labels_char2 = make_labels_for_position(qr_code_labels, 2)
evaluation_labels_char3 = make_labels_for_position(qr_code_labels, 3)
evaluation_labels_char4 = make_labels_for_position(qr_code_labels, 4)
evaluation_labels_char5 = make_labels_for_position(qr_code_labels, 5)
evaluation_labels_char6 = make_labels_for_position(qr_code_labels, 6)
evaluation_labels_char7 = make_labels_for_position(qr_code_labels, 7)
evaluation_labels_char8 = make_labels_for_position(qr_code_labels, 8)
evaluation_labels_char9 = make_labels_for_position(qr_code_labels, 9)
evaluation_labels_char10 = make_labels_for_position(qr_code_labels, 10)

evaluation_labels = [
    np.asarray(evaluation_label_sizes),        # Size labels
    np.asarray(evaluation_labels_char0),       # Character 0 labels
    np.asarray(evaluation_labels_char1),       # Character 1 labels
    np.asarray(evaluation_labels_char2),       # Character 2 labels
    np.asarray(evaluation_labels_char3),       # and so on...
    np.asarray(evaluation_labels_char4),
    np.asarray(evaluation_labels_char5),
    np.asarray(evaluation_labels_char6),
    np.asarray(evaluation_labels_char7),
    np.asarray(evaluation_labels_char8),
    np.asarray(evaluation_labels_char9),
    np.asarray(evaluation_labels_char10)]

"""### Load the Generator Model"""

generator_model = load_model('generator_model.h5')

"""### Generate Composite Images"""

composite_images = generator_model.predict([host_images_array, qr_codes], data_size = 10)

"""### Save the inputs

#### Snapshots
"""

def save_plot(x_fake, X_real, qr_codes, epoch, n):
    print('x_fake.shape: ', x_fake.shape)
    print('X_real.shape: ', X_real.shape)
    qr_codes = qr_codes.reshape(qr_codes.shape[0], qr_codes.shape[1], qr_codes.shape[2], 1)
    print('qr_codes.shape: ', qr_codes.shape)
    fig, axs = plt.subplots(n, 3, figsize=(10, 10))
    for i in range(n):  # Loop through rows
        axs[i, 0].imshow(x_fake[i])  # Plot image from the first list
        axs[i, 0].axis('off')  # Turn off axis labels
        axs[i, 1].imshow(X_real[i].astype('uint8'))  # Plot image from the second list
        axs[i, 1].axis('off')  # Turn off axis labels
        axs[i, 2].imshow(qr_codes[i])  # Plot image from the third list
        axs[i, 2].axis('off')  # Turn off axis labels
        img_file = 'flower_blended/'+str(epoch)+'_'+str(i)+'.png'
        plt.imsave(img_file, x_fake[i].astype('float32'))

    # Add titles to subplots (optional)
    axs[0, 0].set_title('GAN Images')
    axs[0, 1].set_title('Real Images')
    axs[0, 2].set_title('QR Codes')

    # Adjust layout and display the plot
    #pyplot.tight_layout()
    filename = 'flower_blended_combined_output/mnist_e%03d.png' % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()

save_plot(composite_images, host_images_array, qr_codes, epoch = 1, n = composite_images.shape[0])

"""### Define YOLO function"""

# Install necessary libraries
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import cv2
from google.colab.patches import cv2_imshow

# Download YOLOv3 files if they are not already available
!wget -q https://pjreddie.com/media/files/yolov3.weights -O yolov3.weights
!wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -O yolov3.cfg
!wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -O coco.names

# Load YOLOv3 model and class names
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function to perform object detection using YOLOv3
def detect_objects_yolo(image, net, output_layers):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                box = detection[0:4] * np.array([width, height, width, height])
                center_x, center_y, w, h = box.astype("int")
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids, len(class_ids)

"""### Calculate n_host"""

n_host = 0
# Load images
target_shape = (256, 256)
images, _ = load_real_samples(target_shape) ######################### Give Composite Images Here

# Get the output layers from YOLO
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Perform object detection on the loaded images
for i, image in enumerate(images):
    boxes, confidences, class_ids, count = detect_objects_yolo(image, net, output_layers)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(f"Image {i+1} has {len(indices)} objects detected.")
    n_host = n_host + len(indices)
    if len(indices) > 0:
        for j in indices.flatten():
            x, y, w, h = boxes[j]
            label = str(classes[class_ids[j]])
            confidence = confidences[j]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image with bounding boxes
    cv2_imshow(image)
print('n_host = ', n_host)

"""### Calculate n_composite"""

n_composite = 0
# Get the output layers from YOLO
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Perform object detection on the loaded images
for i, image in enumerate(composite_images):
    boxes, confidences, class_ids, count = detect_objects_yolo(image, net, output_layers)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(f"Image {i+1} has {len(indices)} objects detected.")
    n_composite = n_composite + len(indices)
    if len(indices) > 0:
        for j in indices.flatten():
            x, y, w, h = boxes[j]
            label = str(classes[class_ids[j]])
            confidence = confidences[j]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image with bounding boxes
    cv2_imshow(image)
print('n_composite = ', n_composite)

"""### Calculate ODR"""

def compute_odr(N_host, N_composite):
    numerator = np.sum(N_host)
    denominator = np.sum(N_composite)
    odr = numerator / denominator
    return odr

# Calculate and print ODR
odr = compute_odr(n_host, n_composite)
print(f'Object Detection Ratio (ODR): {odr} for {n} images')

"""# New Code"""



"""# Color Images - Dynamic

## Cats Vs Dogs

### Imports
"""

pip install qrcode

import random
import string
import qrcode

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import cv2
import numpy as np
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import random

# the next two lines fix a runtime bug on mac os
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow_datasets as tfds
#import tfds_nightly as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import Reshape
import keras

from keras.layers import Dense, LeakyReLU

from matplotlib import pyplot

from keras.models import load_model

import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Conv2D, Reshape, ZeroPadding2D
from tensorflow.keras.models import Model

from tensorflow.keras.layers import ZeroPadding2D

import cv2
import numpy as np
from sklearn.cluster import KMeans

"""### Function to Load MNIST Dataset"""

def load_real_samples(target_shape, num_samples_to_take = 500):
    constant_label = 1
    caltech_builder = tfds.builder('cats_vs_dogs')
    caltech_builder.download_and_prepare()
    dataset = caltech_builder.as_dataset(split='train')

    shuffled_dataset = dataset.shuffle(buffer_size=num_samples_to_take)
    subset_dataset = shuffled_dataset.take(num_samples_to_take)

    num_images_in_subset = tf.data.experimental.cardinality(subset_dataset).numpy()
    #print(f"Number of images in subset_dataset_with_labels: {num_images_in_subset}")
    images, labels = [], []
    for example in subset_dataset:
        example['image'] = tf.image.resize(example['image'], target_shape)
        images.append(example['image'].numpy())
        labels.append(constant_label)

    images_array = np.array(images)
    labels_array = np.array(labels, dtype=np.int64)

    num_images_in_subset = len(images_array)
    print(f"Number of images in subset_dataset_with_labels: {num_images_in_subset}")

    return images_array, labels_array

"""### Create QR Codes"""

LETTERS = string.ascii_lowercase + ' ' # letters and space
EOI = '#' # end of input
MAX_SIZE = 11 # max input/output size
ALL_CHAR_CLASSES = LETTERS + EOI # all available classes





def find_most_textured_region(image, region_size=(128, 128), n_colors=4):
    # Check the dimensions of the input image
    assert image.shape == (256, 256, 3), "Image must be of shape (256, 256, 3)"

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure the grayscale image is in 8-bit format (CV_8U)
    gray_image = gray_image.astype(np.uint8)

    # Apply the Laplacian filter to find edges (high texture regions)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

    # Take the absolute value of the Laplacian (since it can be negative)
    abs_laplacian = np.abs(laplacian)

    # Use a sliding window to find the region with the most texture
    window_size = region_size[0]
    max_texture_value = -1
    best_position = (0, 0)

    # Scan through the image in (128, 128) windows to find the area with the most texture
    for i in range(image.shape[0] - window_size + 1):
        for j in range(image.shape[1] - window_size + 1):
            window = abs_laplacian[i:i+window_size, j:j+window_size]
            texture_value = np.sum(window)

            if texture_value > max_texture_value:
                max_texture_value = texture_value
                best_position = (i, j)

    # Get the coordinates of the most textured region
    top_left_y, top_left_x = best_position
    textured_region = image[top_left_y:top_left_y+window_size, top_left_x:top_left_x+window_size]

    # Detect the 4 most prominent colors in the textured region using KMeans
    def detect_prominent_colors(region, n_colors):
        # Reshape the image into (num_pixels, 3) for clustering
        pixels = region.reshape(-1, 3)

        # Apply KMeans to find the most prominent colors
        kmeans = KMeans(n_clusters=n_colors)
        kmeans.fit(pixels)

        # Get the cluster centers (the colors) and labels (which cluster each pixel belongs to)
        prominent_colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_

        # Count the number of pixels in each cluster to determine the largest color region
        unique, counts = np.unique(labels, return_counts=True)
        largest_color_index = unique[np.argmax(counts)]
        largest_color = prominent_colors[largest_color_index]

        return prominent_colors, largest_color

    # Get the 4 most prominent colors and the color that covers the maximum area
    prominent_colors, largest_color = detect_prominent_colors(textured_region, n_colors)

    # Convert RGB colors to LAB space
    def rgb_to_lab(color):
        color_rgb = np.uint8([[color]])  # Convert color to the right shape for cv2
        color_lab = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2LAB)[0][0]
        return color_lab

    # Convert all prominent colors and the largest color to LAB
    largest_color_lab = rgb_to_lab(largest_color)
    prominent_colors_lab = [rgb_to_lab(color) for color in prominent_colors]

    # Calculate the Euclidean distance between the largest color and other colors in LAB space
    def euclidean_distance(color1, color2):
        return np.linalg.norm(color1 - color2)

    max_distance = -1
    furthest_color_lab = None
    furthest_color_rgb = None

    # Find the color that is at the maximum distance from the largest color in LAB space
    for color_lab, color_rgb in zip(prominent_colors_lab, prominent_colors):
        distance = euclidean_distance(largest_color_lab, color_lab)
        if distance > max_distance:
            max_distance = distance
            furthest_color_lab = color_lab
            furthest_color_rgb = color_rgb

    # Convert the furthest LAB color back to RGB
    def lab_to_rgb(color_lab):
        color_lab = np.uint8([[color_lab]])  # Convert color to the right shape for cv2
        color_rgb = cv2.cvtColor(color_lab, cv2.COLOR_LAB2RGB)[0][0]
        return color_rgb

    # Convert furthest color from LAB to RGB
    furthest_color_rgb_converted = lab_to_rgb(furthest_color_lab)

    # Ensure colors are returned as tuples
    largest_color_rgb = tuple(largest_color)
    furthest_color_rgb_converted = tuple(furthest_color_rgb_converted)

    # Return the coordinates of the most textured region along with prominent colors and furthest color
    return (top_left_y, top_left_x), prominent_colors, largest_color_rgb, furthest_color_rgb_converted



# define a function for a QR code generation
# box_size is an option to manage size of an output image
def make_qr(custom_fill_color, custom_back_color, text, box_size=12, target_size=(128, 128)):
    qr = qrcode.QRCode(
        version=1,
        box_size=box_size,
        border=0
    )
    qr.add_data(text)
    qr.make(fit=True)
    qr_image = qr.make_image(fill_color=custom_fill_color, back_color=custom_back_color).convert('RGB')

    # Resize the QR code image to the target size
    resized_image = qr_image.resize(target_size)

    # Convert RGBA to RGB
    #rgb_image = resized_image.convert('RGB')

    return np.asarray(resized_image, dtype='uint8')



def pad_qr_code(qr_code, coordinates):
    # Unpack the coordinates tuple
    top_left_y, top_left_x = coordinates

    # Check that the QR code has the correct shape
    assert qr_code.shape == (128, 128, 3), "QR code must be of shape (128, 128, 3)"

    # Create a blank image of size (256, 256, 3) filled with zeros
    padded_image = np.zeros((256, 256, 3), dtype=np.uint8)

    # Calculate the bottom-right corner where the QR code will end
    bottom_right_y = top_left_y + qr_code.shape[0]
    bottom_right_x = top_left_x + qr_code.shape[1]

    # Ensure that the QR code fits within the bounds of the (256, 256) image
    assert bottom_right_y <= 256 and bottom_right_x <= 256, "QR code exceeds image boundaries."

    # Place the QR code in the specified region
    padded_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = qr_code

    return padded_image

# Example usage:
# Assuming 'qr_code' is the QR code image of shape (128, 128, 3)
# and coordinates is a tuple (top_left_y, top_left_x) from the previous function
# new_image_with_qr = pad_qr_code(qr_code, (top_left_y, top_left_x))

# 'new_image_with_qr' will now contain the (256, 256, 3) image with the QR code placed at the textured region.


# a function to generate a train data set
# output: (numpy array of images, list of corresponding texts)
def generate_dataset(host_images, n_of_samples, min_size = 1, max_size = 11):
    data = []
    labels = []
    report_step = int(n_of_samples * .1)
    report = report_step
    print("Generating")
    for i in range(n_of_samples):
        if i == report:
            #print("Done:", report / report_step * 10, "%")
            report = report + report_step
        #select size of the string to be embedded in a QR Code randonly between 1 and 11
        size = random.randint(min_size, max_size)
        #generate the string using random letters and space
        s = ''.join(random.choice(LETTERS) for i in range(size))
        #find the most textured region in the host image and the custom fill color and custom back color for the QR code
        (top_left_y, top_left_x), prominent_colors, largest_color, furthest_color_rgb_converted = find_most_textured_region(host_images[i])
        #make the QR Code
        img = make_qr(largest_color, furthest_color_rgb_converted, s)
        #ensure the size is 256 x 256 pixels
        #assert img.size == (256,256)
        qr = np.asarray(img, dtype='float')
        #print('qr.shape: ', qr.shape)
        #resize QR to match the host image
        padded_qr = pad_qr_code(qr, (top_left_y, top_left_x))
        #Append the QR code to the data list
        data.append(padded_qr)
        #give adding to the QR
        #Append the String of random size generated using random letters to the label list
        labels.append(s)
    print("Done:", "100", "%")
    return (np.asarray(data), labels)

"""### Create inputs for generator"""

def make_labels_for_position(labels, pos):
    chars = list(map(lambda x: x[pos] if pos < len(x) else EOI, labels)) # either a letter or EOI
    return list(map(lambda x: [ALL_CHAR_CLASSES.index(x)], chars)) # all classes are indexed [0..len(ALL_CHAR_CLASSES))

def get_training_label_sizes(training_labels):
  training_label_sizes = list(map(lambda x: [len(x) - 1], training_labels))
  return training_label_sizes

#Host Images
host_images_array, discriminator_1_labels_array = load_real_samples(target_shape = (256, 256), num_samples_to_take = 10)

#QR Codes
qr_codes, qr_code_labels = generate_dataset(host_images_array, n_of_samples = 10)
qr_code_labels = [list(string) for string in qr_code_labels]
evaluation_label_sizes = get_training_label_sizes(qr_code_labels)
evaluation_labels_char0 = make_labels_for_position(qr_code_labels, 0)
evaluation_labels_char1 = make_labels_for_position(qr_code_labels, 1)
evaluation_labels_char2 = make_labels_for_position(qr_code_labels, 2)
evaluation_labels_char3 = make_labels_for_position(qr_code_labels, 3)
evaluation_labels_char4 = make_labels_for_position(qr_code_labels, 4)
evaluation_labels_char5 = make_labels_for_position(qr_code_labels, 5)
evaluation_labels_char6 = make_labels_for_position(qr_code_labels, 6)
evaluation_labels_char7 = make_labels_for_position(qr_code_labels, 7)
evaluation_labels_char8 = make_labels_for_position(qr_code_labels, 8)
evaluation_labels_char9 = make_labels_for_position(qr_code_labels, 9)
evaluation_labels_char10 = make_labels_for_position(qr_code_labels, 10)

evaluation_labels = [
    np.asarray(evaluation_label_sizes),        # Size labels
    np.asarray(evaluation_labels_char0),       # Character 0 labels
    np.asarray(evaluation_labels_char1),       # Character 1 labels
    np.asarray(evaluation_labels_char2),       # Character 2 labels
    np.asarray(evaluation_labels_char3),       # and so on...
    np.asarray(evaluation_labels_char4),
    np.asarray(evaluation_labels_char5),
    np.asarray(evaluation_labels_char6),
    np.asarray(evaluation_labels_char7),
    np.asarray(evaluation_labels_char8),
    np.asarray(evaluation_labels_char9),
    np.asarray(evaluation_labels_char10)]

"""### Load the Generator Model"""

generator_model = load_model('/content/drive/MyDrive/generator_flowers_dynamic_e_10000.h5')

"""### Define YOLO function"""

# Install necessary libraries
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import cv2
from google.colab.patches import cv2_imshow

# Download YOLOv3 files if they are not already available
!wget -q https://pjreddie.com/media/files/yolov3.weights -O yolov3.weights
!wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -O yolov3.cfg
!wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -O coco.names

# Load YOLOv3 model and class names
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function to perform object detection using YOLOv3
def detect_objects_yolo(image, net, output_layers):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                box = detection[0:4] * np.array([width, height, width, height])
                center_x, center_y, w, h = box.astype("int")
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids, len(class_ids)

"""### Give inputs to Generator Model"""

def blend_data(qr_codes, host_images, num_samples):
  #print('type(host_images): ',(host_images.shape))
  #qr_codes = np.repeat(np.expand_dims(qr_codes, axis=-1), 1, axis=-1)
  #print('qr_codes.shape: ',(qr_codes.shape))
  host_images = host_images / 255.0
  average_blending_weights = []
  for i in range (qr_codes.shape[0]):
    #print('i blend_data = ', i)
    image_1 = qr_codes[i]
    image_1 = np.resize(image_1, (256, 256, 3))
    image_2 = host_images[i]
    blending_weight = (image_1 + image_2) / 2.0
    average_blending_weights.append(blending_weight)
  average_blending_weights = np.array(average_blending_weights)
  return [qr_codes, host_images], average_blending_weights

#create only batch/2 samples
def generate_fake_samples (g_model, qr_codes, qr_code_labels, host_images, data_size):

  qr_codes_trimmed = []
  qr_code_labels_trimmed = []

  host_images_trimmed = []

  for i in range(int(data_size)):
    #random_number = random.randint(0, int(data_size)-1)
    qr_codes_trimmed.append(qr_codes[i])
    qr_code_labels_trimmed.append(qr_code_labels[i])
    host_images_trimmed.append(host_images[i])

  qr_codes_trimmed = np.asarray(qr_codes_trimmed)
  qr_code_labels_trimmed = np.asarray(qr_code_labels_trimmed)
  host_images_trimmed = np.asarray(host_images_trimmed)

  test_inputs, average_blending_weights = blend_data(qr_codes=qr_codes_trimmed, host_images=host_images_trimmed, num_samples = data_size) # For testing, we don't need targets
  fake_samples = g_model.predict(test_inputs, data_size)
  fake_sample_labels = np.ones(int(data_size))

  return fake_samples, fake_sample_labels, qr_code_labels_trimmed, average_blending_weights, host_images_trimmed

#composite_images = generator_model.predict([qr_codes, host_images_array],5)
composite_images, y_fake, _, _, _ = generate_fake_samples (generator_model, qr_codes, evaluation_labels, host_images = host_images_array, data_size = 10)

"""### Calculate n_composite"""

n_host = 0
# Load images
target_shape = (256, 256)
#images, _ = host_images_array
# Get the output layers from YOLO
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Perform object detection on the loaded images
for i, image in enumerate(host_images_array):
    #print(type(image))
    boxes, confidences, class_ids, count = detect_objects_yolo(image, net, output_layers)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(f"Image {i+1} has {len(indices)} objects detected.")
    n_host = n_host + len(indices)
    if len(indices) > 0:
        for j in indices.flatten():
            x, y, w, h = boxes[j]
            label = str(classes[class_ids[j]])
            confidence = confidences[j]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image with bounding boxes
    cv2_imshow(image)
print('n_host = ', n_host)

n_composite = 0
# Get the output layers from YOLO
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Perform object detection on the loaded images
for i, image in enumerate(composite_images):
    image = (image * 255).astype(np.uint8)
    #image = cv2.resize(image, target_shape)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print(image)
    boxes, confidences, class_ids, count = detect_objects_yolo(image, net, output_layers)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(f"Image {i+1} has {len(indices)} objects detected.")
    n_composite = n_composite + len(indices)
    if len(indices) > 0:
        for j in indices.flatten():
            x, y, w, h = boxes[j]
            label = str(classes[class_ids[j]])
            confidence = confidences[j]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image with bounding boxes
    cv2_imshow(image)
    #plt.show(image)
print('n_composite = ', n_composite)
