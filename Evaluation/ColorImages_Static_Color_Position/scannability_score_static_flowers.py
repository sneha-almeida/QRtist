# -*- coding: utf-8 -*-
"""
# Color Images - Static

## TensorFlow Flowers Dataset

### Imports
"""
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
from tensorflow.keras.layers import Input, Concatenate, Conv2D, UpSampling2D, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import cv2
import os
from PIL import Image

import tensorflow_datasets as tfds
from keras.layers import Dense, LeakyReLU
from matplotlib import pyplot


"""### Function to Load Tensorflow Flowers Dataset"""

def load_real_samples(target_shape, num_samples_to_take = 500):
    constant_label = 1
    caltech_builder = tfds.builder('tf_flowers')
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

# define a function for a QR code generation
# box_size is an option to manage size of an output image
def make_qr(text, box_size=12, target_size=(128, 128)):
    qr = qrcode.QRCode(
        version=1,
        box_size=box_size,
        border=0
    )
    qr.add_data(text)
    qr.make(fit=True)
    qr_image = qr.make_image(fill_color="black", back_color="white").convert('RGB')

    # Resize the QR code image to the target size
    resized_image = qr_image.resize(target_size)

    # Convert RGBA to RGB
    #rgb_image = resized_image.convert('RGB')

    return np.asarray(resized_image, dtype='uint8')

# a function to generate a train data set
# output: (numpy array of images, list of corresponding texts)
def generate_dataset(n_of_samples, min_size = 1, max_size = 11):
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
        #make the QR Code
        img = make_qr(s)
        #ensure the size is 256 x 256 pixels
        #assert img.size == (256,256)
        qr = np.asarray(img, dtype='float')
        #resize QR to match the host image

        #Append the QR code to the data list
        data.append(qr)
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
host_images_array, discriminator_1_labels_array = load_real_samples(target_shape = (256, 256), num_samples_to_take = 5)

#QR Codes
qr_codes, qr_code_labels = generate_dataset(n_of_samples = 5)
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

generator_model = load_model('generator_flowers_static_e_6000.h5')

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
composite_images, y_fake, _, _, _ = generate_fake_samples (generator_model, qr_codes, evaluation_labels, host_images = host_images_array, data_size = 5)

print(composite_images.shape)

"""### QR Code Scanner"""

import cv2
import numpy as np
qr_code_detection_count = 0
def scan_qr_codes_from_images(images):
    results = []

    # Initialize the QRCode detector
    qr_detector = cv2.QRCodeDetector()

    for image in images:
        # Ensure the image is in the correct format (8-bit unsigned integer)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Detect and decode the QR code
        data, points, _ = qr_detector.detectAndDecode(image)

        if points is not None:
            image_results = {
                'data': data,
                'rect': points.astype(int).tolist()  # Points of the bounding box
            }
            results.append(image_results)
        else:
            results.append(None)  # No QR code detected

    return results

# Example usage with a list of images
image_paths = composite_images  # Assuming composite_images is a list of image arrays
qr_results = scan_qr_codes_from_images(composite_images)
data_list = []
for idx, result in enumerate(qr_results):
    if result:
        print(f"Image {idx + 1}:")
        print(f"Data: {result['data']}, Position: {result['rect']}")
        data_list.append(result['data'])
        qr_code_detection_count = qr_code_detection_count + 1
    else:
        print(f"Image {idx + 1}: No QR code detected")
        data_list.append(None)

scannability_score = (qr_code_detection_count * total_score) / len(qr_code_labels)
print(f"Scannability Score: {scannability_score:.2f}")



