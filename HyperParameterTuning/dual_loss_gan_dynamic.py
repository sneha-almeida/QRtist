
import qrcode
import random
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv2D, UpSampling2D, Reshape, ZeroPadding2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from tensorflow.keras.applications import VGG16

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.cluster import KMeans

import cv2
import os
from PIL import Image


import tensorflow_datasets as tfds
from keras.layers import Dense, LeakyReLU
from matplotlib import pyplot

from ray.tune.schedulers import ASHAScheduler
from ray.air.config import RunConfig

import json
import ray
from ray import tune
from ray.air import session

ray.init(dashboard_port=8265)

"""# Global Speicfications"""


image_width = 256
image_height = 256
channels = 3

"""# Functions

## Load Real Samples
"""

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

"""## Generate QR Codes"""

LETTERS = string.ascii_lowercase + ' ' # letters and space
EOI = '#' # end of input
MAX_SIZE = 11 # max input/output size
ALL_CHAR_CLASSES = LETTERS + EOI # all available classes



import cv2
import numpy as np
from sklearn.cluster import KMeans

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


def generate_dataset(host_images, n_of_samples, min_size = 1, max_size = 11):
    data = []
    labels = []
    report_step = int(n_of_samples * .1)
    report = report_step
    print("Generating")
    for i in range(n_of_samples):
        if i == report:
            
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
        
        #resize QR to match the host image
        padded_qr = pad_qr_code(qr, (top_left_y, top_left_x))
        #Append the QR code to the data list
        data.append(padded_qr)
        #give adding to the QR
        #Append the String of random size generated using random letters to the label list
        labels.append(s)
    
    return (np.asarray(data), labels)

"""## Create Dataset for Discriminator 1

For Discriminator 1, that is supposed to ensure that the image generated by the generator is similar to the host image, we need the following dataset:

consider the data_size i.e., the number of data samples in a batch is n, then dataset should contain n/2 real images and n/2 fake images.
"""

def create_dataset_d1(real_samples, real_labels, fake_samples, fake_labels, data_size):
  data_batch = []
  labels_batch = []
  
  for i in range (int(data_size/2)):
    
    random_number = random.randint(0, int(data_size/2))
    
    data_batch.append(real_samples[random_number])
    labels_batch.append(real_labels[random_number])

  for i in range (int(data_size/2)):
    
    random_number = random.randint(0, int(data_size/2))
    
    data_batch.append(fake_samples[random_number])
    labels_batch.append(fake_labels[random_number])

  data_batch = np.asarray(data_batch)
  labels_batch = np.asarray(labels_batch)

  return data_batch, labels_batch

"""## Create Dataset for Discriminator 2

For discriminator 2 that has to ensure that the QR Code is scannable, we have to create a dataset of fake samples and the qr code text embedded in those.
So we need the following dataset:

consider the data_size i.e., the number of data samples in a batch is n, then dataset should contain n fake images
"""

def create_dataset_d2(fake_samples, qr_code_labels, data_size):
  fake_samples_trimmed = []
  qr_code_labels_trimmed = []
  for i in range (int(data_size)):
    random_number = random.randint(0, int(data_size)-1)
    fake_samples_trimmed.append(fake_samples[i])
    qr_code_labels_trimmed.append(qr_code_labels[i])

  fake_samples_trimmed = np.asarray(fake_samples_trimmed)
  qr_code_labels_trimmed = np.asarray(qr_code_labels_trimmed)
  return fake_samples_trimmed, qr_code_labels_trimmed


"""## Create Dataset of Generator"""

def create_dataset_g(real_samples, qr_codes, qr_code_labels, data_size):
  real_samples_trimmed = []
  qr_codes_trimmed = []
  qr_code_labels_trimmed = []
  for i in range(int(data_size)):
    
    qr_code = qr_codes[i]
    real_sample = real_samples[i]
    
    #padded_qr = find_most_textured_region(real_sample, qr_code, region_size=(128, 128))
    
    real_samples_trimmed.append(real_sample)
    qr_codes_trimmed.append(qr_code)
    qr_code_labels_trimmed.append(qr_code_labels[i])

  real_samples_trimmed = np.asarray(real_samples_trimmed)
  qr_codes_trimmed = np.asarray(qr_codes_trimmed)

  return qr_codes_trimmed, real_samples_trimmed, qr_code_labels_trimmed

"""## Generate Fake Samples Using Generator"""

def blend_data(qr_codes, host_images, num_samples):
  
  host_images = host_images / 255.0
  average_blending_weights = []
  for i in range (qr_codes.shape[0]):
    
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
    qr_code_labels_trimmed.append(qr_code_labels[i])
    qr_code = qr_codes[i]
    host_image = host_images[i]
    host_images_trimmed.append(host_images[i])
    qr_codes_trimmed.append(qr_code)

  qr_codes_trimmed = np.asarray(qr_codes_trimmed)
  qr_code_labels_trimmed = np.asarray(qr_code_labels_trimmed)
  host_images_trimmed = np.asarray(host_images_trimmed)

  test_inputs, average_blending_weights = blend_data(qr_codes=qr_codes_trimmed, host_images=host_images_trimmed, num_samples = data_size) # For testing, we don't need targets
  fake_samples = g_model.predict(test_inputs, data_size)
  fake_sample_labels = np.ones(int(data_size))

  return fake_samples, fake_sample_labels, qr_code_labels_trimmed, average_blending_weights, host_images_trimmed

      
"""## Create Generator Model"""

def generator_model(input_shape_qr=(256, 256, 3), input_shape_host=(256, 256, 3)):
    # Define the input layers for the two images
    input_image1 = Input(shape=input_shape_qr, name='input_image1')
    input_image2 = Input(shape=input_shape_host, name='input_image2')
    
    # Concatenate the input images along the channel axis
    concatenated_images = Concatenate(axis=-1)([input_image1, input_image2])

    # Convolutional layers for blending
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(concatenated_images)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # Output with 3 channels (RGB)

    # Create the model
    model = Model(inputs=[input_image1, input_image2], outputs=x, name='Generator')
    model.summary()
    # keras.utils.plot_model(model, 'generator.png', show_shapes=True)
    return model

#generator_model()

"""## Create Discriminator 1 Model (Perceptual Similarity)"""

def create_discriminator_model_for_perceptual_similarity(input_shape_generator_output = (256,256,3)):

  base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape_generator_output)

    # Freeze the layers in the pre-trained model
  for layer in base_model.layers:
        layer.trainable = False
  input = keras.Input(shape = input_shape_generator_output)
  x = base_model(input)
  x = layers.Conv2D(64, (3,3), activation='relu', padding = 'same', strides = 2)(x)
  #x = layers.Conv2D(64, (3,3), activation='relu', padding = 'same', strides = 2)(x)
  x = layers.Conv2D(128, (3,3), activation='relu', padding = 'same')(x)
  x = layers.Conv2D(256, (3,3), activation='relu', padding = 'same')(x)
  x = layers.Conv2D(128, (3,3), activation='relu', padding = 'same')(x)
  x = layers.Conv2D(64, (3,3), activation='relu', padding = 'same')(x)
  #x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # Output with 3 channels (RGB)
  x = layers.Flatten()(x)
  x = layers.Dropout(0.4)(x)
  output = layers.Dense(1, activation='sigmoid')(x)

  model = keras.Model(inputs = input, outputs = output, name = 'discriminator1')
  binary_crossentropy_loss = tf.keras.losses.BinaryCrossentropy()

  model.compile(optimizer='adam', loss=binary_crossentropy_loss, metrics=['accuracy'])
  model.summary()
  #keras.utils.plot_model(model, 'discriminator1.png', show_shapes = True)
  return model

#create_discriminator_model_for_perceptual_similarity()

"""## Create Discriminator 2 Model (QR Code Scanner)"""

def get_training_label_sizes(training_labels):
  training_label_sizes = list(map(lambda x: [len(x) - 1], training_labels))
  return training_label_sizes

def make_labels_for_position(labels, pos):
    chars = list(map(lambda x: x[pos] if pos < len(x) else EOI, labels)) # either a letter or EOI
    return list(map(lambda x: [ALL_CHAR_CLASSES.index(x)], chars)) # all classes are indexed [0..len(ALL_CHAR_CLASSES))

def define_split_with_size_multi_output_model(learning_rate, beta, input_shape_host = (256,256,3)):
    input_layer = keras.layers.Input(shape = input_shape_host, dtype='float', name='input_qr')
    flatten = keras.layers.Flatten(input_shape = input_shape_host, name='flatten')(input_layer)
    hidden_chars1 = keras.layers.Dense(441, activation='relu', name='hidden_chars1')(flatten)
    hidden_size = keras.layers.Dense(441, activation='relu', name='hidden_size')(flatten)
    size_output = keras.layers.Dense(MAX_SIZE, name='size_output')(hidden_size)

    # stop back propagation since size is independent from actual characters
    size_without_more_optimizations = tf.stop_gradient(size_output, name='size_wo_gradient')

    outputs = [size_output]
    for i in range(11):
        hidden_chars2 = keras.layers.Dense(21*21, activation='relu')(hidden_chars1)
        combined_char_inputs = keras.layers.concatenate([hidden_chars2, size_without_more_optimizations])
        char_output = keras.layers.Dense(len(ALL_CHAR_CLASSES), name='char' + str(i))(combined_char_inputs)
        outputs.append(char_output)

    multi_output_model = keras.Model(inputs=[input_layer], outputs=outputs)
    opt = Adam(lr=learning_rate, beta_1=beta)
    multi_output_model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    multi_output_model.summary()
    #keras.utils.plot_model(multi_output_model, 'discriminator2.png', show_shapes = True)
    #print('type(multi_output_model): ', type(multi_output_model))
    return multi_output_model



"""## Create GAN Model"""

def create_gan_model(learning_rate_gan, beta_gan, learning_rate_discriminator, beta_discriminator, loss_weights, input_shape_qr=(256, 256, 3), input_shape_host=(256, 256, 3), input_shape_generator_output=(256, 256, 3)):
    g_model = generator_model(input_shape_qr=input_shape_qr, input_shape_host=input_shape_host)
    d2_model = define_split_with_size_multi_output_model(learning_rate_discriminator, beta_discriminator,input_shape_host=input_shape_host)

    input_image1 = Input(shape=input_shape_qr, name='input_image1')
    input_image2 = Input(shape=input_shape_host, name='input_image2')

    generator_output = g_model([input_image1, input_image2])
    discriminator2_output = d2_model(generator_output)

    for model in discriminator2_output:
        model.trainable = False

    gan_model = keras.Model(inputs=[input_image1, input_image2], outputs=[generator_output, discriminator2_output], name='gan_model')

    mse_loss = MeanSquaredError()
    sparse_categorical_crossentropy_loss = SparseCategoricalCrossentropy(from_logits=True)
    opt = Adam(lr=learning_rate_gan, beta_1=beta_gan)
    losses = [mse_loss] + [sparse_categorical_crossentropy_loss] * len(discriminator2_output)
    gan_model.compile(optimizer=opt, loss=losses, metrics=['accuracy'],loss_weights=loss_weights)
    gan_model.summary()

    return g_model, d2_model, gan_model


 
def prepare_qr_code_batch(qr_codes, target_shape=(256, 256, 3)):
    def pad_and_resize_single_qr_code(qr_code):
        qr_code = tf.image.resize_with_crop_or_pad(qr_code, target_shape[0], target_shape[1])
        return qr_code
    
    # Apply padding and resizing to each QR code in the batch
    padded_qr_codes = np.array([pad_and_resize_single_qr_code(qr_code).numpy() for qr_code in qr_codes])
    return padded_qr_codes
 
def save_plot(x_fake, X_real, qr_codes, epoch, n):
    
    fig, axs = plt.subplots(n, 3, figsize=(10, 10))
    for i in range(n):  # Loop through rows
        print("x_fake")
        axs[i, 0].imshow(x_fake[i])  # Plot image from the first list
        axs[i, 0].axis('off')  # Turn off axis labels
        print("X_real")
        axs[i, 1].imshow(X_real[i].astype('uint8'))  # Plot image from the second list
        axs[i, 1].axis('off')  # Turn off axis labels
        print("qr_codes")
        axs[i, 2].imshow(qr_codes[i])  # Plot image from the third list
        axs[i, 2].axis('off')  # Turn off axis labels
        img_file = 'resize_99_1_padded_qr_kmeans_color/'+str(epoch)+'_'+str(i)+'.png'
        plt.imsave(img_file, x_fake[i])


    # Add titles to subplots (optional)
    axs[0, 0].set_title('GAN Images')
    axs[0, 1].set_title('Real Images')
    axs[0, 2].set_title('QR Codes')

    # Adjust layout and display the plot
    #pyplot.tight_layout()
    filename = 'resized_full_images_99_1_padded_qr_kmeans_color/flowers_e%03d.png' % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d2_model, data_size):
  qr_code_images_expanded_dims_list = []
  # load real samples for Discriminator 1
  X_real, y_real = load_real_samples(target_shape = (256, 256), num_samples_to_take = data_size)
  #load QR Codes for Discriminator 2
  qr_codes, qr_code_labels = generate_dataset(X_real, data_size)
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
  # prepare fake examples
  x_fake, y_fake, _, _, _ = generate_fake_samples (g_model, qr_codes, evaluation_labels, host_images = X_real, data_size = global_batch_size)
  
  acc_fake_d2 = d2_model.evaluate(x_fake, y_fake, verbose=0)
  
  qr_codes = prepare_qr_code_batch(qr_codes)
  acc_real_d2 = d2_model.evaluate(qr_codes, evaluation_labels, verbose=0)#there is an error here because qr codes have dimensions 256 x 256, but the network expects an image of 256 x 256 x 3
  
  save_plot(x_fake, X_real, qr_codes, epoch, n = x_fake.shape[0])
  
  
  
def plot_losses(d2_loss_list, g_loss_list, global_epochs):
  epochs_list = list(range(0, global_epochs))
  #plt.plot(epochs_list, d1_loss_list, color = 'red')
  plt.plot(epochs_list, d2_loss_list, color = 'green')
  plt.plot(epochs_list, g_loss_list, color = 'blue')
  plt.savefig('resize_99_1_padded_qr_kmeans_color.png')
  plt.close()
  


"""## Train GAN"""

def train_gan(config, checkpoint_dir = None):
    # changes for hyper parameter tuning
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    data_size = config["data_size"]
    gan_lr = config["gan_lr"]
    d1_lr = config["d1_lr"]
    gan_loss_weights = config["gan_loss_weights"]
    gan_beta = config["gan_beta"]
    d1_beta = config["d1_beta"]
    d2_loss_list = []
    g_loss_list = []
    
    image_width=256
    image_height=256
    
    g_model, d2_model, gan_model = create_gan_model(learning_rate_gan = gan_lr, beta_gan = gan_beta, learning_rate_discriminator = d1_lr, beta_discriminator = d1_beta, loss_weights = gan_loss_weights)
    
    
    images_real, labels_real = load_real_samples(target_shape=(image_width, image_height), num_samples_to_take=data_size)
    batch_per_epoch = int(data_size / batch_size)
    qr_codes, qr_code_labels = generate_dataset(images_real, data_size)

    for i in range(epochs):
        print('#############################################################  epoch = ', i)
        for j in range(batch_per_epoch):
            fake_samples, fake_sample_labels, qr_code_labels_trimmed, average_blending_weights, host_images_trimmed = generate_fake_samples(
                g_model, qr_codes, qr_code_labels, host_images=images_real, data_size=data_size)
            
            discriminator_2_data_batch, discriminator_2_labels_batch = create_dataset_d2(
                fake_samples=fake_samples, qr_code_labels=qr_code_labels, data_size=data_size)

            training_label_sizes = get_training_label_sizes(discriminator_2_labels_batch)
            training_labels_char = [make_labels_for_position(discriminator_2_labels_batch, k) for k in range(11)]

            d2_loss, *x = d2_model.train_on_batch(discriminator_2_data_batch, [
                np.asarray(training_label_sizes)] + [np.asarray(label) for label in training_labels_char])
            d2_loss_list.append(d2_loss)

            generator_input1_batch, generator_input2_batch, qr_code_labels_trimmed = create_dataset_g(
                real_samples=images_real, qr_codes=qr_codes, qr_code_labels=qr_code_labels, data_size=data_size)

            training_label_sizes = get_training_label_sizes(qr_code_labels_trimmed)
            training_labels_char = [make_labels_for_position(qr_code_labels_trimmed, k) for k in range(11)]

            # Ensure only two inputs are provided
            # Adjusting fake samples and host images to match shape for MSE loss
            g_loss, *x = gan_model.train_on_batch(
                x=[generator_input1_batch, generator_input2_batch],
                y=[generator_input2_batch, [np.asarray(training_label_sizes)] + [np.asarray(label) for label in training_labels_char]])
            g_loss_list.append(g_loss)

            print('>%d, %d/%d, d2=%.3f, g=%.3f' % (i + 1, j + 1, batch_per_epoch, d2_loss, g_loss))
        session.report({"g_loss": np.mean(g_loss_list), "d2_loss": np.mean(d2_loss_list)})
        

# No need for the following functions:
# 1) create_dataset_d2
# 2) create_dataset_g

# Define the scheduler
scheduler = ASHAScheduler(
    metric="g_loss",  # The metric to track
    mode="min",       # We want to minimize the generator loss
    max_t=10000,  # Maximum epochs
    grace_period=900,  # Minimum epochs before stopping trials
    reduction_factor=2
)

# Define the search space
search_space = {
    "epochs": tune.choice([1000, 2000, 4000, 6000, 8000, 10000]),  # Number of epochs
    "gan_lr": tune.loguniform(1e-5, 1e-1),  # Learning rate for GAN
    "d1_lr": tune.loguniform(1e-5, 1e-1),  # Learning rate for Discriminator 1
    "d1_beta": tune.loguniform(0.5, 0.9),  # decay rate for Discriminator 1
    "gan_beta": tune.loguniform(0.5, 0.9),  # decay rate for generator
    "batch_size": tune.choice([32, 64, 128]),  # Batch size
        "data_size": tune.sample_from(lambda spec: max(spec.config["batch_size"], 1)),  # Data size
    "gan_loss_weights": tune.grid_search([[0.99, 0.01], [0.95, 0.05], [0.75, 0.25], [0.60, 0.40], [0.5, 0.5]])  # Weights for generator and discriminator losses
}


# TuneConfig: Add the scheduler here
tuner = tune.Tuner(
    tune.with_resources(train_gan, resources={"cpu": 60, "gpu": 0}),  # Define CPU and GPU resources for each trial
    param_space=search_space,  # Search space defined above
    tune_config=tune.TuneConfig(
        num_samples=2,    # Number of different configurations to sample
        scheduler=scheduler  # Early stopping scheduler goes here
    ),
    run_config=RunConfig(
        stop={"training_iteration": 1000},  # Maximum training iterations per trial
        local_dir="./ray_results_flowers_dynamic",  # Specify where to save results
    )
)

# Run the tuner
results = tuner.fit()

# Get the best configuration
best_config = results.get_best_result(metric="g_loss", mode="min").config
print("Best hyperparameters: ", best_config)

# Save the best configuration
import json
with open("best_gan_config_flowers_dynamic.json", "w") as f:
    json.dump(best_config, f)


