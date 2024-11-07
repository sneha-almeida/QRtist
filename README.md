# QRtist: Experimenting With GANs to Embed Scannable QR Codes in Host Images

QR codes can be scanned using smartphone cameras. While QR codes are commonly used to share information such as links or contact numbers for convenience, they are also capable of carrying other types of data.
In this work, we conduct experiments to assess the feasibility of embedding QR codes within host images in such a way that the QR codes remain functional, mean- ing the data embedded in the QR code is still scannable. These types of images are particularly relevant in the advertising industry, where space is often limited. How- ever, if the QR code could be embedded subtly such that it is visible only enough to be scanned by a QR code reader, advertisers could preserve space while still bene- fiting from the convenience of the QR code functionality without compromising the aesthetics of the design spaces.
We employ Generative Adversarial Networks (GANs) to achieve this objective. Both QR codes and host images (where the QR codes will be embedded) are provided as inputs to the GAN, which aims to blend them into a single output image referred to as a composite image. Initially, we perform experiments using grayscale images as inputs, for which we develop the Dual Discriminator GAN framework. How- ever, when color images are introduced as inputs, the framework fails to function effectively. As a result, we adapt the framework to handle color images, leading to the development of the Dual Loss GAN. To further improve output quality, we introduce pre-processing steps to the input images, as opposed to using raw inputs. Subsequent experiments reveal that the Dual Loss GAN does not perform well with grayscale images, indicating that GAN architectures are sensitive to the type of input data (grayscale versus color). The presence or absence of color affects GAN training, particularly when the network attempts to map two input distributions (the host image and the QR code) into a single output distribution (the composite image).
We evaluate the composite images produced by these GANs using various visual and statistical metrics to assess how well the GANs meet our objectives while main- taining visual quality. The results demonstrate that the nature of the inputs limits the output quality in the Dual Discriminator GAN. On the other hand, the Dual Loss GAN, especially when given pre-processed inputs, performs better across most evaluations and generates images with higher visual quality.

## Project Structure

The project consists of three main phases:

- **Hyperparameter Tuning**: Perform hyperparameter tuning on each GAN to optimize parameters such as learning rate, batch size, and network architecture.
  - Use a tuning framework (e.g., Ray Tune) to streamline the tuning process.
  
- **Model Training**: Train each GAN using the optimal hyperparameters obtained from the tuning phase.
  - Save the evolution in images produced by the GAN during training, loss curves, loss values, and trained model weights for later evaluation.
  
- **Evaluation**: Evaluate the generator from each GAN model using relevant metrics to measure the quality and diversity of generated samples.
  - Visualize and compare the performance of the trained generators.

## Prerequisites

Ensure you have the following dependencies installed: 

# Install requirements
```
pip install -r requirements.txt
```

## Repository Structure

**HyperParameterTuning**:  
Scripts for hyperparameter tuning of three GAN architectures using Ray Tune.  
JSON files containing optimized hyperparameters for each GAN architecture.

**Training**:  
Scripts for training the GAN models using the best parameters obtained from the tuning phase.  
Output includes the evolution of generated images, loss curves, and values for individual losses.

**Evaluation**:  
Scripts for evaluating each GAN's performance on different datasets and visualizing results.  
Contains evaluations on both grayscale and color images, as well as scannability and object detection scores.

## Usage

### Hyperparameter Tuning
Run the following scripts for tuning each GAN model:
```
python HyperParameterTuning/dual_loss_gan_static.py
```
```
python HyperParameterTuning/dual_discriminator_gan.py
```
```
python HyperParameterTuning/dual_loss_gan_dynamic.py
```


### Model Training
Train each GAN with the optimal hyperparameters as follows:
```
python Training/dual_discriminator_gan_train.py
```
```
python Training/dual_loss_gan_static_train.py
```
```
python Training/dual_loss_gan_dynamic_train.py
```

### Generator Evaluation:
Evaluate the generator models with these commands for both grayscale and color datasets:

#### Grayscale Images (MNIST dataset, Fashion MNIST dataset)
Evaluation on grayscale images:

```
python Evaluation/Greyscale/gan_testing_greyscale_images_mnist.py
```
```
python Evaluation/Greyscale/gan_testing_greyscale_images_fashion_mnist.py
```


Evaluation on grayscale Fashion MNIST images:
```
python Evaluation/Greyscale/scannability_score_greyscale_mnist.py
```
```
python Evaluation/Greyscale/scannability_score_greyscale_fashion_mnist.py
```

#### Color Images with Static and Dynamic Color Positioning (Flowers dataset, Cats vs. Dogs dataset)

**Static Color Positioning**

Evaluation on static color-positioned flowers dataset:
```
python Evaluation/ColorImages_Static_Color_Position/gan_testing_color_images_static_flowers.py
```
```
python Evaluation/ColorImages_Static_Color_Position/gan_testing_color_images_static_cats_vs_dogs.py
```


Evaluation on static color-positioned cats vs. dogs dataset:
```
python Evaluation/ColorImages_Static_Color_Position/scannability_score_static_flowers.py python Evaluation/ColorImages_Static_Color_Position/scannability_score_static_cats_vs_dogs.py
```
```
python Evaluation/ColorImages_Static_Color_Position/object_detection_static.py
```

**Dynamic Color Positioning**

Evaluation on dynamically color-positioned flowers dataset:
```
python Evaluation/ColorImages_Dynamic_Color_Position/gan_testing_color_images_dynamic_flowers.py
```
```
python Evaluation/ColorImages_Dynamic_Color_Position/gan_testing_color_images_dynamic_cats_vs_dogs.py
```

Evaluation on dynamically color-positioned cats vs. dogs dataset:
```
python Evaluation/ColorImages_Dynamic_Color_Position/scannability_score_dynamic_flowers.py python
```
```
Evaluation/ColorImages_Dynamic_Color_Position/scannability_score_dynamic_cats_vs_dogs.py
```
```
python Evaluation/ColorImages_Dynamic_Color_Position/object_detection_dynamic.py
```

## Results and Analysis

The evaluation outputs, including scannability scores, and object detection results, are stored in their respective directories within the Evaluation folder. Each GAN's performance on different datasets can be visualized and analyzed to understand the quality and effectiveness of the generated images.


