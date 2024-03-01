# Convolutional Neural Network

This repository contains a convolutional neural network trained on the MNIST dataset using Keras.

## Dataset

The MNIST dataset is a collection of handwritten digits. It consists of 70,000 images, each of size 28x28 pixels. The dataset is directly compatible with the original MNIST dataset.

### Eminst - Dataset Descriptions

- **byclass**: 814,255 characters. 62 unbalanced classes. Numbers and letters.
- **bymerge**: 814,255 characters. 47 unbalanced classes. Numbers and letters, with similar-looking letters merged.
- **balanced**: 131,600 characters. 47 balanced classes. Numbers and letters, balanced across classes.
- **letters**: 145,600 characters. 26 balanced classes. Letters, balanced across classes.
- **digits**: 280,000 characters. 10 balanced classes. Numbers, balanced across classes.
- **mnist**: 70,000 characters. 10 balanced classes. Numbers, directly compatible with the original MNIST dataset.

For more information about the dataset, please refer to the [EMNIST repository](https://www.nist.gov/itl/products-and-services/emnist-dataset).

## Model Architecture for Digits recognition

The convolutional neural network model consists of the following layers:

1. Input layer: Accepts images of size 28x28 pixels.
2. Convolutional layer: Applies 32 filters of size 3x3 with ReLU activation.
3. Max pooling layer: Performs max pooling with a pool size of 2x2.
4. Convolutional layer: Applies 64 filters of size 3x3 with ReLU activation.
5. Max pooling layer: Performs max pooling with a pool size of 2x2.
6. Flatten layer: Flattens the output of the previous layer.
7. Dropout layer: Applies dropout with a rate of 0.5.
8. Dense layer: Fully connected layer with softmax activation, producing the final output.
