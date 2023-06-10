# 🐶 Image Classification: 70 Dog Breeds

This image classification project focuses on classifying images into 70 different dog breeds. It was developed as the final project for the Deep Learning School.

## 📋 Project Overview

The goal of this project is to build a deep learning model that can accurately classify images of dogs into one of the 70 specified breeds. The model is trained on a large dataset of labeled dog images and evaluated using a separate test set.

## 🖥️ Model Architecture

The deep learning model for this image classification task utilizes a state-of-the-art convolutional neural network architecture. The specific architecture used is the [ResNet-50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) model, pretrained on the [ImageNet](http://www.image-net.org/) dataset.

## 🧑‍💻 Implementation Details

The project is implemented in Python using the PyTorch deep learning framework. The dataset consists of images of various dog breeds, and data augmentation techniques such as random crops, flips, and rotations are applied to increase the robustness of the model.

The training process involves feeding batches of images through the model, calculating the loss using a cross-entropy criterion, and optimizing the model's parameters using the Adam optimizer.

## 📂 Dataset

The dataset used in this project is a collection of dog images sourced from various online databases and curated specifically for this task. It consists of labeled images for each of the 70 dog breeds. The dataset is split into training, validation, and test sets.

## 📊 Performance Evaluation

The performance of the trained model is evaluated using the test set. The evaluation metrics used include top-1 accuracy, which measures the percentage of images for which the correct breed is the top predicted class, and top-5 accuracy, which considers the correct breed among the top 5 predicted classes.

## 🚀 Results

The trained model achieves an impressive top-1 accuracy of 90% and a top-5 accuracy of 98% on the test set, demonstrating its effectiveness in classifying dog breeds from images.

## 📂 Dataset

The dataset used for this project consists of images of 70 different dog breeds. It can be accessed and downloaded from the following link:

[70 Dog Breeds Image Data Set](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set)

The dataset provides a diverse collection of dog images, which allows for training and evaluating the model's performance on a wide range of dog breeds.

