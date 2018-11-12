# Finetune AlexNet with Tensorflow

This repository contains all the code needed to finetune [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) on any arbitrary dataset. 

The code is originally from [here](https://github.com/kratzert/finetune_alexnet_with_tensorflow). Beside the comments in the code itself, the author also wrote an article which you can fine [here](https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html) with further explanation.

## Requirements

- Python 3.5
- TensorFlow 1.0
- Numpy
- OpenCV (If you want to use the provided ImageDataGenerator in `datagenerator.py`)

## TensorBoard support

The code has TensorFlows summaries implemented so that you can follow the training progress in TensorBoard. (--logdir in the config section of `finetune.py`)

## Content

- `alexnet.py`: Class with the graph definition of the AlexNet.
- `finetune.py`: Script to run the finetuning process.
- `datagenerator.py`: Some auxiliary class I wrote to load images into memory and provide batches of images with their labels on function call. Includes random shuffle and horizontal flipping.
- `caffe_classes.py`: List of the 1000 class names of ImageNet (copied from [here](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/)).
- `validate_alexnet_on_imagenet.ipynb`: Notebook to test the correct implementation of AlexNet and the pretrained weights on some images from the ImageNet database.
- `images/*`: contains three example images, needed for the notebook.
- `generate_path_label_dogs_cats.py` generate txt files for [cats vs dogs](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)

## Usage

All you need to touch is the `finetune.py` where you will find a section of configuration settings you have to adapt on your problem. You have to provide two `.txt` files to the script (`train.txt` and `val.txt`). Each of them list the complete path to your train/val images together with the class number in the following structure.

### steps to run the program
- download the file of the pretrained weights bvlc_alexnet.npy
- modify image_path in generate_path_label.py to generate two files
- python3 finetune.py
