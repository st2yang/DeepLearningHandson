"""Evaluate the model"""

import argparse
import logging
import os

import tensorflow as tf
import numpy as np
# TODO: to remove opencv
import cv2

from model.input_fn import input_fn
from model.model_fn import model_fn
from model.utils import Params
from model.utils import set_logger


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/default',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of model dir or file containing the weights")


if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    data_dir = args.data_dir
    test_data_dir = os.path.join(data_dir, "test")

    # Get the filenames from the test set
    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith('.jpg')]

    test_labels = [int(f.split('/')[-1][0]) for f in test_filenames]

    # specify the size of the evaluation set
    params.eval_size = len(test_filenames)

    # create the iterator over the dataset
    test_inputs = input_fn(False, test_filenames, test_labels, params)

    # Define the model
    logging.info("Creating the model...")
    model_spec, model_intfs = model_fn('eval', test_inputs, params, reuse=False)
    # images = tf.placeholder(tf.float32, [None, params.image_size, params.image_size, 3])

    # Read images
    data_dir = os.path.join(args.data_dir, "test")
    train_filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]
    images_data = np.ndarray([90, params.image_size, params.image_size, 3])
    for i in range(len(train_filenames)):
        img = cv2.imread(train_filenames[i])

        # rescale image
        img = cv2.resize(img, (params.image_size, params.image_size))
        img = img.astype(np.float32)

        images_data[i] = img

    logging.info("Starting restore")

    # Initialize tf.Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize the lookup table
        sess.run(model_spec['variable_init_op'])

        # Reload weights from the weights subdirectory
        save_path = os.path.join(args.model_dir, args.restore_from)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)

        embeddings = sess.run(model_intfs['output'], feed_dict={model_intfs['input']: images_data})
        # TODO: to check and visualize the embeddings
