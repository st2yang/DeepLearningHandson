"""Evaluate the model"""

import argparse
import logging
import os

import tensorflow as tf

from helpers.input_fn import input_fn
from models.model_fn import model_fn
from helpers.evaluation import evaluate
from helpers.utils import Params
from helpers.utils import set_logger


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='../experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='../data/default',
                    help="Directory containing the dataset")

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
    test_inputs = input_fn('eval', test_filenames, test_labels, params)

    # Define the model
    logging.info("Creating the model...")
    model_spec, _ = model_fn('eval', test_inputs, params, reuse=False)

    logging.info("Starting evaluation")
    restore_from = os.path.join(args.model_dir, "best_weights")
    evaluate(model_spec, params, restore_from)
