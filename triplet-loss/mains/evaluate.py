"""Evaluate the model"""

import argparse
import logging
import os

import tensorflow as tf

from helpers.input_fn import load_data
from models.tricls_model import TriClsModel
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

    # create the iterator over the dataset
    test_inputs = load_data('eval', data_dir, test_data_dir, params)

    # Define the model
    logging.info("Creating the model...")
    model = TriClsModel('eval', test_inputs, params, reuse=False)

    logging.info("Starting evaluation")
    restore_from = os.path.join(args.model_dir, "best_weights")
    evaluate(model.model_spec, params, restore_from)
