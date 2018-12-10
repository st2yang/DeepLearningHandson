"""Train the model"""

import argparse
import logging
import os

import tensorflow as tf

from helpers.input_fn import load_data
from helpers.utils import Params
from helpers.utils import set_logger
from models.tricls_model import TriClsModel
from helpers.training import train_and_evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='../experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='../data/default',
                    help="Directory containing the dataset")

if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, "train")
    dev_data_dir = os.path.join(data_dir, "dev")

    train_inputs = load_data('train', data_dir, train_data_dir, params)
    eval_inputs = load_data('eval', data_dir, dev_data_dir, params)

    # Define the model
    logging.info("Creating the model...")
    train_model = TriClsModel('train', train_inputs, params)
    eval_model = TriClsModel('eval', eval_inputs, params, reuse=True)

    # Train the model
    logging.info("Starting training.")
    restore_from = os.path.join(args.model_dir, "best_weights")
    train_and_evaluate(train_model.model_spec, eval_model.model_spec, args.model_dir, params, restore_from)
