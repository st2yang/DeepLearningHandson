"""Train the model"""

import argparse
import logging
import os

import tensorflow as tf

from helpers.input_fn import input_fn
from helpers.utils import Params
from helpers.utils import set_logger
from models.model_fn import model_fn
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

    # Get the filenames from the train and dev sets
    train_filenames = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir)
                       if f.endswith('.jpg')]
    eval_filenames = [os.path.join(dev_data_dir, f) for f in os.listdir(dev_data_dir)
                      if f.endswith('.jpg')]

    # Labels will be between 0 and 5 included (6 classes in total)
    train_labels = [int(f.split('/')[-1][0]) for f in train_filenames]
    eval_labels = [int(f.split('/')[-1][0]) for f in eval_filenames]

    # Specify the sizes of the dataset we train on and evaluate on
    params.train_size = len(train_filenames)
    params.eval_size = len(eval_filenames)

    # Create the two iterators over the two datasets
    train_inputs = input_fn('train', train_filenames, train_labels, params)
    eval_inputs = input_fn('eval', eval_filenames, eval_labels, params)

    # Define the model
    logging.info("Creating the model...")
    train_model_spec, _ = model_fn('train', train_inputs, params)
    eval_model_spec, _ = model_fn('eval', eval_inputs, params, reuse=True)

    # Train the model
    logging.info("Starting training.")
    restore_from = os.path.join(args.model_dir, "best_weights")
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, restore_from)
