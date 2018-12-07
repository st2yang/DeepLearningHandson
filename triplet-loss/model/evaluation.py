"""Tensorflow utility functions for evaluation"""

import logging
import os

import tensorflow as tf

from model.utils import save_dict_to_json


def evaluate_sess(sess, model_spec, num_steps, writer=None, params=None):
    """Train the model on `num_steps` batches.

    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries. Is None if we don't log anything
        params: (Params) hyperparameters
    """
    update_metrics = model_spec['update_metrics']
    eval_metrics = model_spec['metrics']
    global_step = tf.train.get_global_step()

    # Load the evaluation dataset into the pipeline and initialize the metrics init op
    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])

    # compute metrics over the dataset
    for _ in range(num_steps):
        sess.run(update_metrics)

    # Get the values of the metrics
    metrics_values = {k: v[0] for k, v in eval_metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Eval metrics: " + metrics_string)

    # Add summaries manually to writer at global_step_val
    if writer is not None:
        global_step_val = sess.run(global_step)
        for tag, val in metrics_val.items():
            summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
            writer.add_summary(summ, global_step_val)

    return metrics_val


def evaluate(eval_model_spec, params, restore_from):
    """Evaluate the model

    Args:
        eval_model_spec: (dict) contains the graph operations or nodes needed for evaluation
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """
    # Initialize tf.Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize the lookup table
        sess.run(eval_model_spec['variable_init_op'])

        # Reload weights from the weights subdirectory
        if os.path.isdir(restore_from) and os.listdir(restore_from):
            save_path = tf.train.latest_checkpoint(restore_from)
            saver.restore(sess, save_path)
        else:
            raise ValueError("No checkpoint folder or file")

        # Evaluate
        num_steps = (params.eval_size + params.batch_size - 1) // params.batch_size
        evaluate_sess(sess, eval_model_spec['tri_related'], num_steps)
        evaluate_sess(sess, eval_model_spec['cls_related'], num_steps)
