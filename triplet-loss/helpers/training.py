"""Tensorflow utility functions for training"""

import logging
import os

from tqdm import trange
import tensorflow as tf

from helpers.utils import save_dict_to_json
from helpers.evaluation import evaluate_sess


def train_sess(sess, train_spec, num_steps, writer, params):
    # Get relevant graph operations or nodes needed for training
    loss = train_spec['loss']
    train_op = train_spec['train_op']
    update_metrics = train_spec['update_metrics']
    metrics = train_spec['metrics']
    summary_op = train_spec['summary_op']
    # TODO: see if global_step has effect on optimizer and model save
    global_step = tf.train.get_global_step()

    # Load the training dataset into the pipeline and initialize the metrics local variables
    sess.run(train_spec['iterator_init_op'])
    sess.run(train_spec['metrics_init_op'])

    # Use tqdm for progress bar
    t = trange(num_steps)
    for i in t:
        # Evaluate summaries for tensorboard only once in a while
        if i % params.save_summary_steps == 0:
            # Perform a mini-batch update
            _, _, loss_val, summ, global_step_val = sess.run([train_op, update_metrics, loss,
                                                              summary_op, global_step])
            # Write summaries for tensorboard
            writer.add_summary(summ, global_step_val)
        else:
            _, _, loss_val = sess.run([train_op, update_metrics, loss])
        # Log the loss in the tqdm progress bar
        t.set_postfix(loss='{:05.3f}'.format(loss_val))

    metrics_values = {k: v[0] for k, v in metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(train_model_spec, eval_model_spec, model_dir, params, restore_from=None):
    """Train the model and evaluate every epoch.

    Args:
        train_model_spec: (dict) contains the graph operations or nodes needed for training
        eval_model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """
    # Initialize tf.Saver instances to save weights during training
    last_saver = tf.train.Saver() # will keep last 5 epochs
    best_saver = tf.train.Saver(max_to_keep=1)  # only keep 1 best checkpoint (best on eval)
    epoch_count = 0

    with tf.Session() as sess:
        # Initialize model variables
        sess.run(train_model_spec['variable_init_op'])

        # Reload weights from directory if specified
        if os.path.isdir(restore_from) and os.listdir(restore_from):
            logging.info("Restoring parameters from {}".format(restore_from))
            restore_from = tf.train.latest_checkpoint(restore_from)
            epoch_count = int(restore_from.split('-')[-1])
            last_saver.restore(sess, restore_from)

        # For tensorboard (takes care of writing summaries to files)
        train_writer = tf.summary.FileWriter(os.path.join(model_dir, 'train_summaries'), sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(model_dir, 'eval_summaries'), sess.graph)

        for epoch in range(epoch_count, params.tri_num_epochs):
            # Run one epoch
            logging.info("Tri Epoch {}/{}".format(epoch + 1, params.tri_num_epochs))
            # Compute number of batches in one epoch (one full pass over the training set)
            num_steps = (params.train_size + params.batch_size - 1) // params.batch_size
            train_sess(sess, train_model_spec['tri_related'], num_steps, train_writer, params)
            epoch_count = epoch_count + 1

            # Evaluate for one epoch on validation set
            num_steps = (params.eval_size + params.batch_size - 1) // params.batch_size
            metrics = evaluate_sess(sess, eval_model_spec['tri_related'], num_steps, eval_writer)

            best_eval_loss = float("inf")
            # If best_eval, best_save_path
            eval_loss = metrics['loss']
            if eval_loss < best_eval_loss:
                # Store new best accuracy
                best_eval_loss = eval_loss
                # Save weights
                best_save_path = os.path.join(model_dir, 'best_weights', 'after-epoch')
                best_save_path = best_saver.save(sess, best_save_path, global_step=epoch + 1)
                logging.info("- Found new best tri_loss, saving in {}".format(best_save_path))
                # Save best eval metrics in a json file in the model directory
                best_json_path = os.path.join(model_dir, "metrics_eval_best_weights.json")
                save_dict_to_json(metrics, best_json_path)

        for epoch in range(epoch_count, params.tri_num_epochs + params.cls_num_epochs):
            # Run one epoch
            logging.info("Cls Epoch {}/{}".format(epoch + 1, params.tri_num_epochs + params.cls_num_epochs))
            # Compute number of batches in one epoch (one full pass over the training set)
            num_steps = (params.train_size + params.batch_size - 1) // params.batch_size
            train_sess(sess, train_model_spec['cls_related'], num_steps, train_writer, params)

            # Evaluate for one epoch on validation set
            num_steps = (params.eval_size + params.batch_size - 1) // params.batch_size
            metrics = evaluate_sess(sess, eval_model_spec['cls_related'], num_steps, eval_writer)

            best_eval_acc = 0
            # If best_eval, best_save_path
            eval_acc = metrics['accuracy']
            if eval_acc > best_eval_acc:
                # Store new best accuracy
                best_eval_acc = eval_acc
                # Save weights
                best_save_path = os.path.join(model_dir, 'best_weights', 'after-epoch')
                best_save_path = best_saver.save(sess, best_save_path, global_step=epoch + 1)
                logging.info("- Found new best accuracy, saving in {}".format(best_save_path))
                # Save best eval metrics in a json file in the model directory
                best_json_path = os.path.join(model_dir, "metrics_eval_best_weights.json")
                save_dict_to_json(metrics, best_json_path)
