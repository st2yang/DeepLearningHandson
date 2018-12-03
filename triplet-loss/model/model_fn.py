"""Define the model."""

import tensorflow as tf
from model.triplet_loss import batch_all_triplet_loss
from model.triplet_loss import batch_hard_triplet_loss


class BaseNet:
    def __init__(self, is_training, images, params):
        self.out = self.build_model(is_training, images, params)

    def build_model(self, is_training, images, params):
        """Compute logits of the model (output distribution)

        Args:
            is_training: (bool) whether we are training or not
            images: this can be `tf.placeholder` or outputs of `tf.data`
            params: (Params) hyperparameters

        Returns:
            output: (tf.Tensor) output of the model
        """

        assert images.get_shape().as_list() == [None, params.image_size, params.image_size, 3]

        out = images
        # Define the number of channels of each convolution
        # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
        num_channels = params.num_channels
        bn_momentum = params.bn_momentum
        channels = [num_channels, num_channels * 2, num_channels * 4, num_channels * 8]
        for i, c in enumerate(channels):
            with tf.variable_scope('block_{}'.format(i + 1)):
                out = tf.layers.conv2d(out, c, 3, padding='same')
                if params.use_batch_norm:
                    out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
                out = tf.nn.relu(out)
                out = tf.layers.max_pooling2d(out, 2, 2)

        assert out.get_shape().as_list() == [None, 4, 4, num_channels * 8]

        out = tf.reshape(out, [-1, 4 * 4 * num_channels * 8])
        with tf.variable_scope('fc_1'):
            out = tf.layers.dense(out, num_channels * 8)
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
        with tf.variable_scope('fc_2'):
            out = tf.layers.dense(out, params.embedding_size)

        return out


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        images = inputs['images']
        model = BaseNet(is_training, images, params)
        embeddings = model.out
        model_intfs = {'input': images, 'output': embeddings}
    embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))
    tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)

    # Define triplet loss
    if params.triplet_strategy == "batch_all":
        loss, fraction = batch_all_triplet_loss(labels, embeddings, margin=params.margin,
                                                squared=params.squared)
    elif params.triplet_strategy == "batch_hard":
        loss = batch_hard_triplet_loss(labels, embeddings, margin=params.margin,
                                       squared=params.squared)
    else:
        raise ValueError("Triplet strategy not recognized: {}".format(params.triplet_strategy))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'loss': tf.metrics.mean(loss),
            'embedding_mean_norm': tf.metrics.mean(embedding_mean_norm)
        }
        if params.triplet_strategy == "batch_all":
            metrics['fraction_positive_triplets'] = tf.metrics.mean(fraction)

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    if params.triplet_strategy == "batch_all":
        tf.summary.scalar('fraction_positive_triplets', fraction)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec['loss'] = loss
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec, model_intfs
