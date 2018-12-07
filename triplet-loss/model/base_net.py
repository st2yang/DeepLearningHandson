import tensorflow as tf


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
