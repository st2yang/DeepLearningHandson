import tensorflow as tf
from models.triplet_loss import batch_all_triplet_loss
from models.triplet_loss import batch_hard_triplet_loss


class TripletNet:
    def __init__(self, is_training, inputs, params):
        """triplet network

        Args:
            is_training: (bool) whether we are training or not
            inputs: this can be `tf.placeholder` or outputs of `tf.data`
            params: (Params) hyperparameters

        Returns:
            interfaces: embeddings, loss, intfs
        """
        self.is_training = is_training
        self.params = params
        images = inputs['images']
        labels = inputs['labels']
        time = inputs['time']
        self.embeddings = self.base_net(images)
        self.embedding_mean_norm = tf.reduce_mean(tf.norm(self.embeddings, axis=1))
        self.triplet_loss(labels, time)
        self.intfs = {'input': images, 'output': self.embeddings}

    def triplet_loss(self, labels, time):
        if self.params.triplet_strategy == "batch_all":
            self.loss, self.fraction, _ = batch_all_triplet_loss(labels, time, self.embeddings, self.params.margin,
                                                                 self.params.margin, 3, squared=self.params.squared)
        elif self.params.triplet_strategy == "batch_hard":
            self.loss = batch_hard_triplet_loss(labels, self.embeddings, margin=self.params.margin,
                                                squared=self.params.squared)
        else:
            raise ValueError("Triplet strategy not recognized: {}".format(self.params.triplet_strategy))

    def base_net(self, images):
        assert images.get_shape().as_list() == [None, self.params.image_size, self.params.image_size, 3]

        out = images
        # Define the number of channels of each convolution
        # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
        num_channels = self.params.num_channels
        bn_momentum = self.params.bn_momentum
        channels = [num_channels, num_channels * 2, num_channels * 4, num_channels * 8]
        for i, c in enumerate(channels):
            with tf.variable_scope('block_{}'.format(i + 1)):
                out = tf.layers.conv2d(out, c, 3, padding='same')
                if self.params.use_batch_norm:
                    out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=self.is_training)
                out = tf.nn.relu(out)
                out = tf.layers.max_pooling2d(out, 2, 2)

        assert out.get_shape().as_list() == [None, 4, 4, num_channels * 8]

        out = tf.reshape(out, [-1, 4 * 4 * num_channels * 8])
        with tf.variable_scope('fc_1'):
            out = tf.layers.dense(out, num_channels * 8)
            if self.params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=self.is_training)
            out = tf.nn.relu(out)
        with tf.variable_scope('fc_2'):
            out = tf.layers.dense(out, self.params.embedding_size)

        return out
