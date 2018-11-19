"""This is an TensorFLow implementation of SiameseTripletNet in the following paper

Paper:
(https://arxiv.org/abs/1505.00687)

"""

import numpy as np
import tensorflow as tf

from alexnet import AlexNet


class SiameseTripletNet(object):
    """Implementation of the SiameseTripletNet."""

    def __init__(self, x, keep_prob, num_classes, mode, train_layers='DEFAULT'):
        """Create the graph of the SiameseTripletNet model.

        Args:
            x: Placeholder for the input tensor.
        """

        # the size of triplet samples
        self.K = 4
        self.N = int(x.get_shape()[0] - 2)
        assert self.K < self.N

        # define alex_net
        if train_layers == 'DEFAULT':
            train_layers = ['fc8', 'fc7', 'fc6', 'conv5', 'conv4', 'conv3', 'conv2', 'conv1']
        self.alex_net = AlexNet(x, keep_prob, num_classes, train_layers)
        self.features = self.alex_net.fc8
        self.loss = tf.cond(mode, self.random_selection, self.hard_negative_mining)

    def load_alex_weights(self, sess):
        self.alex_net.load_initial_weights(sess)

    def random_selection(self):
        random_idx = np.random.permutation(self.N)
        idx = random_idx[0:self.K] + 2
        loss = self.alex_net.loss_l2
        for i in range(self.K):
            loss += hinge_loss(self.features[0], self.features[1], self.features[idx[i]])
        return loss

    def hard_negative_mining(self):
        loss_list = []
        for i in range(self.N):
            loss_list.append(hinge_loss(self.features[0], self.features[1], self.features[i]))
        loss_records = tf.stack(loss_list)
        loss = self.alex_net.loss_l2 + tf.reduce_sum(tf.nn.top_k(loss_records, k=self.K).values)
        return loss


def hinge_loss(anchor, positive, negative):
    M = 0.5
    loss = tf.maximum(0., cos_similarity(anchor, positive) -
                      cos_similarity(anchor, negative) + M)
    return loss


def cos_similarity(x1, x2):
    return tf.losses.cosine_distance(tf.nn.l2_normalize(x1, 0),
                                     tf.nn.l2_normalize(x2, 0), axis=0)


# test
# batch_size = 128
# num_classes = 1000
# x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
# y = tf.placeholder(tf.float32, [batch_size, num_classes])
# keep_prob = tf.placeholder(tf.float32)
# selection_mode = tf.placeholder(tf.bool)
# train_layers = ['fc8', 'fc7']
# model = SiameseTripletNet(x, keep_prob, num_classes, selection_mode, train_layers)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     model.load_alex_weights(sess)
