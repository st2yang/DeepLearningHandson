import tensorflow as tf


class Classifier:
    def __init__(self, embeddings, labels, params):
        self.build_model(embeddings, labels, params)
        self.intfs = {'input': embeddings, 'output': self.predictions}

    def build_model(self, embeddings, labels, params):
        with tf.variable_scope('classify_fc_1'):
            out = tf.layers.dense(embeddings, int(params.embedding_size * 0.5))
            out = tf.nn.relu(out)
        with tf.variable_scope('classify_fc_2'):
            self.logits = tf.layers.dense(out, params.num_labels)

        self.predictions = tf.argmax(self.logits, 1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, self.predictions), tf.float32))
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=self.logits)