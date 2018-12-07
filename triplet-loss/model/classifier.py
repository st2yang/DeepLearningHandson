import tensorflow as tf


class Classifier:
    def __init__(self, embeddings, params):
        self.logits = self.build_model(embeddings, params)

    def build_model(self, embeddings, params):
        with tf.variable_scope('classify_fc_1'):
            out = tf.layers.dense(embeddings, int(params.embedding_size * 0.5))
            out = tf.nn.relu(out)
        with tf.variable_scope('classify_fc_2'):
            logits = tf.layers.dense(out, params.num_classes)

        return logits
