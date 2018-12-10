"""Define the model."""

import tensorflow as tf
from models.triplet_net import TripletNet
from models.classifier import Classifier


class TriClsModel:
    def __init__(self, mode, inputs, params, reuse=False):
        """Model class defining the graph operations.

        Args:
            mode: (string) can be 'train' or 'eval'
            inputs: (dict) contains the inputs of the graph (features, labels...)
                    this can be `tf.placeholder` or outputs of `tf.data`
            params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
            reuse: (bool) whether to reuse the weights

        Interfaces:
            model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
        """
        # super(TriClsModel, self).__init__(mode, inputs, params, reuse)
        self.mode = mode
        self.params = params
        self.reuse = reuse
        self.model_spec, self.tri_intfs, self.cls_intfs = self.build_model(inputs)

    def build_model(self, inputs):
        is_training = (self.mode == 'train')
        labels = inputs['labels']
        labels = tf.cast(labels, tf.int64)

        # -----------------------------------------------------------
        # MODEL: define the layers of the model
        with tf.variable_scope('triplet_net', reuse=self.reuse):
            # Compute the output distribution of the model and the predictions
            tri_model = TripletNet(is_training, inputs, self.params)

        with tf.variable_scope('classify_model', reuse=self.reuse):
            cls_model = Classifier(tri_model.embeddings, labels, self.params)

        # Define training step that minimizes the loss with the Adam optimizer
        if is_training:
            tri_global_step = tf.train.get_or_create_global_step()
            cls_global_step = tf.train.get_or_create_global_step()
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope="classify_model")
            cls_train_op = tf.train.AdamOptimizer(self.params.learning_rate). \
                minimize(cls_model.loss, global_step=cls_global_step, var_list=train_vars)
            if self.params.use_batch_norm:
                # Add a dependency to update the moving mean and variance for batch normalization
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    tri_train_op = tf.train.AdamOptimizer(self.params.learning_rate). \
                        minimize(tri_model.loss, global_step=tri_global_step)
            else:
                tri_train_op = tf.train.AdamOptimizer(self.params.learning_rate). \
                    minimize(tri_model.loss, global_step=tri_global_step)

        # -----------------------------------------------------------
        # METRICS AND SUMMARIES
        # Metrics for evaluation using tf.metrics (average over whole dataset)
        with tf.variable_scope("tri_metrics"):
            tri_metrics = {
                'loss': tf.metrics.mean(tri_model.loss),
                'embedding_mean_norm': tf.metrics.mean(tri_model.embedding_mean_norm)
            }
            if self.params.triplet_strategy == "batch_all":
                tri_metrics['fraction_positive_triplets'] = tf.metrics.mean(tri_model.fraction)

        with tf.variable_scope("cls_metrics"):
            cls_metrics = {
                'accuracy': tf.metrics.accuracy(labels=labels, predictions=cls_model.predictions),
                'loss': tf.metrics.mean(cls_model.loss)
            }

        # Group the update ops for the tf.metrics
        tri_update_metrics_op = tf.group(*[op for _, op in tri_metrics.values()])
        cls_update_metrics_op = tf.group(*[op for _, op in cls_metrics.values()])

        # Get the op to reset the local variables used in tf.metrics
        tri_metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="tri_metrics")
        tri_metrics_init_op = tf.variables_initializer(tri_metric_variables)
        cls_metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="cls_metrics")
        cls_metrics_init_op = tf.variables_initializer(cls_metric_variables)

        # Summaries for training
        tf.summary.scalar('tri_loss', tri_model.loss, collections=['triplet'])
        tf.summary.scalar("embedding_mean_norm", tri_model.embedding_mean_norm, collections=['triplet'])
        if self.params.triplet_strategy == "batch_all":
            tf.summary.scalar('fraction_positive_triplets', tri_model.fraction, collections=['triplet'])
        tf.summary.scalar('cls_loss', cls_model.loss, collections=['classify'])
        tf.summary.scalar('accuracy', cls_model.accuracy, collections=['classify'])

        # -----------------------------------------------------------
        # MODEL SPECIFICATION
        # Create the model specification and return it
        # It contains nodes or operations in the graph that will be used for training and evaluation
        model_spec = inputs
        model_spec['variable_init_op'] = tf.global_variables_initializer()
        model_spec['tri_related'] = {
            'iterator_init_op': inputs['iterator_init_op'],
            'loss': tri_model.loss,
            'metrics_init_op': tri_metrics_init_op,
            'metrics': tri_metrics,
            'update_metrics': tri_update_metrics_op,
            'summary_op': tf.summary.merge_all(key='triplet')}
        model_spec['cls_related'] = {
            'iterator_init_op': inputs['iterator_init_op'],
            'loss': cls_model.loss,
            'metrics_init_op': cls_metrics_init_op,
            'metrics': cls_metrics,
            'update_metrics': cls_update_metrics_op,
            'summary_op': tf.summary.merge_all(key='classify')}
        if is_training:
            model_spec['tri_related']['train_op'] = tri_train_op
            model_spec['cls_related']['train_op'] = cls_train_op

        return model_spec, tri_model.intfs, cls_model.intfs
