# the techniques used in the MLP:
# architecture: input layer-hidden layer(500 units)-output layer
# weights initialization: truncated_normal_initializer
# regularization: L2
# activation: relu
# optimization: SGD with decay learning rate
# accuracy: 0.9846

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

num_input = 784
num_class = 10
num_hidden1 = 500
batch_size = 100
learning_rate_base = 0.8
learning_rate_decay = 0.99
regularization_rate = 0.0001
training_steps = 30000
moving_average_decay = 0.99


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    else:
        tf.add_to_collection('losses', 0.0)

    return weights


# define the forward network
def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([num_input, num_hidden1], regularizer)
        biases = tf.get_variable("biases", [num_hidden1], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([num_hidden1, num_class], regularizer)
        biases = tf.get_variable("biases", [num_class], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2


x = tf.placeholder(tf.float32, [None, num_input], name='x-input')
logits = tf.placeholder(tf.float32, [None, num_class], name='y-input')

regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)

y = inference(x, regularizer)
global_step = tf.Variable(0, trainable=False)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(logits, 1))
cross_entropy_mean = tf.reduce_mean(cross_entropy)
loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, mnist.train.num_examples / batch_size, learning_rate_decay)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for i in range(training_steps):
        xs, ys = mnist.train.next_batch(batch_size)
        _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, logits: ys})

        if i % 1000 == 0:
            print("After %d training step(s), loss on training batch is %g." % (step, loss_value))

    validate_feed = {x: mnist.validation.images, logits: mnist.validation.labels}
    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
    print("validation accuracy = %g" % accuracy_score)