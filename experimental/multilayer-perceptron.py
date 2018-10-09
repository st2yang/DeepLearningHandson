import tensorflow as tf
import numpy as np

# Import MNIST data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

# Parameters
learning_rate = 0.01
num_steps = 40
batch_size = 50

# Network Parameters
num_input = 28 * 28  # MNIST data input (img shape: 28*28)
n_hidden_1 = 300  # 1st layer number of neurons
n_hidden_2 = 100  # 2nd layer number of neurons
num_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder(tf.float32, shape=(None, num_input), name="X")
y = tf.placeholder(tf.int32, shape=None, name="y")


# turns out to be very important for accuracy
# ?reason: use truncated norm distribution instead of normal distribution ensures there
# won't be any large weights, which could slow down training
def weight_init(num_in, num_out):
    stddev = 2 / np.sqrt(num_in)
    init = tf.truncated_normal((num_in, num_out), stddev=stddev)
    W = tf.Variable(init, name="kernel")
    return W


# Store layers weight & bias
weights = {
    'h1': weight_init(num_input, n_hidden_1),
    'h2': weight_init(n_hidden_1, n_hidden_2),
    'out': weight_init(n_hidden_2, num_classes)
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: X_batch, y: y_batch})
        acc = sess.run(accuracy, feed_dict={X: X_valid, y: y_valid})
        print(step, "Val accuracy:", acc)

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: X_test,
                                      y: y_test}))
