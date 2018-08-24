import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('tmp/train/', one_hot=True)

tf.reset_default_graph()
# Define a simple convolutional layer
def conv_layer(inputs, in_channels, out_channels):
    # Weights:
    w = tf.Variable(tf.zeros([5, 5, in_channels, out_channels]))
    # Biases:
    b = tf.Variable(tf.zeros([out_channels]))
    # Convolution operator:
    conv = tf.nn.conv2d(inputs, w, strides=[1,1,1,1], padding="SAME")
    # Activation function:
    act = tf.nn.relu(conv + b)
    return act

# Define a fully connected layer
def fc_layer(inputs, in_channels, out_channels):
    # Weights:
    w = tf.Variable(tf.zeros([in_channels, out_channels]))
    # Biases:
    b = tf.Variable(tf.zeros([out_channels]))
    # Activation function:
    act = tf.nn.relu(tf.matmul(inputs, w) + b)
    return act

# Setup placeholders, and reshape the data:
x = tf.placeholder(tf.float32, shape=[None, 784], name='x-input')
y = tf.placeholder(tf.int64, shape=[None, 10], name='y-input')
x_image = tf.reshape(x, [-1, 28, 28, 1])

''' Create the network: '''
conv1 = conv_layer(inputs=x_image, in_channels=1, out_channels=32)
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

conv2 = conv_layer(inputs=pool1, in_channels=32, out_channels=64)
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
flattened = tf.reshape(pool2, [-1, 7 * 7 * 64])

# two fully connected layers:
fc1 = fc_layer(flattened, 7 * 7 * 64, 1024)
logits = fc_layer(fc1, 1024, 10)

''' Loss and Training: '''
# Compute cross entropy as our loss function:
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# Use an AdamOptimizer to train the network:
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Compute the accuracy so we can print to console:
a_ = tf.argmax(logits, 1)
print(a_)
b_ = tf.argmax(y, 1)
print(b_)
# correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
correct_prediction = tf.equal(a_, b_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

''' Evaluation Step: '''
# mnist = input_data.read_data_sets(train_dir='train')
sess = tf.InteractiveSession()
# Initialize all the variables:
sess.run(tf.global_variables_initializer())

# Train for 2000 steps:
for i in range(2000):
    xs, ys = mnist.train.next_batch(100)
    print(xs.shape)
    print(ys.shape)

    # Occasionally report accuracy:
    if i % 500 == 0:
        [train_accuracy] = sess.run([accuracy], feed_dict={x: xs, y: ys})
        print("step %d, training accuracy %g" % (i, train_accuracy))

sess.close()
