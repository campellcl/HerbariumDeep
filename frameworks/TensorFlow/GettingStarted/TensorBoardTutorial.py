import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('tmp/train/', one_hot=True)

tf.reset_default_graph()


# Define a simple convolutional layer WITH NAMES:
def conv_layer(inputs, in_channels, out_channels, name="conv"):
    with tf.name_scope(name):
        # Weights:
        w = tf.Variable(tf.truncated_normal([5, 5, in_channels, out_channels], stddev=0.1), name="W")
        # Biases:
        b = tf.Variable(tf.constant(0.1, shape=[out_channels]), name="B")
        # Convolution operator:
        conv = tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding="SAME")
        # Activation function:
        act = tf.nn.relu(conv + b)
        # Summaries for logging:
        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', act)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# Define a fully connected layer WITH NAMES:
def fc_layer(inputs, in_channels, out_channels, name="fc"):
    with tf.name_scope(name):
        # Weights:
        w = tf.Variable(tf.truncated_normal([in_channels, out_channels], stddev=0.1), name="W")
        # Biases:
        b = tf.Variable(tf.constant(0.1, shape=[out_channels]), name="B")
        # Activation function:
        return tf.nn.relu(tf.matmul(inputs, w) + b)


# Setup placeholders, and reshape the data:
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
x_image = tf.reshape(x, [-1, 28, 28, 1])
y = tf.placeholder(tf.int64, shape=[None, 10], name='labels')

''' Create the network WITH NAMES: '''
conv1 = conv_layer(inputs=x_image, in_channels=1, out_channels=32, name="conv1")
# pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

conv2 = conv_layer(inputs=conv1, in_channels=32, out_channels=64, name="conv2")
# pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

flattened = tf.reshape(conv2, [-1, 7 * 7 * 64])

# two fully connected layers WITH NAMES:
fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, name='fc1')
logits = fc_layer(fc1, 1024, 10, name='fc2')

''' Loss and Training WITH NAMES: '''
# Compute cross entropy as our loss function:
with tf.name_scope("cross-entro"):
    cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=tf.argmax(y, 1)))

# Use an AdamOptimizer to train the network:
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Compute the accuracy so we can print to console:
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

''' Evaluation Step: '''
sess = tf.InteractiveSession()

''' Let's visualize the default computational graph & some scalars '''
tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('accuracy', accuracy)
tf.summary.image('input', x_image, 3)

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("tmp/log/5")
writer.add_graph(sess.graph)

# Initialize all the variables:
sess.run(tf.global_variables_initializer())

# Train for 2000 steps:
for i in range(2001):
    xs, ys = mnist.train.next_batch(100)
    #     print(xs.shape)
    #     print(ys.shape)

    # Occasionally report accuracy:
    if i % 5 == 0:
        # Run the merged summaries and write the contents to disk for visualization:
        s = sess.run(merged_summary, feed_dict={x: xs, y: ys})
        writer.add_summary(s, i)
        [train_accuracy] = sess.run([accuracy], feed_dict={x: xs, y: ys})
        print("step %d, training accuracy %g" % (i, train_accuracy))

tf.InteractiveSession.close(sess)
