from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

# Wondering about the following error?:
#   2018-04-03 15:50:58.999473: I T:\src\github\tensorflow\tensorflow\core\platform\cpu_feature_guard.cc:140]
#   Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
# See:
#   https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
# TLDR; Don't care about AVX support because most expensive linear algebra operations will be handled via GPU. So
#   perform warning suppression:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
TensorFlow's eager execution is an imperative programming environment that evaluates operations immediately, 
    without an extra graph-building step. Operations return concrete values instead of constructing a 
    computational graph to run later. Once eager execution is enabled, it cannot be disabled within the same program.
'''
tf.enable_eager_execution()

# Print the version of TensorFlow that is installed:
print("TensorFlow version: {}".format(tf.VERSION))
# Print a boolean flag indicating if eager execution is enabled:
print("Eager execution: {}".format(tf.executing_eagerly()))

# Download the iris dataset:
train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
# load the dataset via the high-level Keras API:
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)
print("Local copy of the dataset file: {}".format(train_dataset_fp))


def parse_csv(line):
    """
    parse_csv: Parses the feature and label values into a format TensorFlow can use. Each line/row in the file will be
        passed to this method which grabs the first four feature fields and combines them into a single Tensor. The last
        feild is parsed as a label tensor.
    :param line: A line from the input csv to be extracted into features and labels.
    :returns features, label: Two tensors containing the features and labels of the provided line.
        :return features: The four features (sepal_length, sepal_width, petal_length, petal_width) as a Tensor.
        :return label: The label of the input line as a Tensor.
    """
    # Set the field types to a float, and the label to an int:
    example_defaults = [[0.], [0.], [0.], [0.], [0]]
    parsed_line = tf.decode_csv(line, example_defaults)
    # First 4 fields are features, combine into single 4x1 tensor
    features = tf.reshape(parsed_line[:-1], shape=(4,))
    # Last field is the label
    label = tf.reshape(parsed_line[-1], shape=())
    return features, label

'''
TensorFlow's Dataset API handles many common cases for feeding data into a model. This is a high-level API for 
    reading data and transforming it into a form used for training.  A tf.data.Dataset represents an input pipeline 
    as a collection of elements and a series of transformations that act on those elements.
'''
# This program uses tf.data.TextLineDataset to load a CSV-formatted text file:
train_dataset = tf.data.TextLineDataset(train_dataset_fp)
train_dataset = train_dataset.skip(1)       # skip the first header row of the csv
train_dataset = train_dataset.map(parse_csv)        # Use the pre-defined function to parse each row into Tensors
train_dataset = train_dataset.shuffle(buffer_size=1000)     # Training works best if examples are in a random order.
# Here we specify a batch size to train the model faster using 32 examples per-batch:
train_dataset = train_dataset.batch(32)
# Now we can view a single example entry from a batch
features, label = tfe.Iterator(train_dataset).next()
print("example features:", features[0])
print("example label:", label[0])

'''
Creating the model using Keras:
In this case the model is two dense (fully connected) layers with 10 nodes each and an output layer with 3 nodes which
    represent the label predictions. 
'''
# The first layer's input_shape corresponds to the amount of features from the dataset and is required.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=10, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(units=10, activation='relu'),
    tf.keras.layers.Dense(units=3)
])

'''
Training the model
'''


def loss(model, x, y):
    """
    loss: Computes the loss of the model (how far off its predictions are), in this case via softmax cross entropy
        performed on a sparse matrix.
    :param model: An instance of the Keras model.
    :param x: The input data.
    :param y: The predicted labels.
    :return loss: The loss of the model in the form of a
    """
    # Get the output of the model before conversion to class labels (logits).
    y_ = model(x)
    # Use tf.losses.sparse_softmax_cross_entropy to calculate loss.
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(model, inputs, targets):
    """
    grad: Uses the previously defined loss function and the tfe.GradientTape to record operations that compute the
        gradients used to optimize the model.
    :param model: An instance of the Keras model.
    :param inputs: The input data.
    :param targets: The labels for the input data.
    :return tape.gradient: A tfe.GradientTape which records the operations performed to compute the gradients used
        during model optimization.
    """
    with tfe.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)


'''
Create an optimizer.
'''
# This model uses the tf.train.GradientDescentOptimizer which implements the standard gradient descent (SGD) algorithm.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

'''
Train the model:
'''
# Retain the results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    # Training loop - use mini-batches of 32
    for x, y in tfe.Iterator(train_dataset):
        # Optimize the model
        grads = grad(model=model, inputs=x, targets=y)
        optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step=tf.train.get_or_create_global_step())
        # Track progress
        epoch_loss_avg(loss(model=model, x=x, y=y))     # add current mini-batch loss
        # compare predicted label to actual label:
        epoch_accuracy(tf.argmax(input=model(x), axis=1, output_type=tf.int32), y)

    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    # Print the loss and accuracy every 50 epochs:
    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

'''
Visualize the loss function over time:
'''
# Tensorflow utilizes TensorBoard but we can just use matplotlib for simplicity and familiarity:
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')
axes[0].set_ylabel('Loss', fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel('Accuracy', fontsize=14)
axes[1].set_xlabel('Epoch', fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()

'''
Evaluate the model on the test dataset:
'''
test_url = "http://download.tensorflow.org/data/iris_test.csv"
# This program uses tf.data.TextLineDataset to load a CSV-formatted text file:
test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)
test_dataset = tf.data.TextLineDataset(test_fp)
test_dataset = test_dataset.skip(1)             # skip header row
test_dataset = test_dataset.map(parse_csv)      # parse each row with the function created earlier
test_dataset = test_dataset.shuffle(1000)       # randomize
test_dataset = test_dataset.batch(32)           # use the same batch size as the training set

test_accuracy = tfe.metrics.Accuracy()
for (x, y) in tfe.Iterator(test_dataset):
    prediction = tf.argmax(input=model(x), axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)
print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

'''
Use the trained model to make predictions:
'''
# Save the (index, class) mapping.
class_ids = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
# Provide manually three unlabeled feature vectors for classification:
predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5],
    [5.9, 3.0, 4.2, 1.5,],
    [6.9, 3.1, 5.4, 2.1]
])
predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    name = class_ids[class_idx]
    print("Example {} prediction: {}".format(i, name))
