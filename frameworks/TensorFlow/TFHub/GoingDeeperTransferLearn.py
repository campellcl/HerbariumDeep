import argparse
import sys
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.saved_model import tag_constants
import collections
import os
import re
import random
import numpy as np
from datetime import datetime
from sklearn import model_selection
import time

CMD_ARG_FLAGS = None
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
# A module is understood as instrumented for quantization with TF-Lite
# if it contains any of these ops.
FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars',
                  'FakeQuantWithMinMaxVarsPerChannel')
CHECKPOINT_DIR = 'tmp/_retrain_checkpoint'


def prepare_tensor_board_directories():
    """
    prepare_tensor_board_directories: Ensures that if a TensorBoard storage directory is defined in the command line
        flags, that said directory is purged of old TensorBoard files, and that this program has sufficient permissions
        to write new TensorBoard summaries to the specified path.
    :return:
    """
    # Check to see if the file exists:
    if tf.gfile.Exists(CMD_ARG_FLAGS.summaries_dir):
        # Delete everything in the file recursively:
        tf.gfile.DeleteRecursively(CMD_ARG_FLAGS.summaries_dir)
    # Re-create (or create for the first time) the storage directory:
    tf.gfile.MakeDirs(CMD_ARG_FLAGS.summaries_dir)
    # Check to see if intermediate computational graphs are to be stored:
    if CMD_ARG_FLAGS.intermediate_store_frequency > 0:
        if not os.path.exists(CMD_ARG_FLAGS.intermediate_output_graphs_dir):
            os.makedirs(CMD_ARG_FLAGS.intermediate_output_graphs_dir)
    return


def partition_into_image_lists(image_dir, train_percent=.80, val_percent=.20, test_percent=.20, random_state=0):
    """
    create_image_lists: Creates a dictionary containing all available datasets (train, test, validate) as a list of
        image file paths (indexed by the class label).
    :param train_percent: What percentage of the training data is to remain in the training set.
    :param test_percent: What percentage of the training data is to be allocated to a testing set.
    :param val_percent: What percentage of the remaining training data (after removing test set) is to be allocated
        for a validation set.
    :param random_state: A seed for the random number generator controlling the stratified partitioning.
    :return image_lists: A dictionary containing all available datasets (train, test, validate) as a list of file paths.
    """

    '''
    Check to see if the root directory exists. We use tf.gfile which is a C++ FileSystem API wrapper for the Python
        file API that also supports Google Cloud Storage and HDFS. For more information see:
        https://stackoverflow.com/questions/42256938/what-does-tf-gfile-do-in-tensorflow
    '''
    image_dir = CMD_ARG_FLAGS.image_dir
    if not tf.gfile.Exists(image_dir):
        tf.logging.error("Root image directory '" + image_dir + "' not found.")
        return None

    accepted_extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    image_lists = collections.OrderedDict()

    # TODO: This tf.gfile.Walk takes a very long time, maybe go async? It seems to cache the walk somehow...
    sub_dirs = sorted(x[0] for x in tf.gfile.Walk(image_dir))

    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if is_root_dir:
            is_root_dir = False
            # Skip the root_dir:
            continue
        if dir_name == image_dir:
            # Return control to beginning of for-loop:
            continue
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in accepted_extensions:
            # Get a list of all accepted file extensions and the targeted file_name:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            # Append all items from the file_glob to the list of files (if extension exists):
            file_list.extend(tf.gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning(msg='No files found in \'%s\'. Class label omitted from data sets.' % dir_name)
            # Return control to beginning of for-loop:
            continue
        if len(file_list) < 20:
            tf.logging.warning('WARNING: Folder has less than 20 images, which may cause issues. See: %s for info.'
                               % 'https://stackoverflow.com/questions/38175673/critical-tensorflowcategory-has-no-images-validation')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            tf.logging.warning(
                'WARNING: Folder {} has more than {} images. Some images will '
                'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        # label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        label_name = dir_name.lower()

        ''' Train, val, test spits: '''
        # There isn't a point in stratifying on a class-by-class basis when performing Proportionate allocation
        # train_images, test_images = model_selection.train_test_split(
        #     file_list, train_size=train_percent,
        #     test_size=test_percent, shuffle=True,
        #     stratify=[label_name for _ in range(len(file_list))], random_state=random_state)
        train_images, test_images = model_selection.train_test_split(
            file_list, train_size=train_percent,
            test_size=test_percent, shuffle=True,
            random_state=random_state
        )

        train_images, val_images = model_selection.train_test_split(
            train_images, train_size=train_percent,
            test_size=val_percent, shuffle=True,
            random_state=random_state
        )

        image_lists[label_name] = {
            'dir': dir_name,
            'train': train_images,
            'test': test_images,
            'val': val_images
        }
    return image_lists


def create_module_graph(module_spec):
    """
    create_module_graph: Creates a tensorflow graph from the provided TFHub module.
    source: https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py
    :param module_spec: the hub.ModuleSpec for the image module being used.
    :returns:
        :return graph: The tf.Graph that was created.
        :return bottleneck_tensor: The bottleneck values output by the module.
        :return resized_input_tensor: The input images, resized as expected by the module.
        :return wants_quantization: A boolean value, whether the module has been instrumented with fake quantization
            ops.
    """
    # tf.reset_default_graph()
    # Define the receptive field in accordance with the chosen architecture:
    height, width = hub.get_expected_image_size(module_spec)
    # Create a new default graph:
    with tf.Graph().as_default() as graph:
        with tf.variable_scope('source_model'):
            # Create a placeholder tensor for input to the model.
            resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3], name='resized_input')
            with tf.variable_scope('pre-trained_hub_module'):
                # Declare the model in accordance with the chosen architecture:
                m = hub.Module(module_spec, name='inception_v3_hub')
                # Create another place holder tensor to catch the output of the pre-activation layer:
                bottleneck_tensor = m(resized_input_tensor)
                # Give a name to this tensor:
                tf.identity(bottleneck_tensor, name='bottleneck-pre-activation')
                # This is a boolean flag indicating whether the module has been put through TensorFlow Light and optimized.
                wants_quantization = any(node.op in FAKE_QUANT_OPS
                                         for node in graph.as_graph_def().node)
    return graph, bottleneck_tensor, resized_input_tensor, wants_quantization


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def add_final_retrain_ops(class_count, final_tensor_name, bottleneck_tensor, quantize_layer, is_training):
    """
    add_final_retrain_ops: Adds a new softmax and fully-connected layer for training and model evaluation. In order to
        use the TFHub model as a fixed feature extractor, we need to retrain the top fully connected layer of the graph
        that we previously added in the 'create_module_graph' method. This function adds the right ops to the graph,
        along with some variables to hold the weights, and then sets up all the gradients for the backward pass.

        The set up for the softmax and fully-connected layers is based on:
        https://www.tensorflow.org/tutorials/mnist/beginners/index.html
    :source https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py
    :modified_by: Chris Campell
    :param class_count: An Integer representing the number of new classes we are trying to distinguish between.
    :param final_tensor_name: A name string for the final node that produces the fine-tuned results.
    :param bottleneck_tensor: The output of the main CNN graph (the specified TFHub module).
    :param quantize_layer: Boolean, specifying whether the newly added layer should be
        instrumented for quantization with TF-Lite.
    :param is_training: Boolean, specifying whether the newly add layer is for training
        or eval.
    :returns : The tensors for the training and cross entropy results, tensors for the
        bottleneck input and ground truth input, a reference to the optimizer for archival purposes and use in the
        hyper-string representation of this training run.
    """
    # The batch size
    batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
    assert batch_size is None, 'We want to work with arbitrary batch size when ' \
                               'constructing fully-connected and softmax layers for fine-tuning.'

    # Tensor declarations:
    with tf.variable_scope('re-train_ops'):
        with tf.name_scope('input'):
            # Create a placeholder Tensor of same type as bottleneck_tensor to cache output from TFHub module:
            bottleneck_input = tf.placeholder_with_default(
                bottleneck_tensor,
                shape=[batch_size, bottleneck_tensor_size],
                name='BottleneckInputPlaceholder'
            )

            # Another placeholder Tensor to hold the true class labels
            ground_truth_input = tf.placeholder(
                tf.int64,
                shape=[batch_size],
                name='GroundTruthInput'
            )

        # Additional organization for TensorBoard:
        layer_name = 'final_retrain_ops'
        with tf.name_scope(layer_name):
            # Every layer has the following items:
            with tf.name_scope('weights'):
                # Output random values from truncated normal distribution:
                initial_value = tf.truncated_normal(
                    shape=[bottleneck_tensor_size, class_count],
                    stddev=0.001
                )
                layer_weights = tf.Variable(initial_value=initial_value, name='final_weights')
                # variable_summaries(layer_weights)

            with tf.name_scope('biases'):
                layer_biases = tf.Variable(initial_value=tf.zeros([class_count]), name='final_biases')
                # variable_summaries(layer_biases)

            # pre-activations:
            with tf.name_scope('Wx_plus_b'):
                logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
                tf.summary.histogram('pre_activations(logits)', logits)

        # This is the tensor that will hold the predictions of the fine-tuned (re-trained) model:
        final_tensor = tf.nn.softmax(logits=logits, name=final_tensor_name)

        # The tf.contrib.quantize functions rewrite the graph in place for
        # quantization. The imported model graph has already been rewritten, so upon
        # calling these rewrites, only the newly added final layer will be
        # transformed.
        if quantize_layer:
            if is_training:
                tf.contrib.quantize.create_training_graph()
            else:
                tf.contrib.quantize.create_eval_graph()

        # We will keep a histogram showing the distribution of activation functions.
        tf.summary.histogram('activations', final_tensor)

        # If this is an eval graph, we don't need to add loss ops or an optimizer.
        if not is_training:
            return None, None, bottleneck_input, ground_truth_input, final_tensor, 'No optimizer'

        with tf.name_scope('cross_entropy'):
            # What constitutes sparse in this case?:
            cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(labels=ground_truth_input, logits=logits)

        tf.summary.scalar('cross_entropy', cross_entropy_mean)

        with tf.name_scope('train'):
            optimizer = tf.train.MomentumOptimizer(learning_rate=CMD_ARG_FLAGS.learning_rate, momentum=0.9)
            # TODO: Can we make this not hard-coded? Trouble accessing the params passed to the optim at instantiation.
            if optimizer.get_name() == 'Momentum':
                optimizer_info = optimizer.get_name() + '{momentum=%.2f}' % optimizer._momentum
            else:
                optimizer_info = optimizer.get_name() + '{%s}' % (optimizer.get_slot_names())
                # optimizer_info = {slot_name: slot_value for slot_name, slot_value in zip(optimizer.get_slot_names(), optimizer.'_'.join(...)}
                # optimizer_info = optimizer.get_name() + '{%s}' % optimizer.variables()
                # optimizer_info = optimizer.get_name() + '{%s=%.2f}' % (optimizer.get_slot_names()[0], optimizer._momentum)
            train_step = optimizer.minimize(cross_entropy_mean)

    return train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor, optimizer_info


def add_jpeg_decoding(module_spec):
    """Adds operations that perform JPEG decoding and resizing to the graph..

    Args:
      module_spec: The hub.ModuleSpec for the image module being used.

    Returns:
      Tensors for the node to feed JPEG data into, and the output of the
        preprocessing steps.
    """
    input_height, input_width = hub.get_expected_image_size(module_spec)
    input_depth = hub.get_num_image_channels(module_spec)
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    # Convert from full range of uint8 to range [0,1] of float32.
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                          tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                             resize_shape_as_int)
    return jpeg_data, resized_image


def get_image_path(image_lists, label_name, index, image_dir, category):
    """Returns a path to an image for a label at the given index.

    Args:
      image_lists: OrderedDict of training images for each label.
      label_name: Label string we want to get an image for.
      index: Int offset of the image we want. This will be moduloed by the
      available number of images for the label, so it can be arbitrarily large.
      image_dir: Root folder string of the subfolders containing the training
      images.
      category: Name string of set to pull images from - training, testing, or
      validation.

    Returns:
      File system path string to an image that meets the requested parameters.

    """
    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.',
                         label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category, module_name):
    """
    Returns a path to a bottleneck file for a label at the given index.

    Args:
      image_lists: OrderedDict of training images for each label.
      label_name: Label string we want to get an image for.
      index: Integer offset of the image we want. This will be moduloed by the
      available number of images for the label, so it can be arbitrarily large.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      category: Name string of set to pull images from - training, testing, or
      validation.
      module_name: The name of the image module being used.

    Returns:
      File system path string to an image that meets the requested parameters.
    """
    module_name = (module_name.replace('://', '~')  # URL scheme.
                   .replace('/', '~')  # URL and Unix paths.
                   .replace(':', '~').replace('\\', '~'))  # Windows paths.
    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.',
                         label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = bottleneck_dir
    img_name =  base_name.split(sep='\\')[-1]
    module_name = img_name + '_' + module_name + '.txt'
    full_path = os.path.join(sub_dir, label_name, module_name)
    return full_path


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
    """Runs inference on an image to extract the 'bottleneck' summary layer.

    Args:
      sess: Current active TensorFlow Session.
      image_data: String of raw JPEG data.
      image_data_tensor: Input data layer in the graph.
      decoded_image_tensor: Output of initial image resizing and preprocessing.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: Layer before the final softmax.

    Returns:
      Numpy array of bottleneck values.
    """
    # First decode the JPEG image, resize it, and rescale the pixel values.
    resized_input_values = sess.run(decoded_image_tensor,
                                    {image_data_tensor: image_data})
    # Then run it through the recognition network.
    bottleneck_values = sess.run(bottleneck_tensor,
                                 {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor):
    """Create a single bottleneck file."""
    tf.logging.info('Creating bottleneck at ' + bottleneck_path)
    image_path = get_image_path(image_lists, label_name, index,
                                image_dir, category)
    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                     str(e)))
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             decoded_image_tensor, resized_input_tensor,
                             bottleneck_tensor, module_name):
    """Retrieves or calculates bottleneck values for an image.

    If a cached version of the bottleneck data exists on-disk, return that,
    otherwise calculate the data and save it to disk for future use.

    Args:
      sess: The current active TensorFlow Session.
      image_lists: OrderedDict of training images for each label.
      label_name: Label string we want to get an image for.
      index: Integer offset of the image we want. This will be modulo-ed by the
      available number of images for the label, so it can be arbitrarily large.
      image_dir: Root folder string of the subfolders containing the training
      images.
      category: Name string of which set to pull images from - training, testing,
      or validation.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      jpeg_data_tensor: The tensor to feed loaded jpeg data into.
      decoded_image_tensor: The output of decoding and resizing the image.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: The output tensor for the bottleneck values.
      module_name: The name of the image module being used.

    Returns:
      Numpy array of values produced by the bottleneck layer for the image.
    """
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                          bottleneck_dir, category, module_name)
    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor,
                               bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        tf.logging.warning('Invalid float found, recreating bottleneck')
        did_hit_error = True
    if did_hit_error:
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor,
                               bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        # Allow exceptions to propagate here, since they shouldn't happen after a
        # fresh creation
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor, module_name):
    """Ensures all the training, testing, and validation bottlenecks are cached.

    Because we're likely to read the same image multiple times (if there are no
    distortions applied during training) it can speed things up a lot if we
    calculate the bottleneck layer values once for each image during
    preprocessing, and then just read those cached values repeatedly during
    training. Here we go through all the images we've found, calculate those
    values, and save them off.

    Args:
      sess: The current active TensorFlow Session.
      image_lists: OrderedDict of training images for each label.
      image_dir: Root folder string of the subfolders containing the training
      images.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      jpeg_data_tensor: Input tensor for jpeg data from file.
      decoded_image_tensor: The output of decoding and resizing the image.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: The penultimate output layer of the graph.
      module_name: The name of the image module being used.

    Returns:
      Nothing.
    """
    # tf.logging.log(tf.logging.INFO, msg='cache_bottlenecks called with image_lists: %s' % image_lists)
    # tf.logging.log_first_n(level=tf.logging.INFO, msg='image_lists key\'s: %s' % image_lists.keys(), n=1)
    # tf.logging.log(tf.logging.INFO, msg='\'training\' in image_list\'s keys?: %s' % ('training' in image_lists.keys()))
    how_many_bottlenecks = 0
    if not os.path.exists(bottleneck_dir):
        os.makedirs(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        # TODO: Enable early stopping.
        # for category in ['training', 'testing']:
        for category in ['train', 'val']:
            category_list = label_lists[category]
            # try:
            #     category_list = label_lists[category]
            # except KeyError as key_err:
            #     tf.logging.log(level=tf.logging.INFO, msg='KeyError for label \'%s\': %s' % (label_name, key_err))
            # tf.logging.log_first_n(level=tf.logging.INFO, msg='label_lists[%s]: %s' % (category, label_lists[category]), n=4)
            for index, unused_base_name in enumerate(category_list):
                # TODO: Break this into two functions so two counters can be maintained (one for load one for init.):
                get_or_create_bottleneck(
                    sess, image_lists, label_name, index, image_dir, category,
                    bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                    resized_input_tensor, bottleneck_tensor, module_name)

                how_many_bottlenecks += 1
                if how_many_bottlenecks % 100 == 0:
                    tf.logging.info(
                        str(how_many_bottlenecks) + ' bottleneck files loaded (or created).')


def add_evaluation_step(result_tensor, ground_truth_tensor):
    """Inserts the operations needed to evaluate the accuracy of the results.

      Args:
        result_tensor: The new final node that produces results.
        ground_truth_tensor: The node we feed ground truth data
        into.

      Returns:
        Tuple of (evaluation step, prediction).
    """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            tf.logging.info(msg='result_tensor: %s' % result_tensor)
            # tf.logging.info(msg='result_tensor_shape: %s' % result_tensor.shape)
            tf.logging.info(msg='ground_truth_tensor: %s' % ground_truth_tensor)
            prediction = tf.argmax(result_tensor, 1)
            tf.logging.info(msg='prediction tensor: %s' % prediction)
            # Returns the truth value of (prediction == ground_truth_tensor) element-wise.
            correct_prediction = tf.equal(prediction, ground_truth_tensor)
            tf.logging.info(msg='correct_prediction: %s' % correct_prediction)
        with tf.name_scope('accuracy'):
            # Compute the mean of the elements along the given axis:
            acc_evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # Now compute the top-k accuracy:
        with tf.name_scope('top5_accuracy'):
            top5_acc_eval_step = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=result_tensor, targets=ground_truth_tensor, k=5), tf.float32))
            # top5_acc_eval_step = tf.constant(top5_acc_eval_step)
            # top_k_probs, top_k_indices = tf.nn.top_k(input=result_tensor, k=5)
            # top_k_probs is a [128, 5] matrix. Each row is a sample. Each column is one of the probabilities.
            # top_k_indices is a [128, 5] matrix. Each row is a sample. Each column is the col index containing the top prob.
            # top_k_predictions is equivalent to top_k_indices because the index corresponds to encoded class label
            # top_k_class_labels = tf.gather(params=ground_truth_tensor, indices=top_k_indices)
            # top_k_class_labels = top_k_probs[:, 0]
            # tf.logging.info(msg='top_k_class_labels: %s' % top_k_class_labels)
            # correct_predictions = tf.equal(tf.cast(top_k_class_labels, tf.int64), ground_truth_tensor)
            # tf.logging.info(msg='correct_predictions: %s' % correct_predictions)
            # top_5_acc_eval_step = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            # tf.logging.info(msg='top_5_acc_eval_step: %s' % top_5_acc_eval_step)


            # top_5_probabilities, top_5_predicted_labels = tf.nn.top_k(input=prediction, k=5)
            # # top_5_acc_eval_step = tf.reduce_mean(tf.nn.top_k(input=tf.cast(result_tensor, tf.float32), k=5))
            # k_largest_probs, k_largest_indices = tf.nn.top_k(result_tensor, k=5)
            # tf.logging.info('k_largest_probs: %s' % k_largest_probs)
            # tf.logging.info('k_largest_indices: %s' % k_largest_indices)
            # # truth value of (k_largest_pred == ground_truth_tensor) element wise:
            # correct_predictions = tf.equal(k_largest_probs, tf.cast(ground_truth_tensor, tf.float32))
            # top_5_acc_eval_step = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), axis=1)
            # top_5_acc_eval_step = tf.reduce_mean(tf.cast(tf.nn.top_k(input=result_tensor, k=5), dtype=tf.float32), axis=1)

            # tf.metrics.mean(tf.nn.in_top_k(predictions=prediction, targets=ground_truth_tensor, k=5))
            # tf.logging.info(msg='top_5_probabilities: %s, top_5_predicted_labels: %s' % (top_5_probabilities, top_5_predicted_labels))
            # correct_predictions = tf.equal(top_5_predicted_labels, tf.nn.top_k(input=ground_truth_tensor, k=5))
            # tf.logging.info(msg='predictions: %s' % predictions)
            # top_5_acc_eval_step = tf.get_variable(name="top_5_acc_eval", shape=(1,), trainable=False)
            # top_5_acc_eval_step = tf.metrics.mean(tf.nn.in_top_k(predictions=tf.cast(result_tensor, tf.float32), targets=ground_truth_tensor, k=5, name='in_top_5'), name='top5_accuracy')
            # top_5_acc_eval_step = tf.nn.top_k(predictions=tf.cast(prediction, tf.float32), targets=ground_truth_tensor, k=5)
            # top_5_acc_eval_step = None

    # Export the accuracy of the model for use in tensorboard:
    tf.summary.scalar('accuracy', acc_evaluation_step)
    tf.summary.scalar('top5_accuracy', top5_acc_eval_step)

    return acc_evaluation_step, top5_acc_eval_step, prediction


def build_eval_session(module_spec, class_count):
    """Builds an restored eval session without train operations for exporting.

    Args:
      module_spec: The hub.ModuleSpec for the image module being used.
      class_count: Number of classes

    Returns:
      Eval session containing the restored eval graph.
      The bottleneck input, ground truth, eval step, and prediction tensors.
    """
    # If quantized, we need to create the correct eval graph for exporting.
    eval_graph, bottleneck_tensor, resized_input_tensor, wants_quantization = (
        create_module_graph(module_spec))

    eval_sess = tf.Session(graph=eval_graph)
    with eval_graph.as_default():
        # Add the new layer for exporting.
        (_, _, bottleneck_input,
         ground_truth_input, final_tensor, optimizer_info) = add_final_retrain_ops(
            class_count, CMD_ARG_FLAGS.final_tensor_name, bottleneck_tensor,
            wants_quantization, is_training=False)

        # Now we need to restore the values from the training graph to the eval
        # graph.
        tf.train.Saver().restore(eval_sess, CHECKPOINT_DIR)

        acc_evaluation_step, top5_acc_eval_step, prediction = add_evaluation_step(final_tensor,
                                                          ground_truth_input)

    return (eval_sess, resized_input_tensor, bottleneck_input, ground_truth_input,
            acc_evaluation_step, top5_acc_eval_step, prediction)


def save_graph_to_file(graph, graph_file_name, module_spec, class_count):
    """
    save_graph_to_file: Saves a tensorflow graph to file, creating a valid quantized one if necessary.
    :param graph:
    :param graph_file_name:
    :param module_spec:
    :param class_count:
    :return:
    """
    sess, _, _, _, _, _, _ = build_eval_session(module_spec, class_count)
    graph = sess.graph

    tf.train.write_graph(sess.graph_def, logdir='tmp/summaries', name='session_graph_def', as_text=True)

    # TODO: Currently hardcoding the namescope because it will be a pain to get it otherwise. Come back and fix this.
    # graph_name_scope = graph.get_name_scope()

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), ['re-train_ops/' + CMD_ARG_FLAGS.final_tensor_name])

    with tf.gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())


def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  decoded_image_tensor, resized_input_tensor,
                                  bottleneck_tensor, module_name):
    """Retrieves bottleneck values for cached images.

    If no distortions are being applied, this function can retrieve the cached
    bottleneck values directly from disk for images. It picks a random set of
    images from the specified category.

    Args:
      sess: Current TensorFlow Session.
      image_lists: OrderedDict of training images for each label.
      how_many: If positive, a random sample of this size will be chosen.
      If negative, all bottlenecks will be retrieved.
      category: Name string of which set to pull from - training, testing, or
      validation.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      image_dir: Root folder string of the subfolders containing the training
      images.
      jpeg_data_tensor: The layer to feed jpeg image data into.
      decoded_image_tensor: The output of decoding and resizing the image.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: The bottleneck output layer of the CNN graph.
      module_name: The name of the image module being used.

    Returns:
      List of bottleneck arrays, their corresponding ground truths, and the
      relevant filenames.
    """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    if how_many >= 0:
        # Retrieve a random sample of bottlenecks.
        for unused_i in range(how_many):

            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_name = get_image_path(image_lists, label_name, image_index,
                                        image_dir, category)
            bottleneck = get_or_create_bottleneck(
                sess, image_lists, label_name, image_index, image_dir, category,
                bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                resized_input_tensor, bottleneck_tensor, module_name)
            bottlenecks.append(bottleneck)
            ground_truths.append(label_index)
            filenames.append(image_name)
    else:
        # Retrieve all bottlenecks.
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(
                    image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name, image_index,
                                            image_dir, category)
                bottleneck = get_or_create_bottleneck(
                    sess, image_lists, label_name, image_index, image_dir, category,
                    bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                    resized_input_tensor, bottleneck_tensor, module_name)
                bottlenecks.append(bottleneck)
                ground_truths.append(label_index)
                filenames.append(image_name)
    return bottlenecks, ground_truths, filenames


def main(_):
    # Enable visible logging output:
    tf.logging.set_verbosity(tf.logging.INFO)

    if not CMD_ARG_FLAGS.image_dir:
        # The directory housing the training images was not specified.
        tf.logging.error('The flag --image_dir must be set.')
        return -1

    # Delete any TensorBoard summaries left over from previous runs:
    prepare_tensor_board_directories()
    tf.logging.info(msg='Removed left over tensorboard summaries from previous runs.')

    # Partition images into train, test, validate sets:
    tf.logging.info(msg='Partitioning images into training, validation, testing sets. Using: proportionate allocation stratified shuffled sampling...')
    ts = time.time()
    image_lists = partition_into_image_lists(image_dir=CMD_ARG_FLAGS.image_dir, train_percent=.8, test_percent=.2, val_percent=.2, random_state=0)
    tf.logging.info(msg='Populated image lists, performed partitioning in: %s seconds (%.2f minutes).' % ((time.time() - ts), (time.time() - ts)/60))

    # This is operating under the assumption we have the same number of classes in the training and testing sets:
    class_count = len(image_lists.keys())
    tf.logging.info(msg='Detected %d unique classes.' % class_count)

    # TODO: See if the command-line flags specify any distortions
    # do_distort_images = should_distort_images(
    #   FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
    #   FLAGS.random_brightness)

    # If no restoration from checkpoint was requested, then use the provided tfhub module:
    if not CMD_ARG_FLAGS.resume_final_checkpoint_path:
        # Set up the pre-trained graph:
        module_spec = hub.load_module_spec(CMD_ARG_FLAGS.tfhub_module)
        tf.logging.info(msg='Loaded tensorflow hub module spec: %s' % CMD_ARG_FLAGS.tfhub_module)

        graph, bottleneck_tensor, resized_image_tensor, wants_quantization = (
            create_module_graph(module_spec))
        tf.logging.info(msg='Defined computational graph from the tensorflow hub module spec.')

        # Add the new layer that we'll be training to our new default graph:
        with graph.as_default():
            (train_step, cross_entropy, bottleneck_input,
             ground_truth_input, final_tensor, optimizer_info) = add_final_retrain_ops(
                class_count, CMD_ARG_FLAGS.final_tensor_name, bottleneck_tensor,
                wants_quantization, is_training=True)
        tf.logging.info(msg='Added final retrain ops to the module source graph.')
    else:
        # Restore from checkpoint of a previously trained model:
        tf.logging.info(msg='Attempting to restore from a previously saved checkpoint. '
                            'Expecting previous model to have re-train ops already added to computational graph!')
        graph = tf.Graph()

        # graph, bottleneck_tensor, resized_image_tensor, wants_quantization = (
        #     restore_module_graph(module_checkpoint)
        # )

    ''' Training Loop: '''
    with graph.as_default():
        with tf.Session(graph=graph) as sess:
            # Initialize all weights: for the module to their pretrained values,
            # and for the newly added retraining layer to random initial values.
            # init = tf.global_variables_initializer()
            # init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            if CMD_ARG_FLAGS.resume_final_checkpoint_path:
                # When you restore variables you do not have to initialize them beforehand:
                tf.saved_model.loader.load(sess=sess, tags=[tag_constants.TRAINING], export_dir=CMD_ARG_FLAGS.resume_final_checkpoint_path)
                tf.logging.info(msg='Restored model from saved checkpoint (.pb) file.')

            # sess.run(init)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            with tf.name_scope('re-train_ops'):
                # Set up the image decoding sub-graph.
                jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(module_spec)

            # TODO: Add data augmentation pipeline:
            # if do_distort_images:
            #     # We will be applying distortions, so setup the operations we'll need.
            #     (distorted_jpeg_data_tensor, distorted_image_tensor) = add_input_distortions(
            #         FLAGS.flip_left_right,
            #         FLAGS.random_crop,
            #         FLAGS.random_scale,
            #         FLAGS.random_brightness,
            #         module_spec
            #     )
            # else:
            ''' Ensure that the bottleneck image summaries are calculated and cached on disk'''
            cache_bottlenecks(sess, image_lists, CMD_ARG_FLAGS.image_dir,
                              CMD_ARG_FLAGS.bottleneck_dir, jpeg_data_tensor,
                              decoded_image_tensor, resized_image_tensor,
                              bottleneck_tensor, CMD_ARG_FLAGS.tfhub_module)

            # Create operations to evaluate the accuracy of the new layer (called during validation during training):
            acc_evaluation_step, top5_acc_eval_step, _ = add_evaluation_step(final_tensor, ground_truth_input)

            # Merge all summaries and write them out to the summaries_dir
            merged = tf.summary.merge_all()
            # TODO: This might not work for other models unless the urls are formatted with the same array:
            hyper_string = '%s/MBT_%s,MBV_%s,lr_%.1E,opt_%s' \
                           % (CMD_ARG_FLAGS.tfhub_module.split('/')[-3], CMD_ARG_FLAGS.train_batch_size,
                              CMD_ARG_FLAGS.val_batch_size, CMD_ARG_FLAGS.learning_rate, optimizer_info)
            train_writer = tf.summary.FileWriter(CMD_ARG_FLAGS.summaries_dir + '/train/' + hyper_string, sess.graph)
            val_writer = tf.summary.FileWriter(CMD_ARG_FLAGS.summaries_dir + '/val/' + hyper_string)
            # Create a train saver that is used to restore values into an eval graph:
            train_saver = tf.train.Saver()

            # run training for as many cycles as requested on the command line:
            for i in range(CMD_ARG_FLAGS.num_epochs):
                # Get a batch of input bottleneck values, either calculated fresh every
                # time with distortions applied, or from the cache stored on disk.
                (train_bottlenecks, train_ground_truth, _) = get_random_cached_bottlenecks(
                    sess, image_lists, CMD_ARG_FLAGS.train_batch_size, 'train',
                    CMD_ARG_FLAGS.bottleneck_dir, CMD_ARG_FLAGS.image_dir, jpeg_data_tensor,
                    decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                    CMD_ARG_FLAGS.tfhub_module)

                # Feed the bottlenecks and ground truth into the graph, and run a training
                # step. Capture training summaries for TensorBoard with the `merged` op.
                train_summary, _ = sess.run([merged, train_step],
                                            feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
                train_writer.add_summary(train_summary, i)

                # Every so often, print out how well the graph is training.
                is_last_step = (i + 1 == CMD_ARG_FLAGS.num_epochs)
                if (i % CMD_ARG_FLAGS.eval_step_interval) == 0 or is_last_step:
                    # TODO: Error has something to do with this, keep an eye on cross_entropy_value != predictions
                    train_accuracy, top5_accuracy, cross_entropy_value = sess.run(
                        [acc_evaluation_step, top5_acc_eval_step, cross_entropy],
                        feed_dict={bottleneck_input: train_bottlenecks,
                                   ground_truth_input: train_ground_truth}
                    )
                    tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i, train_accuracy * 100))
                    tf.logging.info('%s: Step %d: Cross entropy = %f' % (datetime.now(), i, cross_entropy_value))
                    # TODO: Make this use an eval graph, to avoid quantization
                    # moving averages being updated by the validation set, though in
                    # practice this makes a negligable difference.
                    validation_bottlenecks, validation_ground_truth, _ = (
                        get_random_cached_bottlenecks(
                                sess=sess, image_lists=image_lists, how_many=CMD_ARG_FLAGS.val_batch_size,
                                category='val', bottleneck_dir=CMD_ARG_FLAGS.bottleneck_dir,
                                image_dir=CMD_ARG_FLAGS.image_dir, jpeg_data_tensor=jpeg_data_tensor,
                                decoded_image_tensor=decoded_image_tensor, resized_input_tensor=resized_image_tensor,
                                bottleneck_tensor=bottleneck_tensor, module_name=CMD_ARG_FLAGS.tfhub_module))
                    # Run a validation step and capture training summaries for TensorBoard with the 'merged' op:
                    validation_summary, validation_accuracy, top5_val_accuracy = sess.run(
                        [merged, acc_evaluation_step, top5_acc_eval_step],
                        feed_dict={bottleneck_input: validation_bottlenecks,
                                   ground_truth_input: validation_ground_truth})
                    val_writer.add_summary(validation_summary, i)
                    tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                                    (datetime.now(), i, validation_accuracy * 100, len(validation_bottlenecks)))
                    tf.logging.info(msg='%s: Step %d: Validation top-5 accuracy = %.1f%% (N=%d)' %
                                        (datetime.now(), i, top5_val_accuracy * 100, len(validation_bottlenecks)))

                # Store intermediate results
                intermediate_frequency = CMD_ARG_FLAGS.intermediate_store_frequency

                if (intermediate_frequency > 0 and (i % intermediate_frequency == 0) and i > 0):
                    # requested print frequency is greater than zero. This isn't the first epoch, it is time to print:
                    # If we do an intermediate save, we must save a checkpoint of the train graph to restore onto the eval
                    #   graph.
                    train_saver.save(sess, CHECKPOINT_DIR)
                    intermediate_file_name = (CMD_ARG_FLAGS.intermediate_output_graphs_dir +
                                              'intermediate_' + str(i) + '.pb')
                    tf.logging.info('Save intermediate result to : ' +
                                    intermediate_file_name)

                    save_graph_to_file(graph, intermediate_file_name, module_spec,
                                       class_count)

            # After training is complete, force one last save of the train checkpoint.
            train_saver.save(sess, CHECKPOINT_DIR)

            # TODO: Add code to run final test evaluation


            # Write out trained graph and labels with weights stored as constants:
            tf.logging.info(msg='Save final result to : ' + CMD_ARG_FLAGS.output_graph)
            if wants_quantization:
                tf.logging.info('The model is instrumented for quantization with TF-Lite')
            save_graph_to_file(graph, CMD_ARG_FLAGS.output_graph, module_spec, class_count)
            # TODO: Export saved model:
            # with tf.gfile.FastGFile(CMD_ARG_FLAGS.output_labels, 'w') as f:
            #     f.write('\n'.join(image_lists.keys())+ '\n')
            #
            # if CMD_ARG_FLAGS.saved_model_dir:
            #     export_model(module_spec, class_count, CMD_ARG_FLAGS.saved_model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TensorFlow Transfer Learning Demo on Going Deeper Herbaria 1K Dataset')
    parser.add_argument(
        '--resume_final_checkpoint_path',
        type=str,
        default='',
        help='The path to a (.pb file) housing a previously trained graph '
             '(i.e. has a re-trained final layer already saved).'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=128,
        help='The number of images per mini-batch during training.'
    )
    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=128,
        help='The number of images per mini-batch during validation.'
    )
    parser.add_argument(
        '--tfhub_module',
        type=str,
        default=(
            'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'),
            help="""\
            Which TensorFlow Hub module to use.
            See https://github.com/tensorflow/hub/blob/r0.1/docs/modules/image.md
            for some publicly available ones.\
            """
    )
    parser.add_argument(
        '--final_tensor_name',
        type=str,
        default='final_result',
        help="""\
            The name of the output classification layer in the retrained graph.\
            """
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='How large a learning rate to use when training.'
    )
    parser.add_argument(
        '--bottleneck_dir',
        type=str,
        default='/tmp/bottlenecks',
        help='Path to cache bottleneck layer values as files.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='Number of training epochs (passes over the entire training dataset).'
    )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='tmp/summaries',
        help='Directory to which TensorBoard summaries will be saved.'
    )
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default='10',
        help='Specifies the number of epochs during training before performing an evaluation step.'
    )
    parser.add_argument(
        '--output_graph',
        type=str,
        default='tmp/saved_model.pb',
        help='Directory to save the trained graph.'
    )
    parser.add_argument(
        '--intermediate_store_frequency',
        type=int,
        default=0,
        help='How many epochs to run before storing an intermediate checkpoint of the graph. If 0, then will not store.'
    )
    parser.add_argument(
        '--intermediate_output_graphs_dir',
        type=str,
        default='tmp/intermediate_graph/',
        help='Directory to save the intermediate graphs.'
    )
    # Parse command line args and identify unknown flags:
    CMD_ARG_FLAGS, unparsed = parser.parse_known_args()
    '''
    Execute this script under a shell instead of importing as a module. Ensures that the main function is called with
    the proper command line arguments (builds on default argparse). For more information see:
    https://stackoverflow.com/questions/33703624/how-does-tf-app-run-work
    '''
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
