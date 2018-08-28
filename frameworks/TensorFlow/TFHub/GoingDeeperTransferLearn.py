import argparse
import sys
import tensorflow as tf
import tensorflow_hub as hub
import collections
import os
import re
import hashlib
import numpy as np

CMD_ARG_FLAGS = None
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
# A module is understood as instrumented for quantization with TF-Lite
# if it contains any of these ops.
FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars',
                  'FakeQuantWithMinMaxVarsPerChannel')


def prepare_tensor_board_directories():
    """
    prepare_tensor_board_directories: Ensures that if a TensorBoard storage directory is defined in the command line
        flags, that said directory is purged of old TensorBoard files, and that this program has sufficient permissions
        to write new TensorBoard summaries to the specified path.
    :return:
    """
    # TODO: Method body.
    return NotImplementedError


def create_image_lists(image_dir, training_image_dir=None, testing_image_dir=None, testing_percentage=80, validation_percentage=20):
    """
    create_image_lists: Creates a dictionary containing all available datasets (train, test, validate) as a list of
        file paths.
    :param image_dir: The root directory for all images.
    :param training_image_dir: The root directory for all training images (if present).
    :param testing_image_dir: The root directory for all testing images (if present).
    :param testing_percentage:
    :param validation_percentage:
    :return image_lists: A dictionary containing all available datasets (train, test, validate) as a list of file paths.
    """

    '''
    Check to see if the root directory exists. We use tf.gfile which is a C++ FileSystem API wrapper for the Python
        file API that also supports Google Cloud Storage and HDFS. For more information see:
        https://stackoverflow.com/questions/42256938/what-does-tf-gfile-do-in-tensorflow
    '''
    if not tf.gfile.Exists(image_dir):
        tf.logging.error("Root image directory '" + image_dir + "' not found.")
        return None

    image_lists = collections.OrderedDict()

    if training_image_dir is not None:
        ''' There is a separate parent folder for training images. '''
        # Run a sanity check on the folder:
        if not tf.gfile.Exists(training_image_dir):
            tf.logging.error("You specified the existence of training image directory '"
                             + training_image_dir + "' which was not found.")
            return None
        training_sub_dirs = sorted(x[0] for x in tf.gfile.Walk(training_image_dir))
        # The root directory comes first, so skip it.
        is_train_root_dir = True
        # Acceptable file extensions:
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

        for train_sub_dir in training_sub_dirs:
            if is_train_root_dir:
                is_train_root_dir = False
                # Skip the root_dir:
                continue
            train_file_list = []
            dir_name = os.path.basename(train_sub_dir)
            if dir_name == training_image_dir:
                # Return control to beginning of for-loop:
                continue
            tf.logging.info("Looking for training images in '" + dir_name + "'")
            for extension in extensions:
                # Get a list of all accepted file extensions and the targeted file_name:
                file_glob = os.path.join(training_image_dir, dir_name, '*.' + extension)
                # Append all items from the file_glob to the list of files (if extension exists):
                train_file_list.extend(tf.gfile.Glob(file_glob))
            if not train_file_list:
                tf.logging.warning("'No files found in '" + dir_name)
                # Return control to beginning of for-loop:
                continue
            if len(train_file_list) < 20:
                tf.logging.warning('WARNING: Folder has less than 20 images, which may cause issues. See: %s for info.'
                                   % ('https://stackoverflow.com/questions/38175673/critical-tensorflowcategory-has-no-images-validation'))
            elif len(train_file_list) > MAX_NUM_IMAGES_PER_CLASS:
                tf.logging.warning(
                    'WARNING: Folder {} has more than {} images. Some images will '
                    'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
            train_label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
            training_images = []
            for train_file_name in train_file_list:
                base_name = os.path.basename(train_file_name)
                training_images.append(base_name)
            image_lists[train_label_name] = {
                'dir': dir_name,
                'training': training_images
            }

        if testing_image_dir is not None:
            ''' There is a separate parent folder for both training and testing images. '''
            # Run a sanity check on the folder:
            if not tf.gfile.Exists(testing_image_dir):
                tf.logging.error("You specified the existence of testing image directory '"
                                 + testing_image_dir + "' which was not found.")
                return None
            testing_sub_dirs = sorted(x[0] for x in tf.gfile.Walk(testing_image_dir))
            # The root directory comes first, so skip it.
            is_test_root_dir = True
            for test_sub_dir in testing_sub_dirs:
                if is_test_root_dir:
                    is_test_root_dir = False
                    # Skip the root_dir:
                    continue
                test_file_list = []
                dir_name = os.path.basename(test_sub_dir)
                if dir_name == testing_image_dir:
                    # Return control to beginning of for-loop:
                    continue
                tf.logging.info("Looking for testing images in '" + dir_name + "'")
                for extension in extensions:
                    # Get a list of all accepted file extensions and the targeted file_name:
                    file_glob = os.path.join(testing_image_dir, dir_name, '*.' + extension)
                    # Append all items form the file_glob to the list of files:
                    test_file_list.extend(tf.gfile.Glob(file_glob))
                if not test_file_list:
                    tf.logging.warning("'No files found in '" + dir_name)
                    # Return control to beginning of for-loop:
                    continue
                if len(test_file_list) > MAX_NUM_IMAGES_PER_CLASS:
                    tf.logging.warning(
                        'WARNING: Folder {} has more than {} images. Some images will '
                        'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
                test_label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
                testing_images = []
                for test_file_name in test_file_list:
                    base_name = os.path.basename(test_file_name)
                    testing_images.append(base_name)
                if image_lists[test_label_name]:
                    image_lists[test_label_name]['dir'] = dir_name
                    image_lists[test_label_name]['testing'] = testing_images
                else:
                    image_lists[test_label_name] = {
                        'dir': dir_name,
                        'testing': testing_images
                    }
            return image_lists
        else:
            ''' There is a separate parent folder for training images, but not testing images. '''
            tf.logging.ERROR("You specified a separate folder for training images but not testing images. This logic has"
                             "not been implemented yet.")
            return NotImplementedError
    else:
        ''' There is not a separate parent folder for training images. Assumed the same is true for testing images.'''
        tf.logging.error("Both training and testing images are located in the same folder. This logic is not currently "
                         "implemented, due to the usecase of partitioning within the class label 'species'.")
        sub_dirs = sorted(x[0] for x in tf.gfile.Walk(image_dir))
        # TODO: Understand the following (see: https://stackoverflow.com/questions/51922602/what-exactly-does-the-data-set-creator-do-to-group-photos-in-tensorflow)
        '''
        We want to ignore anything after '_nohash_' in the file name when
        deciding which set to put an image in, the data set creator has a way of
        grouping photos that are close variations of each other. For example
        this is used in the plant disease data set to group multiple pictures of
        the same leaf.
        '''
        # hash_name = re.sub(r'_nohash_.*$', '', train_file_name)
        '''
        This looks a bit magical, but we need to decide whether this file should
        go into the training, testing, or validation sets, and we want to keep
        existing files in the same set even if more files are subsequently
        added.
        To do that, we need a stable way of deciding based on just the file name
        itself, so we do a hash of that and then use that to generate a
        probability value that we use to assign it.
        '''
        # hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest()
        # percentage_hash = ((int(hash_name_hashed, 16) %
        #                     (MAX_NUM_IMAGES_PER_CLASS + 1)) *
        #                    (100.0 / MAX_NUM_IMAGES_PER_CLASS))
        return NotImplementedError


def create_module_graph(module_spec):
    """Creates a graph and loads Hub Module into it.

    source: https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py

    Args:
        module_spec: the hub.ModuleSpec for the image module being used.

    Returns:
        graph: the tf.Graph that was created.
        bottleneck_tensor: the bottleneck values output by the module.
        resized_input_tensor: the input images, resized as expected by the module.
        wants_quantization: a boolean, whether the module has been instrumented
          with fake quantization ops.
    """
    # Define the receptive field in accordance with the chosen architecture:
    height, width = hub.get_expected_image_size(module_spec)
    with tf.Graph().as_default() as graph:
        # Create a placeholder tensor for input to the model.
        resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
        # Declare the model in accordance with the chosen architecture:
        m = hub.Module(module_spec)
        # Create another place holder tensor to catch the output of the pre-activation layer:
        bottleneck_tensor = m(resized_input_tensor)
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
    add_final_retrain_ops: Adds a new softmax and fully-connected layer for training and eval.

    We need to retrain the top layer to identify our new classes, so this function
    adds the right operations to the graph, along with some variables to hold the
    weights, and then sets up all the gradients for the backward pass.

    The set up for the softmax and fully-connected layers is based on:
    https://www.tensorflow.org/tutorials/mnist/beginners/index.html

    :source url: https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py
    :modified str: Chris Campell
    :param class_count: Integer of how many categories of things we're trying to
        recognize.
    :param final_tensor_name: Name string for the new final node that produces results.
    :param bottleneck_tensor: The output of the main CNN graph.
    :param quantize_layer: Boolean, specifying whether the newly added layer should be
        instrumented for quantization with TF-Lite.
    :param is_training: Boolean, specifying whether the newly add layer is for training
        or eval.
    :returns : The tensors for the training and cross entropy results, and tensors for the
    bottleneck input and ground truth input.
    """
    # The batch size
    batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
    assert batch_size is None, 'We want to work with arbitrary batch size when ' \
                               'constructing fully-connected and softmax layers for fine-tuning.'

    # Tensor declarations:
    with tf.name_scope('input'):
        # Create a placeholder Tensor (assignment at runtime) of the same type as bottleneck_tensor:
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
            variable_summaries(layer_weights)

        with tf.name_scope('biases'):
            layer_biases = tf.Variable(initial_value=tf.zeros([class_count]), name='final_biases')
            variable_summaries(layer_biases)

        # pre-activations:
        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)

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

    tf.summary.histogram('activiations', final_tensor)

    # If this is an eval graph, we don't need to add loss ops or an optimizer.
    if not is_training:
        return None, None, bottleneck_input, ground_truth_input, final_tensor

    with tf.name_scope('cross_entropy'):
        # What constitutes sparse in this case:
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(labels=ground_truth_input, logits=logits)

    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(CMD_ARG_FLAGS.learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor)


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
    return get_image_path(image_lists, label_name, index, bottleneck_dir,
                          category) + '_' + module_name + '.txt'


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
    how_many_bottlenecks = 0
    if not os.path.exists(bottleneck_dir):
        os.makedirs(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        # for category in ['training', 'testing']:
        for category in ['training']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_bottleneck(
                    sess, image_lists, label_name, index, image_dir, category,
                    bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                    resized_input_tensor, bottleneck_tensor, module_name)

                how_many_bottlenecks += 1
                if how_many_bottlenecks % 100 == 0:
                    tf.logging.info(
                        str(how_many_bottlenecks) + ' bottleneck files created.')


def add_evaluation_step(result_tensor, ground_truth_tensor):
    """Inserts the operations we need to evaluate the accuracy of our results.

      Args:
        result_tensor: The new final node that produces results.
        ground_truth_tensor: The node we feed ground truth data
        into.

      Returns:
        Tuple of (evaluation step, prediction).
    """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            # Returns the truth value of (prediction == ground_truth_tensor) element-wise.
            correct_prediction = tf.equal(prediction, ground_truth_tensor)
        with tf.name_scope('accuracy'):
            # Compute the mean of the elements along the given axis:
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Export the accuracy of the model for use in tensorboard:
    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction


def main(_):
    # Enable visible logging output:
    tf.logging.set_verbosity(tf.logging.INFO)

    if not CMD_ARG_FLAGS.image_dir:
        # The directory housing the training images was not specified.
        tf.logging.error('The flag --image_dir must be set.')
        return -1

    # TODO: Create directories for TensorBoard summaries:
    # prepare_tensor_board_directories()

    # Create lists of all the images:
    image_lists = create_image_lists(
        image_dir=CMD_ARG_FLAGS.image_dir,
        training_image_dir=CMD_ARG_FLAGS.train_image_dir,
        testing_image_dir=CMD_ARG_FLAGS.test_image_dir
    )
    # TODO: This needs some work...
    class_count = len(image_lists.keys())

    # TODO: See if the command-line flags specify any distortions
    # do_distort_images = should_distort_images(
    #   FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
    #   FLAGS.random_brightness)

    # Set up the pre-trained graph:
    module_spec = hub.load_module_spec(CMD_ARG_FLAGS.tfhub_module)
    graph, bottleneck_tensor, resized_image_tensor, wants_quantization = (
        create_module_graph(module_spec)
    )

    # Add the new layer that we'll be training:
    with graph.as_default():
        (train_step, cross_entropy, bottleneck_input,
         ground_truth_input, final_tensor) = add_final_retrain_ops(
            class_count, CMD_ARG_FLAGS.final_tensor_name, bottleneck_tensor,
            wants_quantization, is_training=True)

    ''' Training Loop: '''
    with tf.Session(graph=graph) as sess:
        # Initialize all weights: for the module to their pretrained values,
        # and for the newly added retraining layer to random initial values.
        init = tf.global_variables_initializer()
        sess.run(init)

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
        cache_bottlenecks(sess, image_lists, CMD_ARG_FLAGS.train_image_dir,
                          CMD_ARG_FLAGS.bottleneck_dir, jpeg_data_tensor,
                          decoded_image_tensor, resized_image_tensor,
                          bottleneck_tensor, CMD_ARG_FLAGS.tfhub_module)

        # Create operations to evaluate the accuracy of the new layer:
        evaluation_step, _ = add_evaluation_step(final_tensor, ground_truth_input)

        # Merge all summaries and write them out to the summaries_dir
        merged = tf.summary.merge_all()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TensorFlow Transfer Learning Demo on Going Deeper Herbaria 1K Dataset')
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--train_image_dir',
        type=str,
        default='',
        help='Path to folders of labeled training images.'
    )
    parser.add_argument(
        '--test_image_dir',
        type=str,
        default='',
        help='Path to folders of labeled testing images.'
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
        default='/tmp/bottleneck',
        help='Path to cache bottleneck layer values as files.'
    )
    # Parse command line args and identify unknown flags:
    CMD_ARG_FLAGS, unparsed = parser.parse_known_args()
    '''
    Execute this script under a shell instead of importing as a module. Ensures that the main function is called with
    the proper command line arguments (builds on default argparse). For more information see:
    https://stackoverflow.com/questions/33703624/how-does-tf-app-run-work
    '''
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)