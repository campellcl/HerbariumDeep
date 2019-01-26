"""
BottleneckExecutor.py
Manages the computation, storage, and retrieval of all bottlneck files and associated metadata.
"""

import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import collections
import numpy as np
import time

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
MAX_IMAGE_BATCH_SIZE = 100  # With more than 555 sized: [299, 299, 3] images per forward pass, hitting OOM GPU? errors

# class BottleneckExecutor(object):
#     def __init__(self, bottlenecks_compressed_file_path, image_dir=None):
#         """
#         __init__: Handles instantiation of objects of type BottleneckExecutor. Ensures that the provided file path for
#             parameter 'bottleneck_compressed_file_path' actually points to a loadable dataframe object. This method
#             additionally ensures that if an 'image_dir' parameter is provided, it points to an existing directory with
#             read permissions. If either of these preconditions fails, instantiation will not continue, and this instance
#             is to terminate immediately while notifying the invoker.
#         :param bottlenecks_compressed_file_path: <str> The file path pointing to the compressed dataframe of bottleneck
#             files (traditionally named 'bottlenecks.pkl').
#         :param image_dir: <str> The directory containing all images to be used in the creation of new bottleneck vectors
#             in the event that the process was not completed previously.
#         """
#         self.existing_compressed_bottlenecks_file_pre_initialization = False
#         if os.path.exists(bottlenecks_compressed_file_path):
#             # The provided storage location is a valid file path.
#             self.bottlenecks_file_path = bottlenecks_compressed_file_path
#             try:
#                 self.bottlenecks = pd.read_pickle(bottlenecks_compressed_file_path)
#             except Exception as err:
#                 print(err)
#                 exit(-1)
#             # The provided bottleneck file was able to be read from disk:
#             self.existing_compressed_bottlenecks_file_pre_initialization = True
#             tf.logging.info(msg='Successfully read compressed bottleneck file at: \'%s\' from disk.'
#                                 % bottlenecks_compressed_file_path)
#         else:
#             # The provided storage path does not exist.
#             print('Couldn\'t identify the existence of the compressed bottlenecks file at the provided path: \'%s\'.'
#                   % bottlenecks_compressed_file_path)
#         if image_dir is not None:
#             if os.path.exists(image_dir):
#                 self.image_dir = image_dir
#                 tf.logging.info(msg='Confirmed existence of provided image directory: \'%s\'.' % image_dir)
#             else:
#                 print('Couldn\'t find the provided image directory: \'%s\' on this machine.' % image_dir)
#                 exit(-1)
#             # The image_dir flag was supplied indicating that bottleneck computation may be unfinished and needs resume.
#             if not self.existing_compressed_bottlenecks_file_pre_initialization:
#                 # An image directory was supplied and there was no pre-existing bottleneck file. Start from scratch.
#                 self._generate_bottlenecks()
#             else:
#                 '''
#                 An image directory was supplied and there was a pre-existing bottleneck file read from the hard drive.
#                 Bottleneck computation was left previously unfinished, new images have been added, or the user does not
#                     recall if bottleneck computation was previously finished.
#                 '''
#                 # Check to see if any new bottlenecks need to be generated:
#                 tf.logging.info('Checking to see if new bottleneck files need to be computed...')
#                 if self._is_finished_computing_all_bottlenecks():
#                     raise NotImplementedError
#                 else:
#                     self._generate_bottlenecks()
#                 pass
#         else:
#             # The image_dir flag was not supplied.
#             if self.existing_compressed_bottlenecks_file_pre_initialization:
#                 # The bottleneck file was able to be read from disk.
#                 # Warn the user:
#                 print('FATAL WARNING: By invoking this object\'s creation with a valid bottlenecks.pkl file pathway, '
#                       'and by failing to supply an image directory: this script will assume the created bottlenecks.pkl '
#                       'file has been updated with ALL sample images. If this is not the case, and additional bottleneck '
#                       'files do need to be generated (i.e. new sample images have been added, or this script was '
#                       'previously interrupted prior to finishing calculations) then YOU ARE REQUIRED to provide a valid '
#                       'image directory during instantiation!')
#                 print('FATAL WARNING: You have been warned. Mini-batch sampling will target ONLY images in the provided'
#                       'bottleneck.pkl file during training. Instantiation finished, proceeding.')
#
#     def _create_image_lists(self, random_state=0):
#         """
#         create_image_lists: Creates a dictionary containing all available datasets (train, test, validate) as a list of
#         image file paths (indexed by the class label).
#         :param random_state: A seed for the random number generator controlling the stratified partitioning.
#         :return:
#         """
#         image_dir = self.image_dir
#         if not tf.gfile.Exists(image_dir):
#             tf.logging.error("Root image directory '" + image_dir + "' not found.")
#             return None
#         accepted_extensions = ['jpg', 'jpeg']   # Note: Includes JPG and JPEG b/c the glob is case insensitive
#         image_lists = collections.OrderedDict()
#
#         # TODO: This tf.gfile.Walk takes a very long time, maybe go async? It seems to cache the walk somehow...
#         sub_dirs = sorted(x[0] for x in tf.gfile.Walk(image_dir))
#
#         # The root directory comes first, so skip it.
#         is_root_dir = True
#         for sub_dir in sub_dirs:
#             file_list = []
#             dir_name = os.path.basename(sub_dir)
#             if is_root_dir:
#                 is_root_dir = False
#                 # Skip the root_dir:
#                 continue
#             if dir_name == image_dir:
#                 # Return control to beginning of for-loop:
#                 continue
#             tf.logging.info("Looking for images in '" + dir_name + "'")
#             for extension in accepted_extensions:
#                 # Get a list of all accepted file extensions and the targeted file_name:
#                 file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
#                 # Append all items from the file_glob to the list of files (if extension exists):
#                 file_list.extend(tf.gfile.Glob(file_glob))
#             if not file_list:
#                 tf.logging.warning(msg='No files found in \'%s\'. Class label omitted from data sets.' % dir_name)
#                 # Return control to beginning of for-loop:
#                 continue
#             if len(file_list) < 20:
#                 tf.logging.warning('WARNING: Folder has less than 20 images, which may cause issues. See: %s for info.'
#                                    % 'https://stackoverflow.com/questions/38175673/critical-tensorflowcategory-has-no-images-validation')
#             elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
#                 tf.logging.warning(
#                     'WARNING: Folder {} has more than {} images. Some images will '
#                     'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
#             # label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
#             label_name = dir_name.lower()
#             image_lists[label_name] = {
#                 'dir': dir_name,
#                 'train': file_list,
#                 'val': None,
#                 'test': None
#             }
#         self.image_lists = image_lists
#
#     def _is_finished_computing_all_bottlenecks(self):
#         """
#         _is_finished_computing_all_bottlenecks: Returns true if bottleneck vectors have been calculated for every image
#             detected in the provided image_dir. This method caches it's value so that the computation is only performed
#             once.
#         :return:
#         """
#         if not hasattr(self, 'image_lists'):
#             tf.logging.info(msg='Generating image lists, walking all image sub-directories and recording file paths...')
#             self._create_image_lists()
#             tf.logging.info(msg='Finished generating image lists. Asserting existence of a bottleneck file for every '
#                                 'sample image in the generated image lists, cross-referencing compressed bottlenecks '
#                                 'dataframe...')
#         if not hasattr(self, 'finished_all_bottleneck_computations'):
#             num_labels = len(list(self.image_lists.keys()))
#             for i, class_label in enumerate(list(self.image_lists.keys())):
#                 tf.logging.info('[%d/%d] Class: \'%s\' has all bottlenecks computed.' % (i, num_labels, class_label))
#                 for sample_image in self.image_lists[class_label]['train']:
#                     if sample_image not in self.bottlenecks['path'].values:
#                         self.finished_all_bottleneck_computations = False
#                         return False
#             self.finished_all_bottleneck_computations = True
#             tf.logging.info(msg='Every bottleneck accounted for. All sample images under the image_dir: \'%s\' '
#                                 'have a corresponding pre-calculated bottleneck vector.' % self.image_dir)
#             return True
#         else:
#             # TODO: Ensure that the cached value is updated if sample image directories are modified during execution.
#             return self.finished_all_bottleneck_computations
#
#     def image_has_computed_bottleneck(self, image_path):
#         return image_path in self.bottlenecks['path']
#
#
#     def _compute_all_bottlenecks(self):
#         bottlenecks_empty = pd.DataFrame(columns=['class', 'path', 'bottleneck'])
#         for clss in self.image_lists.keys():
#             for category in ['train', 'val', 'test']:
#                 for image_path in self.image_lists[clss][category]:
#                     new_bottleneck_entry = pd.Series([clss, image_path, None], index=['class', 'path', 'bottleneck'])
#                     bottlenecks_empty = bottlenecks_empty.append(new_bottleneck_entry, ignore_index=True)
#         bottlenecks = bottlenecks_empty.copy(deep=True)
#         num_bottlenecks = 0
#         for i, (clss, series) in enumerate(bottlenecks_empty.iterrows()):
#             image_path = series['path']
#             if not tf.gfile.Exists(image_path):
#                 tf.logging.fatal('File does not exist %s', image_path)
#             image_data = tf.gfile.FastGFile(image_path, 'rb').read()
#             bottleneck_tensor_values = _calculate_bottleneck_value(sess=sess, image_path=image_path, image_data=image_data,
#                                                               image_data_tensor=jpeg_data_tensor,
#                                                               decoded_image_tensor=decoded_image_tensor,
#                                                               resized_input_tensor=resized_input_tensor,
#                                                               bottleneck_tensor=bottleneck_tensor)
#             bottlenecks.iat[i, 2] = bottleneck_tensor_values
#             num_bottlenecks += 1
#             if num_bottlenecks % 100 == 0:
#                 tf.logging.info(msg='Computed %d bottleneck arrays.' % num_bottlenecks)
#                 bottlenecks.to_pickle(os.path.basename(CMD_ARG_FLAGS.bottleneck_path))
#
#         tf.logging.info(msg='Finished computing bottlenecks. Saving dataframe to: \'%s\'' % CMD_ARG_FLAGS.bottleneck_path)
#         bottlenecks.to_pickle(os.path.basename(CMD_ARG_FLAGS.bottleneck_path))
#         return bottlenecks
#
#     def _generate_bottlenecks(self):
#         num_sample_images = 0
#         num_bottlenecks_calculated = 0
#         num_bottlenecks_to_calculate = 0
#         if self.existing_compressed_bottlenecks_file_pre_initialization:
#             # Previously existing bottleneck file.
#             if self.image_dir is not None:
#                 # Provided image directory, does computation need to be resumed?
#                 if self._is_finished_computing_all_bottlenecks():
#                     # Just load the bottleneck file:
#                     raise NotImplementedError
#                 else:
#                     tf.logging.info(msg='Identified at least one sample image without a corresponding bottleneck file. '
#                                         'Determining which sample images need corresponding bottlenecks computed...')
#                     images_without_bottlenecks = collections.OrderedDict()
#                     for class_label in list(self.image_lists.keys()):
#                         class_images_without_bottlenecks = []
#                         for sample_image_path in self.image_lists[class_label]['train']:
#                             num_sample_images += 1
#                             if sample_image_path not in self.bottlenecks['path'].values:
#                                 class_images_without_bottlenecks.append(sample_image_path)
#                                 num_bottlenecks_to_calculate += 1
#                             else:
#                                 num_bottlenecks_calculated += 1
#                         images_without_bottlenecks[class_label] = {
#                             'dir': class_label,
#                             'images': class_images_without_bottlenecks
#                         }
#                     tf.logging.info(msg='Determined which images still need bottlenecks computed. '
#                                         'Statistics:\n\t'
#                                         'Total number of sample images: %d\n\t'
#                                         'Number of calculated bottlenecks: %d\n\t'
#                                         'Remaining bottlenecks to calculate: %d'
#                                         % (num_sample_images, num_bottlenecks_calculated, num_bottlenecks_to_calculate))
#                     # Resume computation of the bottlenecks:
#                     raise NotImplementedError
#             else:
#                 # Previously existing bottleneck file, no provided image directory.
#                 # User has been warned during initialization, proceed with bottleneck load:
#                 raise NotImplementedError
#         else:
#             # No previously existing bottleneck file.
#             if self.image_dir is not None:
#                 # Provided image directory, start computation from scratch.
#                 self._compute_all_new_bottlenecks()
#                 raise NotImplementedError
#             else:
#                 # No provided image directory. No provided bottleneck file. Nothing to do.
#                 exit(0)

def _add_batch_jpeg_decoding(module_spec):
    input_height, input_width = hub.get_expected_image_size(module_spec)
    input_depth = hub.get_num_image_channels(module_spec)
    # placeholder tensor of any size, capable of taking input.size = [?, image_height, image_width, num_channels=3]
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')

def _add_jpeg_decoding(module_spec):
    """Adds operations that perform JPEG decoding and resizing to the graph..

    Args:
      module_spec: The hub.ModuleSpec for the image module being used.

    Returns:
      Tensors for the node to feed JPEG data into, and the output of the
        preprocessing steps.
    """
    input_height, input_width = hub.get_expected_image_size(module_spec)
    input_depth = hub.get_num_image_channels(module_spec)   # 3
    # placeholder Tensor of any size, capable of taking current input.shape() = [?, image_height, image_width, num_channels=3]
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    # Decode a single JPEG-encoded image to a unit8 tensor, with the desired number of color channels (3 in this case) for decoded img:
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    # Convert from full range of uint8 to range [0,1] of float32.
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image, tf.float32)
    # Insert a "batch dimension" of 1 to the existing decoded_image_as_float tensor so size is now: [1, ?, image_height, image_width, 3]
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    ''' 
    Tensors are decoded and represented as 3-d unit8 tensors of shape [height, width, channels], that is shape=(3,)
    (see: https://www.tensorflow.org/api_guides/python/image). This tf.stack call seems to go from:
        [input_height=299, input_width=299] -> [input_height=299, input_width=299] with .shape == (2,) e.g. row vector
    I don't see why this call is here:
    '''
    resize_shape = tf.stack([input_height, input_width])
    # Switch back to int32, not sure why we do this, probably to save memory space? Float precision for [0-255] is unnecessary.
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    # resize the decoded image using bilinear interpolation, this produces shape (1, 299, 299, 3) at runtime for a single image.
    #   I am not sure why this is needed for a scalar decoded image, although I see how this might be needed for a batch of images:
    resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)
    return jpeg_data, resized_image


def _build_graph(tfhub_module_url):
    module_spec = hub.load_module_spec(tfhub_module_url)
    height, width = hub.get_expected_image_size(module_spec)
    tf.logging.info(msg='Loaded TensorFlowHub module spec: %s' % tfhub_module_url)
    graph = tf.Graph()
    with graph.as_default():
        # Create a placeholder tensor for image input to the model (when bottleneck has not been pre-computed).
        resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3], name='resized_input')
        # Declare the model in accordance with the chosen architecture:
        m = hub.Module(module_spec)
        # Create a placeholder tensor to catch the output of the pre-activation layer:
        bottleneck_tensor = m(resized_input_tensor)
        tf.logging.info(msg='Defined computational graph from the tensorflow hub module spec.')

        # Image decoding sub-graph:
        with tf.name_scope('image_decoding'):
            jpeg_data_tensor, decoded_image_tensor = _add_jpeg_decoding(module_spec)

    return graph, bottleneck_tensor, resized_input_tensor, jpeg_data_tensor, decoded_image_tensor


def _get_image_lists(image_dir):
    """
    _get_image_lists: Creates a dictionary of file paths to images on the hard drive indexed by class label.
    :param image_dir: <str> The file path pointing to the parent directory housing a series of sample images partitioned
        into their respective subdirectories by class label.
    :return image_lists: <collections.OrderedDict> A dictionary indexed by class label, which provides as its value a
        list of file paths for images belonging to the chosen key/species/class-label.
    """
    '''
    Check to see if the root directory exists. We use tf.gfile which is a C++ FileSystem API wrapper for the Python
        file API that also supports Google Cloud Storage and HDFS. For more information see:
        https://stackoverflow.com/questions/42256938/what-does-tf-gfile-do-in-tensorflow
    '''
    if not tf.gfile.Exists(image_dir):
        tf.logging.error("Root image directory '" + image_dir + "' not found.")
        return None

    accepted_extensions = ['jpg', 'jpeg']   # Note: Includes JPG and JPEG b/c the glob is case insensitive
    image_lists = collections.OrderedDict()

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
        if label_name not in image_lists:
            image_lists[label_name] = file_list
    return image_lists


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


def _calculate_bottleneck_value(sess, image_path, image_data, image_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor):
    """
    calculate_bottleneck_value: Forward propagates the provided image through the original source network to produce the
        output tensor associated with the penultimate layer (pre softmax).
    :param sess: <tf.Session> The current active TensorFlow Session.
    :param image_path: <str> The file path to the original input image (used for debug purposes).
    :param image_data: <str> String of raw JPEG data.
    :param image_data_tensor: <?> Input data layer in the graph.
    :param decoded_image_tensor: <?> Output of initial image resizing and preprocessing.
    :param resized_input_tensor: <?> The input node of the source/recognition graph.
    :param bottleneck_tensor: <?> The penultimate node before the final softmax layer of the source/recognition graph.
    :return bottleneck_tensor_value: <numpy.ndarray> The result of forward propagating the provided image through the
        source/recognition graph.
    """
    tf.logging.info('Creating bottleneck for sample image ' + image_path)
    try:
        bottleneck_tensor_value = run_bottleneck_on_image(sess=sess, image_data=image_data,
                                                    image_data_tensor=image_data_tensor,
                                                    decoded_image_tensor=decoded_image_tensor,
                                                    resized_input_tensor=resized_input_tensor,
                                                    bottleneck_tensor=bottleneck_tensor)
    except Exception as e:
        raise RuntimeError('Error during bottleneck processing file %s (%s)' % (image_path,
                                                                     str(e)))
    return bottleneck_tensor_value


def _calculate_bottleneck_values_in_batch(sess, images_data, image_data_tensor, decoded_image_tensor,
                                          resized_input_tensor, bottleneck_tensor):
    """
    _calculate_bottleneck_values_in_batch: Forward propagates the list of provided images thorugh the original source
        network to produce the output tensor associated with the penultimate layer (pre-softmax).
    :param sess: <tf.Session> The current active TensorFlow session.
    :param images_data: A list of decoded images corresponding to the list of image paths.
    :param image_data_tensor: The placeholder tensor which takes full sized input images and performs resize operations
        in the target domain for compatibility with the source classifier.
    :param decoded_image_tensor: A placeholder tensor to store the result of the resize operations.
    :param resized_input_tensor: A placeholder tensor which takes resized and decoded images and then feeds them into
        the computational graph.
    :param bottleneck_tensor: A placeholder tensor storing the results of the forward propagation.
    :return:
    """
    # tf.logging.info('Creating bottleneck images for class: %s' % os.path.dirname(image_paths[0]).split('\\')[-1])
    resized_input_values = []
    for image_data in images_data:
        resized_input_value = sess.run(decoded_image_tensor, {image_data_tensor: image_data})
        resized_input_values.append(resized_input_value)
    try:
        resized_input_values = np.squeeze(resized_input_values)
        bottleneck_values = sess.run(bottleneck_tensor, {resized_input_tensor: resized_input_values})
        # bottleneck_values = np.squeeze(bottleneck_values)
        return bottleneck_values
    except Exception as err:
        tf.logging.error(msg=err)


def _decode_image_from_path(image_path):
    image_string = tf.read_file(image_path)
    image_decoded = tf.image.decode_jpeg(image_string)
    return image_decoded


def _cache_all_bottlenecks(sess, image_lists, jpeg_data_tensor, decoded_image_tensor, resized_image_tensor, bottleneck_tensor):
    """
    cache_all_bottlenecks: Takes every sample image in every dataset (train, val, test) and forward propagates the
        sample's Tensor through the original source network. At the penultimate layer of the provided source network
        (that is, pre-softmax layer) each sample's output tensors are recorded in lieu of the original input image.
        These recorded output tensors constitute the 'bottlenecks' of the provided image_lists. This method computes
        and stores the calculated bottlenecks in a bottlenecks dataframe for retrieval at a later time during training
        or evaluation.
        WARNING: This method currently assumes that all bottlenecks will fit in a single output dataframe. If this is
            not the case then this method (and prior logic) needs to be updated.
    :param sess: <tf.Session> The current active TensorFlow Session.
    :param image_metadata: <dict> Contains information regarding the distribution of images among training, validation,
        and testing datasets.
        image_metadata['num_train_images']: Number of training images.
        image_metadata['num_val_images']: Number of validation images.
        image_metadata['num_test_images']: Number of testing images.
        image_metadata['num_images']: Total number of sample images.
    :param image_lists: <OrderedDict> of training images for each label.\
    :param jpeg_data_tensor: <tf.Tensor?> Input tensor for jpeg data from file.
    :param decoded_image_tensor: <tf.Tensor?> The tensor holding the output of the image decoding sub-graph.
    :param resized_input_tensor: <tf.Tensor?> The input node of the source/recognition graph.
    :param bottleneck_tensor: <tf.Tensor?> The penultimate (pre-softmax) output node of the source/recognition graph.
    :return bottlenecks: <pd.DataFrame> A dataframe containing the bottleneck values for every tensor in the
        datasets.
    """
    # If the specified bottleneck directory doesn't exist, create it:
    if not os.path.exists(os.path.dirname(bottleneck_path)):
        os.mkdir(os.path.dirname(bottleneck_path))
    col_names = ['class', 'path', 'bottleneck']
    df_bottlenecks = pd.DataFrame(columns=col_names)
    df_bottlenecks['class'] = df_bottlenecks['class'].astype('category')
    num_classes = len(image_lists.keys())
    bottleneck_counts_and_time_stamps = []
    for i, clss in enumerate(image_lists.keys()):
        image_paths = image_lists[clss]
        images_data = [tf.gfile.GFile(image_path, 'rb').read() for image_path in image_paths]
        '''
        Have to use batch size for the forward propagation with images, because OOM RAM errors occur otherwise. 
        '''
        image_data_batches = [images_data[i:i + MAX_IMAGE_BATCH_SIZE] for i in range(0, len(images_data), MAX_IMAGE_BATCH_SIZE)]
        tf.logging.info('[%d/%d] Computing bottleneck values for %d samples in class: \'%s\''
                        % (i, num_classes, len(image_paths), clss))
        for j, image_data_batch in enumerate(image_data_batches):
            tf.logging.info('\tComputing batch [%d/%d]...' % (j, len(image_data_batches)))
            ts = time.time()
            bottlenecks = _calculate_bottleneck_values_in_batch(
                sess=sess, images_data=image_data_batch, image_data_tensor=jpeg_data_tensor,
                decoded_image_tensor=decoded_image_tensor, resized_input_tensor=resized_image_tensor,
                bottleneck_tensor=bottleneck_tensor
            )
            bottleneck_counts_and_time_stamps.append((len(image_data_batch), time.time() - ts))
            for k, img_path in enumerate(image_paths):
                df_bottlenecks.loc[len(df_bottlenecks)] = {'class': clss, 'path': img_path, 'bottleneck': bottlenecks[k]}
        average_bottleneck_computation_rate = sum([num_bottlenecks / elapsed_time for num_bottlenecks, elapsed_time in bottleneck_counts_and_time_stamps])/len(bottleneck_counts_and_time_stamps)
        tf.logging.info(msg='\tFinished computing class bottlenecks. Average bottleneck generation rate: %.2f bottlenecks per second.' % average_bottleneck_computation_rate)
        if i % 100 == 0:
            tf.logging.info(msg='\tBacking up dataframe to: \'%s\'' % bottleneck_path)
            df_bottlenecks.to_pickle(bottleneck_path)

        # file_paths = tf.constant(image_paths)
        # image_tf_dataset = tf.data.Dataset.from_tensor_slices((file_paths))
        # image_tf_dataset = image_tf_dataset.map(_decode_image_from_path).batch(2)
        # iterator = image_tf_dataset.make_one_shot_iterator()
        # next_image_batch = iterator.get_next()
        # TODO: Get batches of images to feed forward

    #     for image_path in image_lists[clss]:
    #         new_bottleneck_entry = pd.Series([clss, image_path, None], index=['class', 'path', 'bottleneck'])
    #         bottlenecks_empty = bottlenecks_empty.append(new_bottleneck_entry, ignore_index=True)
    # bottlenecks = bottlenecks_empty.copy(deep=True)

    # tf.logging.info(msg='Generated empty bottleneck dataframe. Propagating with bottleneck values...')
    # num_bottlenecks = 0
    # for i, (clss, series) in enumerate(bottlenecks_empty.iterrows()):
    #     image_path = series['path']
    #     if not tf.gfile.Exists(image_path):
    #         tf.logging.fatal('File does not exist %s', image_path)
    #     image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    #     bottleneck_tensor_values = _calculate_bottleneck_value(sess=sess, image_path=image_path, image_data=image_data,
    #                                                           image_data_tensor=jpeg_data_tensor,
    #                                                           decoded_image_tensor=decoded_image_tensor,
    #                                                           resized_input_tensor=resized_image_tensor,
    #                                                           bottleneck_tensor=bottleneck_tensor)
    #     bottlenecks.iat[i, 2] = bottleneck_tensor_values
    #     num_bottlenecks += 1
    #     if num_bottlenecks % 1000 == 0:
    #         tf.logging.info(msg='Computed %d bottleneck arrays.' % num_bottlenecks)
    #         bottlenecks.to_pickle(os.path.basename(bottleneck_path))
    #
    tf.logging.info(msg='Finished computing bottlenecks. Saving dataframe to: \'%s\'' % bottleneck_path)
    df_bottlenecks.to_pickle(os.path.basename(bottleneck_path))
    return df_bottlenecks


def _resume_caching_bottlenecks(sess, bottlenecks, jpeg_data_tensor, decoded_image_tensor, resized_image_tensor, bottleneck_tensor):
    bottlenecks_updated = bottlenecks.copy(deep=True)
    num_bottlenecks = 0
    num_already_bottlenecked = 0
    for i, (clss, series) in enumerate(bottlenecks.iterrows()):
        image_path = series['path']
        bottleneck = series['bottleneck']
        if bottleneck is None:
            if not tf.gfile.Exists(image_path):
                tf.logging.fatal('File does not exist %s', image_path)
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()
            bottleneck_tensor_values = _calculate_bottleneck_value(sess=sess, image_path=image_path, image_data=image_data,
                                                                  image_data_tensor=jpeg_data_tensor,
                                                                  decoded_image_tensor=decoded_image_tensor,
                                                                  resized_input_tensor=resized_image_tensor,
                                                                  bottleneck_tensor=bottleneck_tensor)
            bottlenecks_updated.iat[i, 2] = bottleneck_tensor_values
            num_bottlenecks += 1
            if num_bottlenecks % 1000 == 0:
                tf.logging.info(msg='Computed %d new bottleneck vectors.' % num_bottlenecks)
                bottlenecks_updated.to_pickle(bottleneck_path)
        else:
            # bottleneck already computed.
            num_already_bottlenecked += 1
            tf.logging.info('Detected %d samples with bottleneck vectors already present.' % num_already_bottlenecked)

    tf.logging.info(msg='Finished computing bottlenecks. Saving dataframe to: \'%s\'' % bottleneck_path)
    bottlenecks_updated.to_pickle(bottleneck_path)
    return bottlenecks_updated


def main():
    # Set logging verbosity:
    tf.logging.set_verbosity(tf.logging.INFO)
    # Get image lists:
    image_lists = _get_image_lists(image_dir=image_path)
    # Build computational graph for bottleneck generation:
    tfhub_module_url = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'
    graph, bottleneck_tensor, resized_input_tensor, jpeg_data_tensor, decoded_image_tensor = _build_graph(tfhub_module_url=tfhub_module_url)

    # Build computational graph for batch bottleneck generation:
    tfhub_module_url = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'
    #   - Iterator construction:
    # for clss in image_lists.keys():
    #     image_paths = image_lists[clss]
    #     file_paths = tf.constant(image_paths)
    #     image_tf_dataset = tf.data.Dataset.from_tensor_slices((file_paths))
    #     image_tf_dataset = image_tf_dataset.map(_decode_image_from_path).batch(2)
    #     iterator = image_tf_dataset.make_one_shot_iterator()
    #     next_image_batch = iterator.get_next()

    # Run the graph in a session to generate bottleneck tensors:
    with tf.Session(graph=graph) as sess:
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        if os.path.isfile(bottleneck_path):
            bottlenecks_initial = pd.read_pickle(bottleneck_path)
            bottlenecks = _resume_caching_bottlenecks(sess, bottlenecks_initial, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor)
        else:
            bottlenecks = _cache_all_bottlenecks(sess, image_lists, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor)

    # bottleneck_executor = BottleneckExecutor(
    #     bottlenecks_compressed_file_path='D:\\data\\GoingDeeperData\\bottlenecks.pkl',
    #     image_dir='D:\data\GoingDeeperData\images'
    # )


if __name__ == '__main__':
    # Debug Configuration:
    # bottleneck_path = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\TensorFlow\\TFHub\\bottlenecks.pkl'
    # image_path = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\GoingDeeper\\images'
    # GoingDeeper Configuration:
    bottleneck_path = 'D:\\data\\GoingDeeperData\\bottlenecks.pkl'
    image_path = 'D:\\data\\GoingDeeperData\\images'
    main()
