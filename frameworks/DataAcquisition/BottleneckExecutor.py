"""
BottleneckExecutor.py
Manages the computation, storage, and retrieval of all bottlneck files and associated metadata.
"""

import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import collections
from sklearn import model_selection
import numpy as np
import time
from frameworks.DataAcquisition.ImageExecutor import ImageExecutor

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
# MAX_IMAGE_BATCH_SIZE = 400  # With more than 555 sized: [299, 299, 3] images per forward pass, hitting OOM GPU? errors
MAX_IMAGE_BATCH_SIZE = 400

def _add_jpeg_decoding(module_spec):
    """Adds operations that perform JPEG decoding and resizing to the graph...
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


class BottleneckExecutor:
    image_dir = None
    compressed_bottleneck_file_path = None
    tfhub_module_url = None
    image_executor = None
    image_lists = None
    df_bottlenecks = None
    cached_all_bottlenecks = False
    # Tensors to maintain references to:
    graph = None
    _session = None
    bottleneck_tensor = None
    resized_image_tensor = None
    jpeg_data_tensor = None
    decoded_image_tensor = None
    # bottlenecks compressed dataframe:
    bottlenecks = None

    def __init__(self, image_dir, logging_dir, tfhub_module_url, compressed_bottleneck_file_path, image_executor_instance=None):
        self.image_dir = image_dir
        self.logging_dir = logging_dir
        self.tfhub_module_url = tfhub_module_url
        self.compressed_bottleneck_file_path = compressed_bottleneck_file_path
        # Set logging verbosity:
        tf.logging.set_verbosity(tf.logging.INFO)
        if image_executor_instance is not None:
            self.image_executor = image_executor_instance
        else:
            self.image_executor = ImageExecutor(
                img_root_dir=self.image_dir,
                accepted_extensions=['jpg', 'jpeg'],
                logging_dir=logging_dir,
                min_num_images_per_class=20
            )
        # Clean and retrieve image lists:
        self.image_lists = self.image_executor.get_image_lists()
        # Build computational graph for bottleneck generation:
        # self.tfhub_module_url = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'
        self.graph, self.bottleneck_tensor, self.resized_image_tensor, \
            self.jpeg_data_tensor, self.decoded_image_tensor = _build_graph(tfhub_module_url=self.tfhub_module_url)

    def _calculate_bottleneck_value(self, image_path, image_data=None):
        if self._session is not None:
            # Use the existing session:
            if image_data is None:
                # Need to run image decoding:
                image_data = tf.gfile.GFile(image_path, 'rb').read()
            # First decode the JPEG image, resize it, and rescale the pixel values.
            resized_input_value = self._session.run(self.decoded_image_tensor, {self.jpeg_data_tensor: image_data})
            # Then run it through the source network:
            bottleneck_value = self._session.run(self.bottleneck_tensor, {self.resized_image_tensor: resized_input_value})
            bottleneck_value = np.squeeze(bottleneck_value)
            return bottleneck_value

    def _calculate_bottleneck_values_in_batch(self, images_data):
        """
        _calculate_bottleneck_values_in_batch: Forward propagates the list of provided images thorugh the original source
            network to produce the output tensor associated with the penultimate layer (pre-softmax).
        :param images_data: A list of decoded images corresponding to the list of image paths.
        :return bottleneck_values: A tensor of bottleneck values for the provided images.
        """
        if self._session is not None:
            resized_input_values = []
            for image_data in images_data:
                resized_input_value = self._session.run(self.decoded_image_tensor, {self.jpeg_data_tensor: image_data})
                resized_input_values.append(resized_input_value)
            try:
                resized_input_values = np.squeeze(resized_input_values)
                bottleneck_values = self._session.run(self.bottleneck_tensor, {self.resized_image_tensor: resized_input_values})
                del resized_input_values
                # bottleneck_values = np.squeeze(bottleneck_values)
                return bottleneck_values
            except Exception as err:
                tf.logging.error(msg=err)

    def _cache_all_bottlenecks(self):
        """
        _cache_all_bottlenecks: Takes every sample image in every dataset (train, val, test) and forward propagates the
            sample's Tensor through the original source network. At the penultimate layer of the provided source network
            (that is, pre-softmax layer) each sample's output tensors are recorded in lieu of the original input image.
            These recorded output tensors constitute the 'bottlenecks' of the provided image_lists. This method computes
            and stores the calculated bottlenecks in a bottlenecks dataframe for retrieval at a later time during training
            or evaluation.
            WARNING: This method currently assumes that all bottlenecks will fit in a single output dataframe. If this is
                not the case then this method (and prior logic) needs to be updated.
        :return bottlenecks: <pd.DataFrame> A dataframe containing the bottleneck values for every tensor in the
            datasets.
        """
        # If the specified bottleneck directory doesn't exist, create it:
        if not os.path.exists(os.path.dirname(self.compressed_bottleneck_file_path)):
            os.mkdir(os.path.dirname(self.compressed_bottleneck_file_path))
        # Do some dataframe setup:
        col_names = ['class', 'path', 'bottleneck']
        df_bottlenecks = pd.DataFrame(columns=col_names)
        df_bottlenecks['class'] = df_bottlenecks['class'].astype('category')
        target_classes = list(self.image_lists.keys())
        num_classes = len(target_classes)
        # Keep track of the amount of bottlenecks created, and the elapsed time taken to create them:
        bottleneck_counts_and_time_stamps = []

        # Run the graph in a session to generate bottleneck tensors:
        self._session = tf.Session(graph=self.graph)
        with self._session as sess:
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
            for i, clss in enumerate(target_classes):
                image_paths = self.image_lists[clss]
                '''
                Since the images are high quality, they take quite a bit of RAM. Only load in images in denominations of
                    MAX_IMAGE_BATCH_SIZE.
                '''
                image_path_batches = [image_paths[i:i + MAX_IMAGE_BATCH_SIZE] for i in range(0, len(image_paths), MAX_IMAGE_BATCH_SIZE)]
                tf.logging.info('[%d/%d] Computing bottleneck vectors for %d samples in class \'%s\'' % (i+1, num_classes, len(image_paths), clss))
                # Iterate over batches of image paths:
                for j, image_path_batch in enumerate(image_path_batches):
                    image_data_batch = [tf.gfile.GFile(image_path, 'rb').read() for image_path in image_path_batch]
                    tf.logging.info('\tComputing batch [%d/%d]...' % (j+1, len(image_path_batches)))
                    ts = time.time()
                    if len(image_data_batch) == 1:
                        # The batch partitioning resulted in a single scalar by itself:
                        bottleneck = self._calculate_bottleneck_value(
                            image_path=image_path_batches[j][0],
                            image_data=image_data_batch[0]
                        )
                        bottleneck_counts_and_time_stamps.append((1, time.time() - ts))
                        # Append the generated bottleneck to the dataframe:
                        df_bottlenecks.loc[len(df_bottlenecks)] = {
                            'class': clss,
                            'path': image_path_batches[j][0],
                            'bottleneck': bottleneck
                        }
                    else:
                        # A single batch of images:
                        bottlenecks = self._calculate_bottleneck_values_in_batch(images_data=image_data_batch)
                        bottleneck_counts_and_time_stamps.append((len(image_data_batch), time.time() - ts))
                        # Append the generated bottlenecks to the dataframe:
                        for k, img_path in enumerate(image_path_batches[j]):
                            df_bottlenecks.loc[len(df_bottlenecks)] = {'class': clss, 'path': img_path, 'bottleneck': bottlenecks[k]}
                average_bottleneck_computation_rate = sum([num_bottlenecks / elapsed_time for num_bottlenecks, elapsed_time in bottleneck_counts_and_time_stamps])/len(bottleneck_counts_and_time_stamps)
                tf.logging.info(msg='\tFinished computing class bottlenecks. Average bottleneck generation rate: %.2f bottlenecks per second.' % average_bottleneck_computation_rate)
                if i % 10 == 0:
                    tf.logging.info(msg='\tBacking up dataframe to: \'%s\'' % self.compressed_bottleneck_file_path)
                    df_bottlenecks.to_pickle(self.compressed_bottleneck_file_path)
            self.df_bottlenecks = df_bottlenecks
            self.cached_all_bottlenecks = True
            tf.logging.info(msg='Finished computing ALL bottlenecks. Saving final dataframe to: \'%s\'' % self.compressed_bottleneck_file_path)
            df_bottlenecks.to_pickle(self.compressed_bottleneck_file_path)

    def _load_bottlenecks(self):
        bottlenecks = None
        bottleneck_path = self.compressed_bottleneck_file_path
        if os.path.isfile(bottleneck_path):
            # Bottlenecks .pkl file exists, read from disk:
            tf.logging.info(msg='Bottleneck file successfully located at the provided path: \'%s\'' % bottleneck_path)
            try:
                # bottlenecks = pd.read_pickle(bottleneck_path)
                bottlenecks = pd.read_csv(bottleneck_path.replace('.pkl', '.csv'))
                bottlenecks['bottleneck'] = list(np.load(bottleneck_path.replace('.pkl', '.npy')))
                # dfb['bottleneck'] = list(np.load('bottlenecks_partial.npy'))
                tf.logging.info(msg='Bottleneck file \'%s\' successfully restored from disk.'
                                    % os.path.basename(bottleneck_path))
            except EOFError as err:
                tf.logging.error(msg='Failed to read the bottleneck file, it may have become corrupted. Recommend manual '
                                     'restoration from D:/backups; and re-executing script.')
                exit(-1)
        else:
            tf.logging.error(msg='Bottleneck file not located at the provided path: \'%s\'. '
                                 'Have you run BottleneckExecutor.py?' % bottleneck_path)
            exit(-1)
        return bottlenecks

    def _resume_caching_bottlenecks(self):
        if os.path.exists(self.compressed_bottleneck_file_path):
            self.df_bottlenecks = self._load_bottlenecks()
            # df_bottlenecks = existing_bottlenecks.copy(deep=True)
        # else:
        #     self._cache_all_bottlenecks()
        #     exit(0)
        image_lists = self.image_executor.get_image_lists()
        resume_class_label = None
        resume_class_label_index = -1
        for i, (clss_label, image_paths) in enumerate(image_lists.items()):
            if clss_label not in self.df_bottlenecks['class'].values:
                # Resume from this class label
                resume_class_label = clss_label
                resume_class_label_index = i
                break
        if resume_class_label_index == -1:
            tf.logging.info(msg='All bottlenecks have already been computed! BottleneckExecutor will provide access '
                                'to verbatim loaded bottlenecks\' dataframe.')
            # self.df_bottlenecks = df_bottlenecks
            self.cached_all_bottlenecks = True
            return
        target_classes = list(self.image_lists.keys())
        num_classes = len(target_classes)
        tf.logging.info(msg='Resuming previously interrupted bottleneck computation at [%d/%d] class label: \'%s\'' % (resume_class_label_index, num_classes, resume_class_label))

        # Keep track of the amount of bottlenecks created, and the elapsed time taken to create them:
        bottleneck_counts_and_time_stamps = []

        # Run the graph in a session to generate bottleneck tensors:
        self._session = tf.Session(graph=self.graph)

        with self._session as sess:
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
            sess.graph.finalize()
            for i, clss in enumerate(target_classes):
                if i < resume_class_label_index:
                    continue
                image_paths = self.image_lists[clss]
                '''
                Since the images are high quality for the BOONE dataset, they take a large amount of RAM. Only load in
                    images in denominations of MAX_IMAGE_BATCH_SIZE. 
                '''
                image_path_batches = [image_paths[i:i + MAX_IMAGE_BATCH_SIZE] for i in range(0, len(image_paths), MAX_IMAGE_BATCH_SIZE)]
                tf.logging.info('[%d/%d] Computing bottleneck values for %d samples in class: \'%s\''
                                % (i+1, num_classes, len(image_paths), clss))
                # Iterate over the batches of image paths:
                for j, image_path_batch in enumerate(image_path_batches):
                    image_data_batch = [tf.gfile.GFile(image_path, 'rb').read() for image_path in image_path_batch]
                    tf.logging.info('\tComputing batch [%d/%d]...' % (j+1, len(image_path_batches)))
                    ts = time.time()
                    if len(image_data_batch) == 1:
                        # The batch partitioning resulted in a single scalar by itself:
                        bottleneck = self._calculate_bottleneck_value(
                            image_path=image_path_batches[j][0],
                            image_data=image_data_batch[0]
                        )
                        bottleneck_counts_and_time_stamps.append((1, time.time() - ts))
                        # Append the generated bottleneck to the dataframe:
                        self.df_bottlenecks.loc[len(self.df_bottlenecks)] = {
                            'class': clss,
                            'path': image_path_batches[j][0],
                            'bottleneck': bottleneck
                        }
                    else:
                        # A single batch of images:
                        bottlenecks = self._calculate_bottleneck_values_in_batch(images_data=image_data_batch)
                        bottleneck_counts_and_time_stamps.append((len(image_data_batch), time.time() - ts))
                        # Append the generated bottlenecks to the dataframe:
                        for k, img_path in enumerate(image_path_batches[j]):
                            self.df_bottlenecks.loc[len(self.df_bottlenecks)] = {'class': clss, 'path': img_path, 'bottleneck': bottlenecks[k]}
                            '''
                            I know it's odd to have saving logic here. However, the MAX_IMAGE_BATCH_SIZE is determined
                                by the size of the image sizes and the amount of system RAM. Therefore, in order to avoid starving the OS
                                of RAM, it is necessary to periodically update the dataframe in denominations of MAX_IMAGE_BATCH_SIZE. If
                                this is not done, then the backup may be performed with more image data than is capable of fitting into memory. 
                            '''
                            if k == 0:
                                continue
                            if k % (MAX_IMAGE_BATCH_SIZE * 3) == 0:
                                tf.logging.info(msg='\tBacking up dataframe to: \'%s\'' % self.compressed_bottleneck_file_path)
                                self.df_bottlenecks[['class', 'path']].to_csv(self.compressed_bottleneck_file_path.replace('.pkl', '.csv'), index=False)
                                np.save(self.compressed_bottleneck_file_path.replace('.pkl', '.npy'), np.vstack(self.df_bottlenecks['bottleneck']), allow_pickle=False)
                                # dfb = pd.read_csv('bottlenecks_partial.csv')
                                # dfb['bottleneck'] = list(np.load('bottlenecks_partial.npy'))
                                # self.df_bottlenecks.to_pickle(self.compressed_bottleneck_file_path, pickle)
                                # self.df_bottlenecks.to_hdf(self.compressed_bottleneck_file_path.replace('.pkl', '.h5'), key='bottlenecks', mode='w')
                average_bottleneck_computation_rate = sum([num_bottlenecks / elapsed_time for num_bottlenecks, elapsed_time in bottleneck_counts_and_time_stamps])/len(bottleneck_counts_and_time_stamps)
                tf.logging.info(msg='\tFinished computing class bottlenecks. Average bottleneck generation rate: %.2f bottlenecks per second.' % average_bottleneck_computation_rate)
            self.cached_all_bottlenecks = True
            tf.logging.info(msg='Finished computing ALL bottlenecks. Saving final dataframe to: \'%s\'' % self.compressed_bottleneck_file_path)
            self.df_bottlenecks.to_pickle(self.compressed_bottleneck_file_path)

    def get_partitioned_bottlenecks(self, train_percent=.80, val_percent=.20, test_percent=.20, random_state=42):
        bottlenecks = self.get_bottlenecks()
        train_bottlenecks, test_bottlenecks = model_selection.train_test_split(
            bottlenecks, train_size=train_percent,
            test_size=test_percent, shuffle=True,
            random_state=random_state
        )
        train_bottlenecks, val_bottlenecks = model_selection.train_test_split(
            train_bottlenecks, train_size=train_percent,
            test_size=val_percent, shuffle=True,
            random_state=random_state
        )
        return train_bottlenecks, val_bottlenecks, test_bottlenecks

    def get_bottlenecks(self):
        if not self.cached_all_bottlenecks:
            self._resume_caching_bottlenecks()
        return self.df_bottlenecks

    # def cache_all_bottlenecks(self):
    #     """
    #     cache_all_bottlenecks: Takes every sample image in every dataset (train, val, test) and forward propagates the
    #         sample's Tensor through the original source network. At the penultimate layer of the provided source network
    #         (that is, pre-softmax layer) each sample's output tensors are recorded in lieu of the original input image.
    #         These recorded output tensors constitute the 'bottlenecks' of the provided image_lists. This method computes
    #         and stores the calculated bottlenecks in a bottlenecks dataframe for retrieval at a later time during training
    #         or evaluation.
    #         WARNING: This method currently assumes that all bottlenecks will fit in a single output dataframe. If this is
    #             not the case then this method (and prior logic) needs to be updated.
    #     :param sess: <tf.Session> The current active TensorFlow Session.
    #     :param image_metadata: <dict> Contains information regarding the distribution of images among training, validation,
    #         and testing datasets.
    #         image_metadata['num_train_images']: Number of training images.
    #         image_metadata['num_val_images']: Number of validation images.
    #         image_metadata['num_test_images']: Number of testing images.
    #         image_metadata['num_images']: Total number of sample images.
    #     :param image_lists: <OrderedDict> of training images for each label.\
    #     :param jpeg_data_tensor: <tf.Tensor?> Input tensor for jpeg data from file.
    #     :param decoded_image_tensor: <tf.Tensor?> The tensor holding the output of the image decoding sub-graph.
    #     :param resized_input_tensor: <tf.Tensor?> The input node of the source/recognition graph.
    #     :param bottleneck_tensor: <tf.Tensor?> The penultimate (pre-softmax) output node of the source/recognition graph.
    #     :return bottlenecks: <pd.DataFrame> A dataframe containing the bottleneck values for every tensor in the
    #         datasets.
    #     """
    #     # If the specified bottleneck directory doesn't exist, create it:
    #     if not os.path.exists(os.path.dirname(self.compressed_bottleneck_file_path)):
    #         os.mkdir(os.path.dirname(self.compressed_bottleneck_file_path))
    #     bottlenecks_empty = pd.DataFrame(columns=['class', 'path', 'bottleneck'])
    #     bottlenecks_empty['class'] = bottlenecks_empty['class'].astype('category')
    #     for clss in self.image_lists.keys():
    #         for image_path in self.image_lists[clss]:
    #             new_bottleneck_entry = pd.Series([clss, image_path, None], index=['class', 'path', 'bottleneck'])
    #             bottlenecks_empty = bottlenecks_empty.append(new_bottleneck_entry, ignore_index=True)
    #     bottlenecks = bottlenecks_empty.copy(deep=True)
    #     num_bottlenecks = 0
    #     for i, (clss, series) in enumerate(bottlenecks_empty.iterrows()):
    #         image_path = series['path']
    #         if not tf.gfile.Exists(image_path):
    #             tf.logging.fatal('File does not exist %s', image_path)
    #         image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    #         bottleneck_tensor_values = self._calculate_bottleneck_value(image_path=image_path, image_data=image_data)
    #         bottlenecks.iat[i, 2] = bottleneck_tensor_values
    #         num_bottlenecks += 1
    #         if num_bottlenecks % 100 == 0:
    #             tf.logging.info(msg='Computed %d bottleneck vectors.' % num_bottlenecks)
    #         if num_bottlenecks % 10000 == 0:
    #             tf.logging.info(msg='Computed %d bottleneck vectors. Backing up dataframe to: \'%s\'' % (num_bottlenecks, self.compressed_bottleneck_file_path))
    #             bottlenecks.to_pickle(self.compressed_bottleneck_file_path)
    #
    #     tf.logging.info(msg='Finished computing bottlenecks. Saving dataframe to: \'%s\'' % self.compressed_bottleneck_file_path)
    #     bottlenecks.to_pickle(self.compressed_bottleneck_file_path)


if __name__ == '__main__':
    # Disable double logging output:

    # Debug Configurations:
    # bottleneck_path = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\GoingDeeper\\images\\bottlenecks.pkl'
    # image_path = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\GoingDeeper\\images'
    # logging_path = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\DataAcquisition\\CleaningResults\\DEBUG'

    # BOON Configuration:
    # bottleneck_path = 'D:\\data\\BOON\\bottlenecks.pkl'
    # image_path = 'D:\\data\\BOON\\images\\'
    # logging_path = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\DataAcquisition\\CleaningResults\\BOON'

    # GoingDeeper Configuration:
    bottleneck_path = 'D:\\data\\GoingDeeperData\\bottlenecks.pkl'
    image_path = 'D:\\data\\GoingDeeperData\\images'
    logging_path = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\DataAcquisition\\CleaningResults\\GoingDeeper'

    # SERNEC Cofiguration:
    # bottleneck_path = 'D:\\data\\SERNEC\\bottlenecks.pkl'
    # image_path = 'D:\\data\\SERNEC\\images'
    # logging_path = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\DataAcquisition\\CleaningResults\\SERNEC'


    bottleneck_executor = BottleneckExecutor(
        image_dir=image_path,
        tfhub_module_url='https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1',
        compressed_bottleneck_file_path=bottleneck_path,
        logging_dir=logging_path
    )
    # bottleneck_executor._cache_all_bottlenecks()
    # bottleneck_executor._resume_caching_bottlenecks()
    bottlenecks_df = bottleneck_executor.get_bottlenecks()
    train_bottlenecks, val_bottlenecks, test_bottlenecks = bottleneck_executor.get_partitioned_bottlenecks()
