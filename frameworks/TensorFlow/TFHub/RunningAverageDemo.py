import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
from sklearn import model_selection

tfhub_module_url = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'
tb_log_dir = 'C:\\Users\\ccamp\Documents\\GitHub\\HerbariumDeep\\frameworks\\TensorFlow\\TFHub\\tmp\\summaries'

class RunningAverageDemoClassifier:
    def __init__(self, class_labels, train_batch_size=-1, val_batch_size=-1, initializer=tf.initializers.truncated_normal, activation=tf.nn.elu, optimizer=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)):
        self.class_labels = class_labels
        self.num_classes = len(class_labels)

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self.initializer = initializer
        self.activation = activation
        self.optimizer = optimizer

        self._graph = None          # TensorFlow computational graph
        self._module_spec = None    # TFHub module spec


    @staticmethod
    def create_module_graph(graph, module_spec):
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
        with graph.as_default() as source_model_graph:
            with source_model_graph.name_scope('source_model'):
                # Create a placeholder tensor for input to the model.
                resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3], name='resized_input')
                with tf.variable_scope('pre_trained_hub_module'):
                    # Declare the model in accordance with the chosen architecture:
                    m = hub.Module(module_spec, name='inception_v3_hub')
                    # Create another place holder tensor to catch the output of the pre-activation layer:
                    bottleneck_tensor = m(resized_input_tensor)
        augmented_graph = source_model_graph
        return augmented_graph, bottleneck_tensor, resized_input_tensor

    def _build_graph(self):
        self._graph = tf.Graph()
        self._module_spec = hub.load_module_spec(tfhub_module_url)
        tf.logging.info(msg='Loaded module_spec: %s' % self._module_spec)

        augmented_graph, self._bottleneck_tensor, self._resized_input_tensor = \
            RunningAverageDemoClassifier.create_module_graph(graph=self._graph, module_spec=self._module_spec)

        # Add transfer learning re-train Ops to training graph:

        with augmented_graph.as_default() as train_graph:

            # module_spec_tag_set = self._module_spec.get_tags()[1]
            # tf.logging.info(msg='TensorFlow-Hub ModuleSpec: self._module_spec\'s tags: %s' % module_spec_tag_set)
            # module_spec_tag_set_signature_names = self._module_spec.get_signature_names(module_spec_tag_set)
            # tf.logging.info(msg='TensorFlow-Hub ModuleSpec: self._module_spec\'s signature names for tag set \'%s\': %s' % (module_spec_tag_set, module_spec_tag_set_signature_names))
            # module_spec_tag_set_sig_def_input_info_dict = self._module_spec.get_input_info_dict(signature=module_spec_tag_set_signature_names[1], tags=module_spec_tag_set)
            # tf.logging.info(msg='TensorFlow-Hub ModuleSpec: self._module_spec\'s input info dict for signature \'%s\' and tags %s is: %s' % (module_spec_tag_set_signature_names[1], module_spec_tag_set, module_spec_tag_set_sig_def_input_info_dict))
            height, width = hub.get_expected_image_size(self._module_spec)
            with train_graph.name_scope('source_model'):
                resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3], name='resized_input_tensor')
                m = hub.Module(self._module_spec, trainable=True)
                bottleneck_tensor = m(resized_input_tensor)

            batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
            with train_graph.name_scope('retrain_ops'):
                with train_graph.name_scope('input'):
                    # Create a placeholder Tensor of same type as bottleneck_tensor to cache output from TFHub module:
                    X = tf.placeholder_with_default(
                        bottleneck_tensor,
                        shape=[batch_size, bottleneck_tensor_size],
                        name='X'
                    )
                    # Another placeholder Tensor to hold the true class labels
                    y = tf.placeholder(
                        tf.int64,
                        shape=[batch_size],
                        name='y'
                    )
                    predictions = tf.placeholder(
                        tf.int64,
                        shape=[batch_size],
                        name='predictions'
                    )
                with train_graph.name_scope('final_retrain_ops'):
                    # The final layer of target domain re-train operations is composed of the following:
                    logits = tf.layers.dense(X, self.num_classes, activation=self.activation, use_bias=True, kernel_initializer=self.initializer, trainable=True, name='logits')
                    # tf.summary.histogram('logits', logits)
                    # This is the tensor that will hold the predictions of the fine-tuned (re-trained) model:
                    y_proba = tf.nn.softmax(logits=logits, name='y_proba')

            with train_graph.name_scope('eval_ops'):
                batch_preds = tf.math.argmax(y_proba, axis=1)

                batch_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
                tf.summary.histogram('batch_xentropy', batch_xentropy)

                batch_loss = tf.reduce_mean(batch_xentropy, name='batch_loss')
                tf.summary.scalar('batch_loss', batch_loss)

                loss_exponential_moving_average = tf.train.ExponentialMovingAverage(decay=0.9)
                # tf.summary.scalar('loss_exponential_moving_average', loss_exponential_moving_average)

                minimization_op = self.optimizer.minimize(batch_loss)

                with tf.control_dependencies([minimization_op]):
                    training_op = loss_exponential_moving_average.apply([batch_loss])

            # trainable_vars = tf.trainable_variables()
            init = tf.global_variables_initializer()

            merged_summaries = tf.summary.merge_all()

        self._batch_preds = batch_preds
        self._batch_loss = batch_loss
        self._graph_global_init = init
        self._X = X
        self._y = y
        self._training_op = training_op
        self._merged_summaries = merged_summaries

    @staticmethod
    def _shuffle_batch(X, y, batch_size):
        """
        _shuffle_batch: Yields an iterable of batch tuples that has been permuted and shuffled so that sampling without
            replacement occurs and each sample is included in at least one batch.
        :source HandsOnML: https://github.com/ageron/handson-ml/blob/master/13_convolutional_neural_networks.ipynb
        :param X: The training data.
        :param y: The target data.
        :param batch_size: The desired batch size. The last array will have len(array) <= batch_size, all others will
            have len(batch_size) length.
        :return:
        """
        # If the batch size is set via shorthand to -1, infer the batch size from dim(0) of X:
        if batch_size == -1:
            batch_size = len(X)
        # Yield batch tuples until exhausted:
        rnd_idx = np.random.permutation(len(X))
        n_batches = len(X) // batch_size
        for batch_idx in np.array_split(rnd_idx, n_batches):
            X_batch, y_batch = X[batch_idx], y[batch_idx]
            yield X_batch, y_batch

    def fit(self, X, y, X_valid=None, y_valid=None, num_epochs=100):

        if self.train_batch_size == -1:
            self.train_batch_size = len(X)
        if self.val_batch_size == -1:
            if X_valid is not None:
                self.val_batch_size = len(X_valid)
            else:
                self.val_batch_size = None

        if self._graph is None:
            self._build_graph()

        with tf.Session(graph=self._graph) as sess:
            self._train_writer = tf.summary.FileWriter(tb_log_dir, sess.graph)
            self._graph_global_init.run()
            # trainable_vars_with_weights = [trainable_var for trainable_var in trainable_vars if 'weights' in trainable_var.name]
            # tf.logging.info(msg='trainable_vars[0]: %s' % trainable_vars[0])
            # tf.logging.info(msg='trainable_vars[0] initial weights: %s' % trainable_vars[0].eval(sess))
            # tf.logging.info(msg='trainable_vars: %s' % trainable_vars)
            for i in range(num_epochs):
                for X_batch, y_batch in self._shuffle_batch(X, y, batch_size=self.train_batch_size):
                    # Run a training step, have to capture results at the mini-batch level:
                    # batch_train_summary, batch_train_preds = sess.run([self._merged_summaries, self._batch_preds, self._training_op], feed_dict={self._X: X_batch, self._y: y_batch})
                    _ = sess.run(self._training_op, feed_dict={self._X: X_batch, self._y: y_batch})
                    # self._train_writer.add_summary(batch_train_summary)

def _get_class_labels(bottlenecks):
    """
    _get_class_labels: Obtains a list of unique class labels contained in the bottlenecks dataframe for use in one-hot
        encoding.
    :param bottlenecks: The bottlenecks dataframe.
    :return class_labels: <list> An array of unique class labels whose indices can be used to one-hot encode target
        labels.
    """
    class_labels = set()
    for unique_class in bottlenecks['class'].unique():
        class_labels.add(unique_class)
    # Convert back to list for one-hot encoding using array indices:
    class_labels = list(class_labels)
    return class_labels

def _load_bottlenecks(compressed_bottleneck_file_path):
    bottlenecks = None
    bottleneck_path = compressed_bottleneck_file_path
    if os.path.isfile(bottleneck_path):
        # Bottlenecks .pkl file exists, read from disk:
        tf.logging.info(msg='Bottleneck file successfully located at the provided path: \'%s\'' % bottleneck_path)
        try:
            bottlenecks = pd.read_pickle(bottleneck_path)
            tf.logging.info(msg='Bottleneck file \'%s\' successfully restored from disk.'
                                % os.path.basename(bottleneck_path))
        except Exception as err:
            tf.logging.error(msg=err)
            bottlenecks = None
            exit(-1)
    else:
        tf.logging.error(msg='Bottleneck file not located at the provided path: \'%s\'. '
                             'Have you run BottleneckExecutor.py?' % bottleneck_path)
        exit(-1)
    return bottlenecks

def _partition_bottlenecks_dataframe(bottlenecks, train_percent=.80, val_percent=.20, test_percent=.20, random_state=0):
    """
    _partition_bottlenecks_dataframe: Partitions the bottlenecks dataframe into training, testing, and validation
        dataframes.
    :param bottlenecks: <pd.DataFrame> The bottlenecks dataframe containing image-labels, paths, and bottleneck values.
    :param train_percent: What percentage of the training data is to remain in the training set.
    :param test_percent: What percentage of the training data is to be allocated to a testing set.
    :param val_percent: What percentage of the remaining training data (after removing test set) is to be allocated
        for a validation set.
    :param random_state: A seed for the random number generator controlling the stratified partitioning.
    :return:
    """
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

def _get_all_cached_bottlenecks(bottleneck_dataframe, class_labels):
    """
    _get_all_cached_bottlenecks: Returns the bottleneck values from the dataframe and performs one-hot encoding on the
        class labels.
    :param bottleneck_dataframe: One of the partitioned bottleneck dataframes [train, val, test].
    :param class_labels: A list of unique class labels to be used consistently for one-hot encoding of target labels.
    :returns bottleneck_values, bottleneck_ground_truth_indices:
        :return bottleneck_values: The bottleneck values from the bottleneck dataframe associated with the returned
            ground truth indices (representing one-hot encoded class labels).
        :return bottleneck_ground_truth_indices: The ground truth indices corresponding to the returned bottleneck
            values for use in classification and evaluation.
    """
    bottleneck_values = bottleneck_dataframe['bottleneck'].tolist()
    bottleneck_values = np.array(bottleneck_values)
    bottleneck_ground_truth_labels = bottleneck_dataframe['class'].values
    # Convert the labels into indices (one hot encoding by index):
    bottleneck_ground_truth_indices = np.array([class_labels.index(ground_truth_label)
                                                for ground_truth_label in bottleneck_ground_truth_labels])
    return bottleneck_values, bottleneck_ground_truth_indices

def main(run_config):

    bottlenecks = _load_bottlenecks(run_config['bottleneck_path'])

    class_labels = _get_class_labels(bottlenecks)

    if run_config['dataset'] != 'SERNEC':
        train_bottlenecks, val_bottlenecks, test_bottlenecks = _partition_bottlenecks_dataframe(bottlenecks=bottlenecks, random_state=0)
    else:
        raise NotImplementedError
    bottleneck_dataframes = {'train': train_bottlenecks, 'val': val_bottlenecks, 'test': test_bottlenecks}
    tf.logging.info(
        'Partitioned (N=%d) total bottleneck vectors into training (N=%d), validation (N=%d), and testing (N=%d) datasets.'
        % (bottlenecks.shape[0], train_bottlenecks.shape[0], val_bottlenecks.shape[0], test_bottlenecks.shape[0])
    )

    train_bottlenecks, train_ground_truth_indices = _get_all_cached_bottlenecks(
        bottleneck_dataframe=bottleneck_dataframes['train'],
        class_labels=class_labels
    )

    val_bottlenecks, val_ground_truth_indices = _get_all_cached_bottlenecks(
        bottleneck_dataframe=bottleneck_dataframes['val'],
        class_labels=class_labels
    )
    tf.logging.info(msg='Obtained bottleneck values from dataframe. Performed corresponding encoding of class labels')

    num_epochs = 100
    running_avg_classifier = RunningAverageDemoClassifier(class_labels=class_labels)
    running_avg_classifier.fit(X=train_bottlenecks, y=train_ground_truth_indices, num_epochs=num_epochs)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    run_configs = {
        'DEBUG': {
            'image_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\GoingDeeper\\images',
            'bottleneck_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\GoingDeeper\\images\\bottlenecks.pkl',
            'dataset': 'DEBUG'
        },
        'BOON': {
            'image_dir': 'D:\\data\\BOON\\images',
            'bottleneck_path': 'D:\\data\\BOON\\bottlenecks.pkl',
            'dataset': 'BOON'
        },
        'GoingDeeper': {
            'image_dir': 'D:\\data\\GoingDeeperData\\images',
            'bottleneck_path': 'D:\\data\\GoingDeeperData\\bottlenecks.pkl',
            'dataset': 'GoingDeeper'
        },
        'SERNEC': {
            'image_dir': 'D:\\data\\SERNEC\\images',
            'bottleneck_path': 'D:\\data\\SERNEC\\bottlenecks.pkl',
            'dataset': 'SERNEC'
        }
    }
    main(run_configs['DEBUG'])

