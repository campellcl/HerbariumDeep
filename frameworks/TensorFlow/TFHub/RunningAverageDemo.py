import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from sklearn import model_selection
import os


# def main():
#     tfhub_module_url = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'
#     _module_spec = hub.load_module_spec(tfhub_module_url)
#     tf.logging.info(msg='Loaded module_spec: %s' % _module_spec)
#
#     _graph = tf.Graph()
#
#     with _graph.as_default() as source_model_graph:
#         height, width = hub.get_expected_image_size(_module_spec)
#         resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3], name='resized_input')
#         m = hub.Module(_module_spec, name='inception_v3_hub')
#         bottleneck_tensor = m(resized_input_tensor)
#     _graph = source_model_graph
#     with _graph.as_default() as further_augmented_graph:
#         batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
#         assert batch_size is None, 'We want to work with arbitrary batch size when ' \
#                                'constructing fully-connected and softmax layers for fine-tuning.'
#         with tf.name_scope('retrain_ops'):
#             with tf.name_scope('input'):
#                 # Create a placeholder Tensor of same type as bottleneck_tensor to cache output from TFHub module:
#                 X = tf.placeholder_with_default(
#                     bottleneck_tensor,
#                     shape=[batch_size, bottleneck_tensor_size],
#                     name='X'
#                 )
#                 # Another placeholder Tensor to hold the true class labels
#                 y = tf.placeholder(
#                     tf.int64,
#                     shape=[batch_size],
#                     name='y'
#                 )
#                 predictions = tf.placeholder(
#                     tf.int64,
#                     shape=[batch_size],
#                     name='predictions'
#                 )
#         with tf.name_scope('final_retrain_ops'):
#             # The final layer of target domain re-train operations is composed of the following:
#             logits = tf.layers.dense(X, num_classes, activation=activation, use_bias=True, kernel_initializer=initializer, trainable=True, name='logits')
#             # tf.summary.histogram('logits', logits)
#             # This is the tensor that will hold the predictions of the fine-tuned (re-trained) model:
#             y_proba = tf.nn.softmax(logits=logits, name=final_tensor_name)

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


def main(run_config):
    train_batch_size = 10

    bottlenecks = _load_bottlenecks(run_config['bottleneck_path'])

    class_labels = _get_class_labels(bottlenecks)
    num_classes = len(class_labels)

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

    img = tf.random_uniform([10, 299, 299, 3])
    tfhub_module_url = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'
    _module_spec = hub.load_module_spec(tfhub_module_url)
    tf.logging.info(msg='Loaded module_spec: %s' % _module_spec)

    batch_losses = []



    # Add to collection;
    # tf.add_to_collection(name=tf.GraphKeys.MOVING_AVERAGE_VARIABLES, value=loss_ema)
    # loss_vars = tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)

    _graph = tf.Graph()
    _session = tf.Session(graph=_graph)

    with _graph.as_default() as source_model_graph:
        height, width = hub.get_expected_image_size(_module_spec)
        resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3], name='resized_input')
        m = hub.Module(_module_spec, name='inception_v3_hub')
        bottleneck_tensor = m(resized_input_tensor)
        batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()

        # batch_losses = tf.Variable(tf.int64)

        batch_index = tf.placeholder(
            tf.int32,
            shape=[]
        )
        X = tf.placeholder_with_default(
            bottleneck_tensor,
            shape=[batch_size, bottleneck_tensor_size],
            name='X'
        )
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
        logits = tf.layers.dense(X, num_classes, tf.nn.elu, use_bias=True, kernel_initializer=tf.initializers.truncated_normal, trainable=True, name='logits')
        y_proba = tf.nn.softmax(logits=logits, name='y_proba')
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name='loss')
        minimization_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(loss)
        # batch_loss = tf.placeholder(
        #     tf.int64,
        #     shape=(),
        #     name='batch_loss'
        # )
        batch_loss_sum = tf.Variable(0.0, dtype=tf.float32, name='batch_loss_sum_var')
        batch_loss_moving_average_var = batch_loss_sum / tf.cast(batch_index, tf.float32)
        clear_batch_loss_moving_average_op = tf.assign(batch_loss_sum, 0.0)
        with tf.control_dependencies([minimization_op]):
            increment_batch_loss_op = batch_loss_sum.assign_add(loss)
            training_op = tf.group(increment_batch_loss_op)

        # batch_loss_moving_average, batch_loss_moving_average_update_op = tf.metrics.mean(
        #     batch_loss_moving_average_var,
        #     metrics_collections=[tf.GraphKeys.METRIC_VARIABLES, tf.GraphKeys.MOVING_AVERAGE_VARIABLES],
        #     updates_collections=[tf.GraphKeys.UPDATE_OPS]
        # )

        graph_global_init = tf.global_variables_initializer()

    session = tf.Session(graph=_graph)
    with session.as_default() as sess:
        graph_global_init.run()
        for epoch in range(num_epochs):
            # sess.run(batch_loss_moving_average_update_op.initializer)
            for batch_num, (X_batch, y_batch) in enumerate(_shuffle_batch(train_bottlenecks, train_ground_truth_indices, batch_size=train_batch_size)):
                # Now any invocation of training_op will force an update of the moving exponential average:
                _ = sess.run([training_op], feed_dict={X: X_batch, y: y_batch, batch_index: batch_num})
            # To actually compute the average of the acquired shadow variables, we run this op:
            batch_loss_avg = sess.run(batch_loss_moving_average_var, feed_dict={batch_index: batch_num})
            print('\t%d\tbatch_loss_moving_average: %.2f' % (epoch, batch_loss_avg))
            sess.run(clear_batch_loss_moving_average_op)
            # moving_average_value = sess.run(get_moving_average_op)
            # To actually get a value (as opposed to a Tensor) we need to run the Variable that stores the result within session context:
            # print('\t%d\tloss_ema: %.2f' % (epoch, sess.run(moving_average)))


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
    tb_log_dir = 'C:\\Users\\ccamp\Documents\\GitHub\\HerbariumDeep\\frameworks\\TensorFlow\\TFHub\\tmp\\summaries'
    if tf.gfile.Exists(tb_log_dir):
        tf.gfile.DeleteRecursively(tb_log_dir)
    tf.gfile.MakeDirs(tb_log_dir)
    main(run_configs['DEBUG'])
