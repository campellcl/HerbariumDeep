"""
GoingDeeper.py
Implementation of Pipelined TensorFlow Transfer Learning.
"""
import os
import sys
import argparse
import tensorflow as tf
import collections
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import ParameterGrid
import time
from frameworks.TensorFlow.TFHub.TFHClassifier import TFHClassifier
import pandas as pd
import numpy as np

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def _prepare_tensor_board_directories(tb_summaries_dir, intermediate_output_graphs_dir=None):
    """
    _prepare_tensor_board_directories: Ensures that if a TensorBoard storage directory is defined in the command line
        flags, that said directory is purged of old TensorBoard files, and that this program has sufficient permissions
        to write new TensorBoard summaries to the specified path.
    :return None: see above ^
    """
    # Check to see if the file exists:
    if tf.gfile.Exists(tb_summaries_dir):
        # Delete everything in the file recursively:
        tf.gfile.DeleteRecursively(tb_summaries_dir)
    # Re-create (or create for the first time) the storage directory:
    tf.gfile.MakeDirs(tb_summaries_dir)
    # Check to see if intermediate computational graphs are to be stored:
    if intermediate_output_graphs_dir:
        if not os.path.exists(intermediate_output_graphs_dir):
            os.makedirs(intermediate_output_graphs_dir)
    return


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


def _get_random_cached_bottlenecks(bottleneck_dataframes, how_many, category, class_labels):
    """
    get_random_cached_bottlenecks: Retrieve a random sample of rows from the bottlenecks dataframe of size 'how_many'.
        Performs random sampling with replacement.
    :param bottlenecks: The dataframe containing pre-computed bottleneck values.
    :param how_many: The number of bottleneck samples to retrieve.
    :param category: Which subset of dataframes to partition.
    :param class_labels: <list> A list of all unique class labels in the training and testing datasets.
    :returns bottleneck_values, bottleneck_ground_truth_labels:
        :return bottleneck_values: <list> A Python array of size 'how_many' by 2048 (the size of the penultimate output
            layer).
        :return bottleneck_ground_truth_indices: <list> A Python list of size 'how_many' by one, containing the index
            into the class_labels array that corresponds with the ground truth label name associated with each
            bottlneck array.
    """
    bottleneck_dataframe = bottleneck_dataframes[category]
    if how_many >= 0:
        random_mini_batch_indices = np.random.randint(low=0, high=bottleneck_dataframe.shape[0], size=(how_many, ))
        minibatch_samples = bottleneck_dataframe.iloc[random_mini_batch_indices]
        bottleneck_values = minibatch_samples['bottleneck'].tolist()
        bottleneck_values = np.array(bottleneck_values)
        bottleneck_ground_truth_labels = minibatch_samples['class'].values

    else:
        bottleneck_values = bottleneck_dataframe['bottleneck'].tolist()
        bottleneck_values = np.array(bottleneck_values)
        bottleneck_ground_truth_labels = bottleneck_dataframe['class'].values

    # Convert to index (encoded int class label):
    bottleneck_ground_truth_indices = np.array([class_labels.index(ground_truth_label)
                                       for ground_truth_label in bottleneck_ground_truth_labels])
    return bottleneck_values, bottleneck_ground_truth_indices


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


def get_initializer_options():
    """
    get_initializer_options: Returns a dictionary of weight initialization methods for use with SKLearn's GridSearch.
        For details see: https://www.tensorflow.org/api_docs/python/tf/initializers
    :return initializer_options: <dict> A dictionary of valid weight initialization methods.
    """
    # he_normal = tf.initializers.he_normal
    initializer_options = {
        'he_normal': tf.variance_scaling_initializer(),
        'he_uniform': tf.initializers.he_uniform,
        'truncated_normal': tf.truncated_normal,
        'random_normal_dist': tf.random_normal,
        'uniform_normal_dist': tf.random_uniform
    }
    return initializer_options


def get_activation_options(leaky_relu_alpha=None):
    """
    get_activation_options: Returns a dictionary of activation methods for use with SKLearn's GridSearch. For details
        see: https://www.tensorflow.org/api_docs/python/tf/nn
    :return activation_options: <dict> A dictionary of valid neural network activation functions.
    """
    activation_options = {
        'ELU': tf.nn.elu
    }
    if leaky_relu_alpha:
        activation_options['LeakyReLU'] = tf.nn.leaky_relu
    return activation_options


def get_optimizer_options(static_learning_rate, momentum_const=None, adam_beta1=None, adam_beta2=None, adam_epsilon=None, adagrad_init_accum=None, adadelta_rho=None, adadelta_epsilon=None):
    """
    get_optimizer_options: Returns a dictionary of optimizer methods for use with SKLearn's GridSearch. Ensures that the
        method invoker supplies all arguments for the desired optimizer class (no defaults allowed). For details
        see: https://www.tensorflow.org/api_docs/python/tf/train
    :return:
    """
    optimizer_options = {
        'GradientDescent': tf.train.GradientDescentOptimizer(
            learning_rate=static_learning_rate
        )
    }
    if momentum_const:
        optimizer_options['Momentum'] = tf.train.MomentumOptimizer(
            learning_rate=static_learning_rate,
            momentum=momentum_const,
            use_nesterov=False
        )
        optimizer_options['Nesterov'] = tf.train.MomentumOptimizer(
            learning_rate=static_learning_rate,
            momentum=momentum_const,
            use_nesterov=True
        )
    if adam_epsilon and adam_beta1 and adam_beta2:
        optimizer_options['Adam'] = tf.train.AdamOptimizer(
            learning_rate=static_learning_rate,
            beta1=adam_beta1,
            beta2=adam_beta2,
            epsilon=adam_epsilon
        )
    if adagrad_init_accum:
        optimizer_options['AdaGrad'] = tf.train.AdagradOptimizer(
            learning_rate=static_learning_rate,
            initial_accumulator_value=0.1
        )
    if adadelta_rho and adadelta_epsilon:
        optimizer_options['AdaDelta'] = tf.train.AdadeltaOptimizer(
            learning_rate=static_learning_rate,
            rho=adadelta_rho,
            epsilon=adadelta_epsilon
        )
    # momentum_low = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.1)
    # momentum_high = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    # nesterov_low = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.1, use_nesterov=True)
    # nesterov_high = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
    # adagrad = tf.train.AdagradOptimizer(learning_rate=learning_rate, name='Adagrad')
    # adadelta = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=0.95, epsilon=1e-08, name='Adadelta')
    return optimizer_options


def _run_grid_search(train_bottlenecks, train_ground_truth_indices, initializers, activations, optimizers, class_labels, val_bottlenecks=None, val_ground_truth_indices=None):
    # params = {
    #     'initializer': [random_normal_dist, uniform_normal_dist, truncated_normal, he_normal, he_uniform],
    #     'optimizer_class': [gradient_descent, adam, momentum_low, momentum_high]
    # }
    # params = {
    #     'initializer': [he_normal],
    #     'activation': [elu],
    #     'optimizer': [nesterov_high],
    #     'learning_rate': [learning_rate]
    # }

    # params = {
    #     'initializer': list(initializers.values()),
    #     'activation': list(activations.values()),
    #     'optimizer': list(optimizers.values())
    # }

    # params = {
    #     'initializer': [initializers['he_normal']],
    #     'activation': [activations['LeakyReLU']],
    #     'optimizer': [optimizers['Adam']]
    #     # 'train_batch_size': [1000]
    # }

    # params = {
    #     'initializer': [initializers['he_normal']],
    #     'activation': [activations['LeakyReLU']],
    #     'optimizer': [optimizers['Nesterov']],
    #     'train_batch_size': [20, 60, 100, -1]
    # }

    params = {
        'initializer': [initializers['he_normal']],
        'fixed_feature_extractor': [False],
        'activation': [activations['LeakyReLU']],
        'optimizer': [optimizers['Nesterov']],
        'train_batch_size': [20, 60]
    }

    tf.logging.info(msg='Initialized SKLearn parameter grid: %s' % params)
    tfh_classifier = TFHClassifier(random_state=42, class_labels=class_labels)
    tf.logging.info(msg='Initialized TensorFlowHub Classifier (TFHClassifier)')
    # This looks odd, but drops the CV from GridSearchCV. See: https://stackoverflow.com/a/44682305/3429090
    cv = [(slice(None), slice(None))]
    grid_search = GridSearchCV(tfh_classifier, params, cv=cv, verbose=2, refit=False)
    tf.logging.info(msg='Running GridSearch...')
    grid_search.fit(
        X=train_bottlenecks,
        y=train_ground_truth_indices,
        X_valid=val_bottlenecks,
        y_valid=val_ground_truth_indices,
        n_epochs=10000,
        ckpt_freq=0
    )
    tf.logging.info(msg='Finished GridSearch! Restoring best performing parameter set...')
    best_params = grid_search.best_params_
    # This is a refit operation, notify TensorBoard to replace the previous run's logging data:
    best_params['refit'] = True
    # Replace the current model parameters with the best combination from the GridSearch:
    current_params = tfh_classifier.get_params()
    current_params.update(best_params)
    tfh_classifier.set_params(**current_params)
    tf.logging.info(msg='Model hyperparmaters have been set to the highest scoring settings reported by GridSearch. '
                        'Now fitting a classifier with these hyperparameters...')
    # Re-fit the model using the best parameter combination from the GridSearch:
    tfh_classifier.fit(
        X=train_bottlenecks,
        y=train_ground_truth_indices,
        X_valid=val_bottlenecks,
        y_valid=val_ground_truth_indices,
        n_epochs=10000,
        ckpt_freq=0
    )
    tf.logging.info(msg='Classifier re-fit! Model ready for inference.')
    y_pred = tfh_classifier.predict(X=val_bottlenecks)
    print('Classifier accuracy_score: %.2f' % accuracy_score(val_ground_truth_indices, y_pred))


def _run_setup(bottleneck_path, tb_summaries_dir):
    """
    _run_setup: Performs initial setup operations by:
        1) Setting verbosity of TensorFlow logging output
        2) Purging (or creating new) TensorBoard logging directories
        3) Retrieving the bottlenecks dataframe from disk
        4) Partitioning the bottlenecks dataframe into training and testing sets.
    :param bottleneck_path: The file path to the compressed bottlenecks dataframe.
    :return:
    """
    # Enable visible logging output:
    tf.logging.set_verbosity(tf.logging.INFO)

    # Delete any TensorBoard summaries left over from previous runs:
    _prepare_tensor_board_directories(tb_summaries_dir=tb_summaries_dir)
    tf.logging.info(msg='Removed left over TensorBoard summaries from previous runs.')

    # Retrieve the bottlenecks dataframe or alert the user and terminate:
    bottlenecks = _load_bottlenecks(bottleneck_path)

    # Get a list of unique class labels for use in one-hot encoding:
    class_labels = _get_class_labels(bottlenecks)

    # Partition the bottlenecks dataframe:
    train_bottlenecks, val_bottlenecks, test_bottlenecks = _partition_bottlenecks_dataframe(
        bottlenecks=bottlenecks,
        random_state=0
    )
    bottleneck_dataframes = {'train': train_bottlenecks, 'val': val_bottlenecks, 'test': test_bottlenecks}
    tf.logging.info(
        'Partitioned (N=%d) total bottleneck vectors into training (N=%d), validation (N=%d), and testing (N=%d) datasets.'
        % (bottlenecks.shape[0], train_bottlenecks.shape[0], val_bottlenecks.shape[0], test_bottlenecks.shape[0])
    )
    return bottleneck_dataframes, class_labels


def main(run_config):
    """
    main:
    :return:
    """
    """
    TensorBoard summaries directory:
    """
    summaries_dir = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\TensorFlow\\TFHub\\tmp\\summaries'

    # Run preliminary setup operations and retrieve partitioned bottlenecks dataframe:
    bottleneck_dataframes, class_labels = _run_setup(bottleneck_path=run_config['bottleneck_path'], tb_summaries_dir=summaries_dir)
    tf.logging.info('Detected %d unique class labels in the bottlenecks dataframe' % len(class_labels))

    train_bottlenecks, train_ground_truth_indices = _get_all_cached_bottlenecks(
        bottleneck_dataframe=bottleneck_dataframes['train'],
        class_labels=class_labels
    )

    val_bottlenecks, val_ground_truth_indices = _get_all_cached_bottlenecks(
        bottleneck_dataframe=bottleneck_dataframes['val'],
        class_labels=class_labels
    )

    tf.logging.info(msg='Obtained bottleneck values from dataframe. Performed corresponding one-hot encoding of class labels')
    initializer_options = get_initializer_options()
    activation_options = get_activation_options(leaky_relu_alpha=0.2)
    # User MUST provide all default arguments or a KeyError will be thrown:
    optimizer_options = get_optimizer_options(
        static_learning_rate=0.001,
        momentum_const=0.9,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08
    )
    _run_grid_search(
        train_bottlenecks=train_bottlenecks,
        train_ground_truth_indices=train_ground_truth_indices,
        initializers=initializer_options,
        activations=activation_options,
        optimizers=optimizer_options,
        class_labels=class_labels,
        val_bottlenecks=val_bottlenecks,
        val_ground_truth_indices=val_ground_truth_indices
    )

    # minibatch_train_bottlenecks, minibatch_train_ground_truth_indices = _get_random_cached_bottlenecks(
    #     bottleneck_dataframes=bottleneck_dataframes,
    #     how_many=train_batch_size,
    #     category='train',
    #     class_labels=class_labels
    # )
    # minibatch_val_bottlenecks, minibatch_val_ground_truth_indices = _get_random_cached_bottlenecks(
    #     bottleneck_dataframes=bottleneck_dataframes,
    #     how_many=val_batch_size,
    #     category='val',
    #     class_labels=class_labels
    # )
    # tf.logging.info(msg='Partitioned bottleneck dataframe into train, val, test splits.')
    # ON RESUME: Need to test checkpoint save and restore during training since it will take long time. Code previously
    #     written in one of these branches or for networking. Then need to check saved model export and load for inference.


if __name__ == '__main__':
    run_configs = {
        'debug': {
            'image_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\GoingDeeper\\images',
            'bottleneck_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\TensorFlow\\TFHub\\bottlenecks.pkl'
        },
        'BOON': {
            'image_dir': 'D:\\data\\BOON\\images',
            'bottleneck_path': 'D:\\data\\BOON\\bottlenecks.pkl'
        },
        'GoingDeeper': {
            'image_dir': 'D:\\data\\GoingDeeperData\\images',
            'bottleneck_path': 'D:\\data\\GoingDeeperData\\bottlenecks.pkl'
        },
        'SERNEC': {
            'image_dir': 'D:\\data\\SERNEC\\images',
            'bottleneck_path': 'D:\\data\\SERNEC\\bottlenecks.pkl'
        }
    }
    main(run_configs['BOON'])
    '''
    Execute this script under a shell instead of importing as a module. Ensures that the main function is called with
    the proper command line arguments (builds on default argparse). For more information see:
    https://stackoverflow.com/questions/33703624/how-does-tf-app-run-work
    '''
    # tf.app.run(main=main)
