"""
GoingDeeper.py
Implementation of Pipelined TensorFlow Transfer Learning.
"""
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
import argparse
import tensorflow as tf
import collections
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, ShuffleSplit
# from sklearn.model_selection import ParameterGrid
import time
from frameworks.TensorFlow.TFHub.TFHClassifier import TFHClassifier
from frameworks.DataAcquisition.BottleneckExecutor import BottleneckExecutor
from frameworks.Sklearn.GridSearchCVSaveRestore import GridSearchCVSaveRestore
import pandas as pd
import numpy as np

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


class CrossValidationSplitter(ShuffleSplit):

    def __init__(self, n_splits, test_size=None, train_size=None, random_state=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        yield([i for i in range(self.train_size)], [j for j in range(self.train_size, self.train_size + self.test_size)])


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
    # Re-create (or create for the first time) the tensorboard root storage directory:
    tf.gfile.MakeDirs(tb_summaries_dir)
    # Check to see if intermediate computational graphs are to be stored:
    if intermediate_output_graphs_dir:
        if not os.path.exists(intermediate_output_graphs_dir):
            os.makedirs(intermediate_output_graphs_dir)
    # Create tensorboard grid search subfolders:
    grid_search_root_dir = os.path.join(tb_summaries_dir, 'gs')
    tf.gfile.MakeDirs(grid_search_root_dir)

    grid_search_winner_root_dir = os.path.join(tb_summaries_dir, 'gs_winner')
    tf.gfile.MakeDirs(grid_search_winner_root_dir)

    grid_search_train_dir = os.path.join(grid_search_root_dir, 'train')
    tf.gfile.MakeDirs(grid_search_train_dir)

    grid_search_winner_train_dir = os.path.join(grid_search_winner_root_dir, 'train')
    tf.gfile.MakeDirs(grid_search_winner_train_dir)

    grid_search_val_dir = os.path.join(grid_search_root_dir, 'val')
    tf.gfile.MakeDirs(grid_search_val_dir)

    grid_search_winner_val_dir = os.path.join(grid_search_winner_root_dir, 'val')
    tf.gfile.MakeDirs(grid_search_winner_val_dir)

    return


def _prepare_model_export_directories(model_export_dir):
    if os.path.exists(model_export_dir):
        tf.gfile.DeleteRecursively(model_export_dir)
    return


def _clear_temp_folder(temp_logdir):
    if os.path.exists(temp_logdir):
        tf.gfile.DeleteRecursively(temp_logdir)
    os.mkdir(temp_logdir)
    return


def _is_bottleneck_for_every_sample(image_lists, bottlenecks):
    train_image_paths = []
    val_image_paths = []
    test_image_paths = []
    for species, datasets in image_lists.items():
        # Training set images:
        species_train_image_paths = datasets['train']
        for species_train_image_path in species_train_image_paths:
            train_image_paths.append(species_train_image_path)
        # Validation set images:
        species_val_image_paths = datasets['val']
        for species_val_image_path in species_val_image_paths:
            val_image_paths.append(species_val_image_path)
        # Testing set images:
        species_test_image_paths = datasets['test']
        for species_test_image_path in species_test_image_paths:
            test_image_paths.append(species_test_image_path)
    # Ensure every training image has a bottleneck entry in the bottlenecks dataframe:
    for train_image_path in train_image_paths:
        if train_image_path not in bottlenecks['path'].values:
            return False
    # Ensure every validation image has a bottleneck tensor in bottlenecks dataframe:
    for val_image_path in val_image_paths:
        if val_image_path not in bottlenecks['path'].values:
            return False
    # Ensure every test image has a bottleneck tensor in the bottlenecks dataframe:
    for test_image_path in test_image_paths:
        if test_image_path not in bottlenecks['path'].values:
            return False
    return True


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


def _partition_image_lists(image_lists, train_percent, val_percent, test_percent, random_state):
    """
    _partition_image_lists: Partitions the provided dict of class labels and file paths into training, validation, and
        testing datasets.
    :param image_lists: <collections.OrderedDict> A dictionary indexed by class label, which provides as its value a
        list of file paths for images belonging to the chosen key/species/class-label.
    :param train_percent: What percentage of the training data is to remain in the training set.
    :param val_percent: What percentage of the remaining training data (after removing test set) is to be allocated
        for a validation set.
    :param test_percent: What percentage of the training data is to be allocated to a testing set.
    :param random_state: A seed for the random number generator controlling the stratified partitioning.
    :return partitioned_image_lists: <collections.OrderedDict> A dictionary indexed by class label which returns another
        dictionary indexed by the dataset type {'train','val','test'} which in turn, returns the list of image file
        paths that correspond to the chosen class label and that reside in the chosen dataset.
    """
    partitioned_image_lists = collections.OrderedDict()
    for class_label, image_paths in image_lists.items():
        class_label_train_images, class_label_test_images = model_selection.train_test_split(
            image_paths, train_size=train_percent,
            test_size=test_percent, shuffle=True,
            random_state=random_state
        )
        class_label_train_images, class_label_val_images = model_selection.train_test_split(
            class_label_train_images, train_size=train_percent,
            test_size=val_percent, shuffle=True,
            random_state=random_state
        )
        partitioned_image_lists[class_label] = {
            'train': class_label_train_images,
            'val': class_label_val_images,
            'test': class_label_test_images
        }
    return partitioned_image_lists


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


def _run_setup(tb_summaries_dir):
    """
    _run_setup: Performs initial setup operations by:
        1) Setting verbosity of TensorFlow logging output
        2) Purging (or creating new) TensorBoard logging directories
    :param bottleneck_path: The file path to the compressed bottlenecks dataframe.
    :return:
    """
    # Enable visible logging output:
    tf.logging.set_verbosity(tf.logging.INFO)

    # Delete any TensorBoard summaries left over from previous runs:
    _prepare_tensor_board_directories(tb_summaries_dir=tb_summaries_dir)
    tf.logging.info(msg='Removed left over tensorboard summaries from previous runs.')
    return


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
        'he_uniform': tf.initializers.he_uniform(),
        'truncated_normal': tf.initializers.truncated_normal,
        'random_normal_dist': tf.initializers.random_normal,
        'uniform_normal_dist': tf.initializers.random_uniform
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


def _run_grid_search(dataset, train_bottlenecks, train_ground_truth_indices, initializers, activations, optimizers,
                     class_labels, log_dir, model_export_dir, val_bottlenecks=None, val_ground_truth_indices=None):
    num_train_samples = train_bottlenecks.shape[0]
    num_val_samples = val_bottlenecks.shape[0]
    """
    Note on Train Batch Sizes:
        16 Comes from the paper: Going Deeper in the Automated Identification of Herbarium Specimens
        20 and 60 come from the paper: Plant Identification Using Deep Neural Networks with Hyperparameter Optimization via Transfer Learning
    """
    if dataset == 'SERNEC':
        params = {
            'initializer': [initializers['he_normal'], initializers['he_uniform'], initializers['truncated_normal']],
            'activation': [activations['LeakyReLU'], activations['ELU']],
            'optimizer': [optimizers['Nesterov'], optimizers['Adam']],
            'train_batch_size': [20, 60, 100, 1000]
        }
        # num_epochs = 100000  # 100,000
        # eval_freq = 100
        # early_stopping_eval_freq = 1
        num_epochs = 100000     # 100,000
        eval_freq = 10
        early_stopping_eval_freq = 5
        ckpt_freq = 0
        tf.logging.info(msg='Initialized SKLearn parameter grid: %s' % params)
    elif dataset == 'GoingDeeper':
        params = {
            'initializer': [initializers['he_normal'], initializers['he_uniform'], initializers['truncated_normal']],
            'activation': [activations['LeakyReLU'], activations['ELU']],
            'optimizer': [optimizers['Nesterov'], optimizers['Adam']],
            'train_batch_size': [20, 60, 100, 1000]
        }
        num_epochs = 100000  # 100,000
        eval_freq = 10
        early_stopping_eval_freq = 5
        ckpt_freq = 0
        ''' Debug Configurations for Grid Search save and restore functionality testing: '''
        # num_epochs = 2
        # eval_freq = 1
        # early_stopping_eval_freq = 1
        # ckpt_freq = 0
        tf.logging.info(msg='Initialized SKLearn parameter grid: %s' % params)
    elif dataset == 'BOON':
        params = {
            'initializer': [initializers['he_normal'], initializers['he_uniform'], initializers['truncated_normal']],
            'activation': [activations['LeakyReLU'], activations['ELU']],
            'optimizer': [optimizers['Nesterov'], optimizers['Adam']],
            'train_batch_size': [20, 60, 100, 1000]
        }
        num_epochs = 100000  # 100,000
        eval_freq = 10
        early_stopping_eval_freq = 5
        ckpt_freq = 0
        tf.logging.info(msg='Initialized SKLearn parameter grid: %s' % params)
    elif dataset == 'DEBUG':
        params = {
            'initializer': [initializers['he_normal'], initializers['he_uniform']],
            'activation': [activations['LeakyReLU']],
            'optimizer': [optimizers['Nesterov'], optimizers['Adam']],
            'train_batch_size': [10, 20]
        }
        num_epochs = 1000
        eval_freq = 10
        early_stopping_eval_freq = 5
        ckpt_freq = 0
        tf.logging.info(msg='Initialized SKLearn parameter grid: %s' % params)
    else:
        tf.logging.error(msg='FATAL ERROR: Could not recognize the provided dataset: \'%s\', exiting.' % dataset)
        params = None
        num_epochs = None
        eval_freq = None
        ckpt_freq = None
        early_stopping_eval_freq = None
        exit(-1)
    tfh_classifier = TFHClassifier(dataset=dataset, random_state=42, class_labels=class_labels, tb_logdir=log_dir)
    tf.logging.info(msg='Initialized TensorFlowHub Classifier (TFHClassifier) Instance')
    # This looks odd, but drops the CV from GridSearchCV. See: https://stackoverflow.com/a/44682305/3429090
    custom_cv_splitter = CrossValidationSplitter(train_size=num_train_samples, test_size=num_val_samples, n_splits=1)
    ''' New Custom Grid Search with Save and Restore Code '''
    grid_search = GridSearchCVSaveRestore(
        estimator=tfh_classifier, param_grid=params, cv_results_save_freq=1, cv=custom_cv_splitter,
        verbose=2, refit=False, return_train_score=False, error_score='raise', scoring=None
    )
    tf.logging.info(msg='Instantiated GridSearch.')
    X = np.concatenate((train_bottlenecks, val_bottlenecks))
    y = np.concatenate((train_ground_truth_indices, val_ground_truth_indices))
    grid_search.fit(
        X=X,
        y=y,
        X_valid=val_bottlenecks,
        y_valid=val_ground_truth_indices,
        n_epochs=num_epochs,
        eval_freq=eval_freq,
        ckpt_freq=ckpt_freq,
        early_stopping_eval_freq=early_stopping_eval_freq
    )
    ''' Legacy Sklearn as driver code: '''
    # grid_search = GridSearchCV(tfh_classifier, params, cv=custom_cv_splitter, verbose=2, refit=False, return_train_score=False)
    # tf.logging.info(msg='Running GridSearch...')
    # NOTE: This looks counter intuitive, but the custom_cv_splitter will separate these back out when called by GS:
    # X = np.concatenate((train_bottlenecks, val_bottlenecks))
    # y = np.concatenate((train_ground_truth_indices, val_ground_truth_indices))
    # grid_search.fit(
    #     X=X,
    #     y=y,
    #     X_valid=val_bottlenecks,
    #     y_valid=val_ground_truth_indices,
    #     n_epochs=num_epochs,
    #     eval_freq=eval_freq,
    #     ckpt_freq=ckpt_freq,
    #     early_stop_eval_freq=early_stopping_eval_freq
    # )
    best_params = grid_search.best_params_
    tf.logging.info(msg='Finished GridSearch! Best performing parameter set: %s' % best_params)
    # This is a refit operation, notify TensorBoard to replace the previous run's logging data:
    best_params['refit'] = True
    # Replace the current model parameters with the best combination from the GridSearch:
    current_params = tfh_classifier.get_params()
    tf.logging.info(msg='Serializing Grid Search CV results to: %s' % os.path.join(log_dir, 'gs_results.pkl'))
    gs_results = grid_search.cv_results_
    df_gs_results = pd.DataFrame.from_dict(gs_results)
    gs_results_path = os.path.join(log_dir, 'gs_results.csv')
    df_gs_results.to_csv(gs_results_path)
    # print(df_gs_results.head())
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
        n_epochs=num_epochs,
        eval_freq=eval_freq,
        ckpt_freq=ckpt_freq,
        early_stop_eval_freq=early_stopping_eval_freq
    )
    tf.logging.info(msg='Classifier re-fit! Model ready for inference.')
    y_pred = tfh_classifier.predict(X=val_bottlenecks)
    print('Classifier accuracy_score: %.2f' % accuracy_score(val_ground_truth_indices, y_pred))

    tf.logging.info(msg='Exporting re-fit model for future inference and evaluation to: %s' % model_export_dir)
    tfh_classifier.export_model(saved_model_dir=model_export_dir, human_readable_class_labels=class_labels)


def main(run_config):
    """
    main:
    :return:
    """
    """
    TensorBoard summaries directory:
    """
    summaries_dir = 'C:\\tmp\\summaries'
    model_export_dir = os.path.join('C:\\tmp\\summaries', 'trained_model')
    _clear_temp_folder(os.path.join(summaries_dir, os.pardir))
    # _prepare_tensor_board_directories(tb_summaries_dir='C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\TensorFlow\\TFHub\\tmp\\summaries\\trained_model\\')
    _prepare_model_export_directories(model_export_dir=model_export_dir)

    # Run preliminary setup operations and retrieve partitioned bottlenecks dataframe:
    _run_setup(tb_summaries_dir=summaries_dir)
    bottleneck_executor = BottleneckExecutor(
        image_dir=run_config['image_dir'],
        tfhub_module_url='https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1',
        compressed_bottleneck_file_path=run_config['bottleneck_path'],
        logging_dir=run_config['logging_dir']
    )
    bottlenecks = bottleneck_executor.get_bottlenecks()
    train_bottlenecks, val_bottlenecks, test_bottlenecks = bottleneck_executor.get_partitioned_bottlenecks(train_percent=0.80, val_percent=0.20, test_percent=0.20)
    class_labels = list(bottlenecks['class'].unique())
    tf.logging.info(
        'Partitioned (N=%d) total bottleneck vectors into training (N=%d), validation (N=%d), and testing (N=%d) datasets.'
        % (bottlenecks.shape[0], train_bottlenecks.shape[0], val_bottlenecks.shape[0], test_bottlenecks.shape[0])
    )
    tf.logging.info('Detected %d unique class labels in the bottlenecks dataframe' % len(class_labels))
    train_bottleneck_values = train_bottlenecks['bottleneck'].tolist()
    train_bottleneck_values = np.array(train_bottleneck_values)
    val_bottleneck_values = val_bottlenecks['bottleneck'].tolist()
    val_bottleneck_values = np.array(val_bottleneck_values)
    train_bottleneck_ground_truth_labels = train_bottlenecks['class'].values
    # Convert the labels into indices (one hot encoding by index):
    train_bottleneck_ground_truth_indices = np.array([class_labels.index(ground_truth_label)
                                                      for ground_truth_label in train_bottleneck_ground_truth_labels])
    val_bottleneck_ground_truth_labels = val_bottlenecks['class'].values
    # Convert the labels into indices (one hot encoding by index):
    val_bottleneck_ground_truth_indices = np.array([class_labels.index(ground_truth_label)
                                                    for ground_truth_label in val_bottleneck_ground_truth_labels])
    # train_bottlenecks, train_ground_truth_indices = _get_all_cached_bottlenecks(
    #     bottleneck_dataframe=bottleneck_dataframes['train'],
    #     class_labels=class_labels
    # )
    #
    # val_bottlenecks, val_ground_truth_indices = _get_all_cached_bottlenecks(
    #     bottleneck_dataframe=bottleneck_dataframes['val'],
    #     class_labels=class_labels
    # )
    # tf.logging.info(msg='Obtained bottleneck values from dataframe. Performed corresponding one-hot encoding of class labels')
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
    tb_log_dir = 'C:\\tmp\\summaries'
    _run_grid_search(
        dataset=run_config['dataset'],
        train_bottlenecks=train_bottleneck_values,
        train_ground_truth_indices=train_bottleneck_ground_truth_indices,
        initializers=initializer_options,
        activations=activation_options,
        optimizers=optimizer_options,
        class_labels=class_labels,
        val_bottlenecks=val_bottleneck_values,
        val_ground_truth_indices=val_bottleneck_ground_truth_indices,
        log_dir=tb_log_dir,
        model_export_dir=model_export_dir
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
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info(msg='TensorFlow Version: %s' % tf.VERSION)
    tf.logging.info(msg='tf.keras Version: %s' % tf.keras.__version__)
    run_configs = {
        'DEBUG': {
            'dataset': 'DEBUG',
            'image_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\GoingDeeper\\images',
            'bottleneck_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\GoingDeeper\\images\\bottlenecks.pkl',
            'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\DataAcquisition\\CleaningResults\\DEBUG'
        },
        'BOON': {
            'dataset': 'BOON',
            'image_dir': 'D:\\data\\BOON\\images',
            'bottleneck_path': 'D:\\data\\BOON\\bottlenecks.pkl',
            'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\DataAcquisition\\CleaningResults\\BOON'
        },
        'GoingDeeper': {
            'dataset': 'GoingDeeper',
            'image_dir': 'D:\\data\\GoingDeeperData\\images',
            'bottleneck_path': 'D:\\data\\GoingDeeperData\\bottlenecks.pkl',
            'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\DataAcquisition\\CleaningResults\\GoingDeeper'
        },
        'SERNEC': {
            'dataset': 'SERNEC',
            'image_dir': 'D:\\data\\SERNEC\\images',
            'bottleneck_path': 'D:\\data\\SERNEC\\bottlenecks.pkl',
            'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\DataAcquisition\\CleaningResults\\SERNEC'
        }
    }
    main(run_configs['BOON'])
    '''
    Execute this script under a shell instead of importing as a module. Ensures that the main function is called with
    the proper command line arguments (builds on default argparse). For more information see:
    https://stackoverflow.com/questions/33703624/how-does-tf-app-run-work
    '''
    # tf.app.run(main=main)
