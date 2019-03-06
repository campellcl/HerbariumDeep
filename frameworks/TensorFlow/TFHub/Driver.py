import math
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from frameworks.TensorFlow.Keras.Estimators.InceptionV3Estimator import InceptionV3Estimator
from frameworks.DataAcquisition.BottleneckExecutor import BottleneckExecutor
from frameworks.DataAcquisition.ImageExecutor import ImageExecutor


def preprocess_image(image, height=299, width=299, num_channels=3):
    image = tf.image.decode_jpeg(image, channels=num_channels)
    image = tf.image.resize_images(image, [height, width])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


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


def get_optimizer_options(static_learning_rate, momentum_const=None, adam_beta1=None, adam_beta2=None,
                          adam_epsilon=None, adagrad_init_accum=None, adadelta_rho=None, adadelta_epsilon=None):
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


def _run_grid_search_from_drive(train_image_paths, train_ground_truth_labels, class_labels, initializers, activations,
                                optimizers, tb_log_dir, val_image_paths=None, val_ground_truth_labels=None):
    """
    _run_grid_search_from_drive: Performs an exhaustive hyperparameter Grid Search via SKLearn directly from images on
        the hard drive. Data is coerced into a tf.Dataset to take advantage of the prefetch buffering to raise GPU
        efficiency. Preliminary tests show an increase from 20% to 85% GPU utilization when streaming direct from the
        prefetch buffer to the GPU.
    :param train_image_paths: A list of images that comprise the training dataset.
    :param train_ground_truth_labels: A list of ground truth labels (one-hot encoded) which correspond to the provided
        train_image_paths.
    :param class_labels: A list of human readable labels for use in visualization and metrics.
    :param initializers: A list of possible weight initializers dependent upon the provided meta-hyperparameters at
        runtime.
    :param activations: A list of possible activation functions dependent upon the provided meta-hyperparameters at
        runtime.
    :param optimizers: A list of possible optimizers (loss minimizers) dependent upon the provided meta-hyperparameters
        at runtime.
    :param val_image_paths: A list of images that comprise the validation training dataset.
    :param val_ground_truth_labels: A list of ground truth labels (one-hot encoded) which correspond to the provided
        val_image_paths.
    :return:
    """
    params = {
        'is_fixed_feature_extractor': [True],
        'initializer': [initializers['he_normal'], initializers['he_uniform'], initializers['truncated_normal']],
        'activation': [activations['LeakyReLU'], activations['ELU']],
        'optimizer': [optimizers['Adam'], optimizers['Nesterov']],
        'train_batch_size': [16, 20, 60]
    }
    num_epochs = 1000
    eval_freq = 10
    ckpt_freq = 0
    keras_classifier = InceptionV3Estimator(class_labels=class_labels, num_classes=len(class_labels), random_state=42, tb_log_dir=tb_log_dir)
    cv = [(slice(None), slice(None))]
    grid_search = GridSearchCV(keras_classifier, params, cv=cv, verbose=2, refit=False, n_jobs=1)
    tf.logging.info(msg='Running GridSearch...')
    grid_search.fit(
        X=train_image_paths,
        y=train_ground_truth_labels,
        num_epochs=num_epochs,
        eval_freq=eval_freq,
        ckpt_freq=ckpt_freq,
        fed_bottlenecks=False,
        X_val=val_image_paths,
        y_val=val_ground_truth_labels
    )
    tf.logging.info(msg='Finished GridSearch! Restoring best performing parameter set...')
    best_params = grid_search.best_params_
    # If this is a refit operation, notify TensorBoard to log to a different directory to avoid conflicting summaries:
    best_params['is_refit'] = True
    current_params = keras_classifier.get_params()
    current_params.update(best_params)
    keras_classifier.set_params(**current_params)
    tf.logging.info(msg='Model hyperparameters have been set to the highest scoring settings reported by GridSearch. Now fitting a classifier with these hyperparameters: %s' % (current_params))
    # Re-fit the model using the best parameter combination from the GridSearch:
    keras_classifier.fit(
        X_train=train_image_paths,
        y_train=train_ground_truth_labels,
        num_epochs=num_epochs,
        eval_freq=eval_freq,
        ckpt_freq=ckpt_freq,
        fed_bottlenecks=False,
        X_val=val_image_paths,
        y_val=val_ground_truth_labels
    )
    tf.logging.info(msg='Classifier re-fit! Model ready for inference.')
    y_pred = keras_classifier.predict(X=val_image_paths)
    # print('Classifier accuracy_score: %.2f' % accuracy_score(val_ground_truth_indices, y_pred))

def _run_grid_search_from_memory(train_bottlenecks, train_ground_truth_indices, class_labels, initializers, activations,
                                 optimizers, val_bottlenecks=None, val_ground_truth_indices=None):
    """
    _run_grid_search_from_memory: Utilizes a feed-dict based approach to input bottleneck tensors which have already
        undergone forward propagation in the source network directly from memory. This method is included to support
        legacy code which has already leveraged TFHub modules (and by association TFSlim modules) to pre-compute
        bottleneck vectors in the ConvNet as Fixed-Feature Extractor transfer learning context. If this was production
        code, this method would be slated for deprecation; as upward of a 60% increase in GPU utilization was achieved
        by not relying on feed-dicts and instead opting for tf.Dataset interoperability.
    :param train_bottlenecks:
    :param train_ground_truth_indices:
    :param class_labels:
    :param initializers:
    :param activations:
    :param optimizers:
    :param val_bottlenecks:
    :param val_ground_truth_indices:
    :return:
    """
    params = {
        'is_fixed_feature_extractor': [True],
        'optimizer': [optimizers['Adam']],
        'train_batch_size': [20, 40]
    }

    # params = {
    #     'initializer': [initializers['he_normal'], initializers['he_uniform'], initializers['truncated_normal']],
    #     'activation': [activations['LeakyReLU'], activations['ELU']],
    #     'optimizer': [optimizers['Adam'], optimizers['Nesterov']],
    #     'train_batch_size': [20, 60, 100]
    # }
    num_epochs = 100
    eval_freq = 10
    ckpt_freq = 0
    # tfh_classifier = TFHClassifier(random_state=42, class_labels=class_labels)
    keras_classifier = InceptionV3Estimator(num_classes=len(class_labels), random_state=42)
    cv = [(slice(None), slice(None))]
    grid_search = GridSearchCV(keras_classifier, params, cv=cv, verbose=2, refit=False, n_jobs=1)
    tf.logging.info(msg='Running GridSearch...')
    grid_search.fit(X=train_bottlenecks, y=train_ground_truth_indices, is_bottlenecks=True)
    tf.logging.info(msg='Finished GridSearch! Restoring best performing parameter set...')


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
    if intermediate_output_graphs_dir is not None:
        if not os.path.exists(intermediate_output_graphs_dir):
            os.makedirs(intermediate_output_graphs_dir)
    return

def main(run_config):
    tb_log_dir = 'C:\\Users\\ccamp\Documents\\GitHub\\HerbariumDeep\\frameworks\\TensorFlow\\TFHub\\tmp\\summaries'
    _prepare_tensor_board_directories(tb_summaries_dir=tb_log_dir, intermediate_output_graphs_dir=None)
    # image_executor = ImageExecutor(img_root_dir=run_config['image_dir'], logging_dir=run_config['logging_dir'], min_num_images_per_class=20, accepted_extensions=['jpg', 'jpeg'])
    bottleneck_executor = BottleneckExecutor(
        image_dir=run_config['image_dir'],
        logging_dir=run_config['logging_dir'],
        tfhub_module_url='https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1',
        compressed_bottleneck_file_path=run_config['bottleneck_path']
    )
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
    all_bottlenecks = bottleneck_executor.get_bottlenecks()
    class_labels = list(all_bottlenecks['class'].unique())
    train_bottlenecks, val_bottlenecks, test_bottlenecks = bottleneck_executor.get_partitioned_bottlenecks()
    train_bottleneck_values = train_bottlenecks['bottleneck'].tolist()
    train_bottleneck_values = np.array(train_bottleneck_values)
    train_bottleneck_ground_truth_labels = train_bottlenecks['class'].values
    # Convert the labels into indices (one hot encoding by index):
    train_bottleneck_ground_truth_indices = np.array([class_labels.index(ground_truth_label)
                                                      for ground_truth_label in train_bottleneck_ground_truth_labels])
    val_bottleneck_values = val_bottlenecks['bottleneck'].tolist()
    val_bottleneck_values = np.array(val_bottleneck_values)
    val_bottleneck_ground_truth_labels = val_bottlenecks['class'].values
    # Convert the labels into indices (one hot encoding by index):
    val_bottleneck_ground_truth_indices = np.array([class_labels.index(ground_truth_label)
                                                    for ground_truth_label in val_bottleneck_ground_truth_labels])

    _run_grid_search_from_drive(
        train_image_paths=train_bottlenecks['path'].values,
        train_ground_truth_labels=train_bottleneck_ground_truth_indices,
        val_image_paths=val_bottlenecks['path'].values,
        val_ground_truth_labels=val_bottleneck_ground_truth_indices,
        initializers=initializer_options,
        optimizers=optimizer_options,
        activations=activation_options,
        class_labels=class_labels,
        tb_log_dir=tb_log_dir
    )

    # _run_grid_search_from_memory(
    #     train_bottlenecks=train_bottleneck_values,
    #     train_ground_truth_indices=train_bottleneck_ground_truth_indices,
    #     initializers=initializer_options,
    #     activations=activation_options,
    #     optimizers=optimizer_options,
    #     class_labels=class_labels
    # )

    # train_bottlenecks, train_ground_truth_indices = _get_all_cached_bottlenecks(
    #     bottleneck_dataframe=bottlenecks['train'],
    #     class_labels=class_labels
    # )
    # val_bottlenecks, val_ground_truth_indices = _get_all_cached_bottlenecks(
    #     bottleneck_dataframe=bottlenecks['val'],
    #     class_labels=class_labels
    # )
    # _run_grid_search(
    #     train_bottlenecks=train_bottlenecks,
    #     train_ground_truth_indices=train_ground_truth_indices,
    #     initializers=initializer_options,
    #     activations=activation_options,
    #     optimizers=optimizer_options,
    #     class_labels=class_labels
    # )

    # num_samples = bottlenecks.shape[0]
    # num_classes = len(bottlenecks['class'].unique())
    # all_image_paths = bottlenecks['path'].values
    # all_image_labels = bottlenecks['class'].values
    # # Labels to one-hot encoding mapping
    # label_to_index = dict((name, index) for index, name in enumerate(class_labels))
    # all_image_labels_one_hot = [label_to_index[label] for label in all_image_labels]
    # path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    # image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.contrib.data.AUTOTUNE)
    # label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels_one_hot, tf.int64))
    # image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    # steps_per_epoch = math.ceil(len(all_image_paths)/BATCH_SIZE)
    # ds = image_label_ds.cache()
    # # Buffer size must be at least as large as number of sample images or sampling with replacement will occur.
    # ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=num_samples))
    # ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    # _run_grid_search(
    #     tensorflow_dataset=ds,
    #     train_ds_x=image_ds,
    #     train_ds_y=label_ds,
    #     num_classes=num_classes,
    #     initializers=initializer_options,
    #     activations=activation_options,
    #     optimizers=optimizer_options
    # )
    # print()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info(msg='TensorFlow Version: %s' % tf.VERSION)
    tf.logging.info(msg='tf.Keras Version: %s' % tf.keras.__version__)
    run_configs = {
        'DEBUG': {
            'image_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\GoingDeeper\\images',
            'bottleneck_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\GoingDeeper\\images\\bottlenecks.pkl',
            'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeepKeras\\frameworks\\DataAcquisition\\CleaningResults\\DEBUG'
        },
        'BOON': {
            'image_dir': 'D:\\data\\BOON\\images',
            'bottleneck_path': 'D:\\data\\BOON\\bottlenecks.pkl',
            'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeepKeras\\frameworks\\DataAcquisition\\CleaningResults\\BOON'
        },
        'GoingDeeper': {
            'image_dir': 'D:\\data\\GoingDeeperData\\images',
            'bottleneck_path': 'D:\\data\\GoingDeeperData\\bottlenecks.pkl',
            'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeepKeras\\frameworks\\DataAcquisition\\CleaningResults\\GoingDeeper'
        },
        'SERNEC': {}
    }

    main(run_configs['BOON'])
