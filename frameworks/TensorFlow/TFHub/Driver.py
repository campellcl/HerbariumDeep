import math
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from frameworks.TensorFlow.Keras.Estimators.InceptionV3 import InceptionV3Estimator
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


def _run_grid_search_from_memory(train_bottlenecks, train_ground_truth_indices, class_labels, initializers, activations, optimizers, val_bottlenecks=None, val_ground_truth_indices=None):
    params = {
        'is_fixed_feature_extractor': [True],
        'optimizer': [optimizers['Adam']]
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
    grid_search.fit(X=train_bottlenecks, y=train_ground_truth_indices)
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


def main(run_config):
    BATCH_SIZE = 20
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
    _run_grid_search_from_memory(
        train_bottlenecks=train_bottleneck_values,
        train_ground_truth_indices=train_bottleneck_ground_truth_indices,
        initializers=initializer_options,
        activations=activation_options,
        optimizers=optimizer_options,
        class_labels=class_labels
    )

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
            'bottleneck_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\TensorFlow\\TFHub\\bottlenecks.pkl',
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

    main(run_configs['DEBUG'])
