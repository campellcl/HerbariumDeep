import os
import numpy as np
import tensorflow as tf
import pytest
from frameworks.DataAcquisition.BottleneckExecutor import BottleneckExecutor
from frameworks.Sklearn.GridSearchCVSaveRestore import GridSearchCVSaveRestore
from frameworks.Sklearn.GridSearchCVSaveRestore import CrossValidationSplitter
from frameworks.TensorFlow.TFHub.TFHClassifier import TFHClassifier



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


@pytest.fixture(scope='class')
def set_up(request):
    run_configs = {
        'BOON': {
            'dataset': 'BOON',
            'image_dir': 'D:\\data\\BOON\\images',
            'bottleneck_path': 'D:\\data\\BOON\\bottlenecks.pkl',
            'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\DataAcquisition\\CleaningResults\\BOON'
        }
    }
    run_config = run_configs['BOON']

    model_export_dir = os.path.join('C:\\tmp\\summaries', 'trained_model')
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
    initializer_options = get_initializer_options()
    activation_options = get_activation_options(leaky_relu_alpha=0.2)
    optimizer_options = get_optimizer_options(
        static_learning_rate=0.001,
        momentum_const=0.9,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08
    )
    # Inject class variables:
    request.cls.dataset = run_config['dataset']
    request.cls.train_bottlenecks = train_bottleneck_values
    request.cls.train_ground_truth_indices = train_bottleneck_ground_truth_indices
    request.cls.class_labels = class_labels
    request.cls.val_bottlenecks = val_bottleneck_values
    request.cls.val_ground_truth_indices = val_bottleneck_ground_truth_indices
    request.cls.initializers = initializer_options
    request.cls.activations = activation_options
    request.cls.optimizers = optimizer_options
    request.cls.log_dir = 'C:\\tmp\\summaries'
    yield


@pytest.mark.usefixtures('set_up')
class TestGridSearchSaveRestore:

    def test_grid_search_save_restore(self):
        num_train_samples = self.train_bottlenecks.shape[0]
        num_val_samples = self.val_bottlenecks.shape[0]
        cv_results_save_loc = 'C:\\cv_results\\'
        params = None

        if self.dataset == 'SERNEC':
            raise NotImplementedError
        elif self.dataset == 'GoingDeeper':
            raise NotImplementedError
        elif self.dataset == 'BOON':
            params = {
                'initializer': [self.initializers['he_normal'], self.initializers['he_uniform'], self.initializers['truncated_normal']],
                'activation': [self.activations['LeakyReLU'], self.activations['ELU']],
                'optimizer': [self.optimizers['Nesterov'], self.optimizers['Adam']],
                'train_batch_size': [20, 60, 100, 1000]
            }
            tf.logging.warning(msg='WARNING: DEBUG SETTINGS IN EFFECT FOR TESTING')
            num_epochs = 3
            eval_freq = 1
            early_stopping_eval_freq = 1
            ckpt_freq = 0
            tf.logging.info(msg='Initialized SkLearn parameter grid: %s' % params)
        elif self.dataset == 'DEBUG':
            raise NotImplementedError
        else:
            tf.logging.error(msg='FATAL ERROR: Could not recognize the provided dataset: \'%s\', exiting.' % self.dataset)
            exit(-1)

        tfh_classifier = TFHClassifier(dataset=self.dataset, random_state=42, class_labels=self.class_labels, tb_logdir=self.log_dir)
        tf.logging.info(msg='Initialized TensorFlowHub Classifier (TFHClassifier) Instance')
        custom_cv_splitter = CrossValidationSplitter(train_size=num_train_samples, test_size=num_val_samples, n_splits=1)
        grid_search = GridSearchCVSaveRestore(
            estimator=tfh_classifier, param_grid=params, cv_results_save_freq=1, cv_results_save_loc=cv_results_save_loc,
            early_termination_model_for_testing=1,
            cv=custom_cv_splitter, verbose=2, refit=False, return_train_score=False, error_score='raise', scoring=None
        )
        tf.logging.info(msg='Instantiated GridSearch.')
        X = np.concatenate((self.train_bottlenecks, self.val_bottlenecks))
        y = np.concatenate((self.train_ground_truth_indices, self.val_ground_truth_indices))
        grid_search.fit(
            X=X,
            y=y,
            X_valid=self.val_bottlenecks,
            y_valid=self.val_ground_truth_indices,
            n_epochs=num_epochs,
            eval_freq=eval_freq,
            ckpt_freq=ckpt_freq,
            early_stop_eval_freq=early_stopping_eval_freq
        )
        assert True is True


