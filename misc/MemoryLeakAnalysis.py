import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.exceptions import NotFittedError
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from frameworks.DataAcquisition.BottleneckExecutor import BottleneckExecutor
import math
from tensorflow.keras.utils import to_categorical


class MemoryLeakTestClassifier(BaseEstimator, ClassifierMixin, tf.keras.Model):

    def __init__(self, train_from_bottlenecks, num_classes, initializer=tf.variance_scaling_initializer(), activation=tf.nn.elu, optimizer=tf.train.AdamOptimizer, train_batch_size=-1, val_batch_size=-1):
        super(MemoryLeakTestClassifier, self).__init__(name='memory_leak_test_classifier')
        self.train_from_bottlenecks = train_from_bottlenecks
        self.num_classes = num_classes
        self.initializer = initializer
        self.activation = activation
        self.optimizer = optimizer
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self._is_trained = False

    def _build_model_and_graph_def(self):
        tf.keras.backend.clear_session()
        tf.logging.warning(msg='Cleared Keras\' backend session to attempt to free memory.')

        base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))

        for layer in base_model.layers:
            layer.trainable = False

        if not self.train_from_bottlenecks:
            raise NotImplementedError
        else:
            bottlenecks = Input(shape=(base_model.output_shape[-1],), name='bottleneck')
            logits = Dense(self.num_classes, activation=self.activation)(bottlenecks)
            y_proba = Dense(self.num_classes, activation='softmax')(logits)
            self._keras_model = Model(inputs=bottlenecks, outputs=y_proba)

    def _tf_data_generator_from_memory(self, image_bottlenecks, image_encoded_labels, is_training):
        # Convert to categorical format for keras (see bottom of page: https://keras.io/losses/):
        bottleneck_ds = tf.data.Dataset.from_tensor_slices(image_bottlenecks)
        categorical_labels = to_categorical(image_encoded_labels, num_classes=self.num_classes)
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(categorical_labels, tf.int64))
        bottleneck_label_ds = tf.data.Dataset.zip((bottleneck_ds, label_ds))
        num_images = len(image_bottlenecks)
        # 1. Cache dataset:
        ds = bottleneck_label_ds.cache()
        # 2. Shuffle entire dataset:
        ds = ds.shuffle(buffer_size=num_images)
        # 3. Apply the shuffle operation immediately:
        # ds = ds.repeat()
        # 4. Partition into batches:
        if is_training:
            ds = ds.batch(batch_size=self.train_batch_size)
        else:
            ds = ds.batch(batch_size=self.val_batch_size)
        # 5. Apply the batch operation immediately:
        ds = ds.repeat()
        # 6. Allocate prefetch buffer:
        ds = ds.prefetch(tf.contrib.data.AUTOTUNE)
        return ds

    def fit(self, X_train, y_train, fed_bottlenecks=False, num_epochs=1000, eval_freq=1, ckpt_freq=0, early_stopping_eval_freq=1, X_val=None, y_val=None):
        self._build_model_and_graph_def()
        # X_train is an array of bottlenecks, y_train is the associated one-hot encoded labels.
        num_train_bottlenecks = len(X_train)
        if self.train_batch_size == -1:
            self.train_batch_size = num_train_bottlenecks
        train_steps_per_epoch = math.ceil(num_train_bottlenecks/self.train_batch_size)

        if X_val is not None:
            num_val_bottlenecks = len(X_val)
            if self.val_batch_size == -1:
                self.val_batch_size = num_val_bottlenecks
            val_steps_per_epoch = math.ceil(num_val_bottlenecks/self.val_batch_size)

        self._keras_model.compile(
            optimizer=self.optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        tf.logging.info(msg='Compiled Keras model.')

        # train_ds = self._tf_data_generator_from_memory(image_bottlenecks=X_train, image_encoded_labels=y_train, is_training=True)

        self._is_trained = True
        return self

    def predict(self, X):
        if not self._is_trained:
            class_indices = np.argmax(self.predict_proba(X=X, batch_size=self.train_batch_size), axis=1)
        else:
            class_indices = np.argmax(self.predict_proba(X=X, batch_size=self.val_batch_size), axis=1)
        return np.array(class_indices, np.int32)

    def predict_proba(self, X, batch_size):
        if not self._is_trained:
            raise NotFittedError('This \'%s\' instance is not yet fitted.' % self.__class__.__name__)
        if not self.train_from_bottlenecks:
            raise NotImplementedError
        else:
            num_bottlenecks = len(X)
            steps_per_epoch = math.ceil(num_bottlenecks/batch_size)
            y_proba = self._keras_model.predict(X, batch_size=batch_size, verbose=0, steps=steps_per_epoch)
            return y_proba

    def _get_model_params(self):
        raise NotImplementedError

    def _restore_model_params(self):
        raise NotImplementedError


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


def main(run_config):
    static_learning_rate=0.001
    momentum_const = 0.9
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_epsilon = 1e-08

    bottleneck_executor = BottleneckExecutor(
        image_dir=run_config['image_dir'],
        logging_dir=run_config['logging_dir'],
        tfhub_module_url='https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1',
        compressed_bottleneck_file_path=run_config['bottleneck_path']
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
    num_train_samples = len(train_bottleneck_values)
    num_val_samples = len(val_bottleneck_values)

    params = {
        'initializer': [tf.initializers.he_normal(), tf.initializers.he_uniform(), tf.initializers.truncated_normal],
        'activation': [tf.nn.leaky_relu, tf.nn.elu],
        'optimizer': [
            tf.train.MomentumOptimizer(
                learning_rate=static_learning_rate,
                momentum=momentum_const,
                use_nesterov=True
            ),
            tf.train.AdamOptimizer(
                learning_rate=static_learning_rate,
                beta1=adam_beta1,
                beta2=adam_beta2,
                epsilon=adam_epsilon
            )
        ],
        'train_batch_size': [20, 60, 100]
    }
    num_epochs = 10000  #10,000
    eval_freq = 10
    early_stopping_eval_freq = 5
    ckpt_freq = 0

    mem_leak_test_clf = MemoryLeakTestClassifier(train_from_bottlenecks=True, num_classes=len(class_labels))
    custom_cv_splitter = CrossValidationSplitter(train_size=num_train_samples, test_size=num_val_samples, n_splits=1)
    grid_search = GridSearchCV(mem_leak_test_clf, params, cv=custom_cv_splitter, verbose=2, refit=False, n_jobs=1, return_train_score=False)
    tf.logging.info('Running GridSearch...')
    X = np.concatenate((train_bottleneck_values, val_bottleneck_values))
    y = np.concatenate((train_bottleneck_ground_truth_indices, val_bottleneck_ground_truth_indices))
    grid_search.fit(
        X=X,
        y=y,
        num_epochs=num_epochs,
        eval_freq=eval_freq,
        ckpt_freq=ckpt_freq,
        early_stopping_eval_freq=early_stopping_eval_freq,
        fed_bottlenecks=True,
        X_val=val_bottlenecks,
        y_val=val_bottleneck_ground_truth_indices
    )
    tf.logging.info(msg='Finished GridSearch! Restoring best performing parameter set...')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info(msg='TensorFlow Version: %s' % tf.VERSION)
    tf.logging.info(msg='tf.keras Version: %s' % tf.keras.__version__)
    run_configs = {
        'DEBUG': {
            'dataset': 'DEBUG',
            'image_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\GoingDeeper\\images',
            'bottleneck_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\GoingDeeper\\images\\bottlenecks.pkl',
            'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeepKeras\\frameworks\\DataAcquisition\\CleaningResults\\DEBUG'
        },
        'BOON': {
            'dataset': 'BOON',
            'image_dir': 'D:\\data\\BOON\\images',
            'bottleneck_path': 'D:\\data\\BOON\\bottlenecks.pkl',
            'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeepKeras\\frameworks\\DataAcquisition\\CleaningResults\\BOON'
        },
        'GoingDeeper': {
            'dataset': 'GoingDeeper',
            'image_dir': 'D:\\data\\GoingDeeperData\\images',
            'bottleneck_path': 'D:\\data\\GoingDeeperData\\bottlenecks.pkl',
            'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeepKeras\\frameworks\\DataAcquisition\\CleaningResults\\GoingDeeper'
        },
        'SERNEC': {}
    }
    main(run_configs['BOON'])
