from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import math

from frameworks.DataAcquisition.BottleneckExecutor import BottleneckExecutor


class InceptionV3Estimator(BaseEstimator, ClassifierMixin, tf.keras.Model):

    def __init__(self, num_classes, train_batch_size=-1, val_batch_size=-1, activation=tf.nn.elu,
                 optimizer=tf.train.AdamOptimizer, is_fixed_feature_extractor=True, random_state=None):
        super(InceptionV3Estimator, self).__init__(name='inception_v3_estimator')
        self.num_classes = num_classes
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.activation = activation
        self.optimizer = optimizer
        self.random_state = random_state
        self.is_fixed_feature_extractor = is_fixed_feature_extractor
        self._keras_model = None

        self._session = None
        self._graph = None
        # TensorBoard setup:
        self._train_writer = None
        self._val_writer = None

    @staticmethod
    def _preprocess_image(image, height=299, width=299, num_channels=3):
        image = tf.image.decode_jpeg(image, channels=num_channels)
        image = tf.image.resize_images(image, [height, width])
        image /= 255.0  # normalize to [0,1] range
        return image

    @staticmethod
    def _load_and_preprocess_image(path):
        image = tf.read_file(path)
        return InceptionV3Estimator._preprocess_image(image)

    def _build_model_and_graph_def(self):
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        if self.is_fixed_feature_extractor:
            # input_tensor = tf.placeholder(dtype=tf.float32, shape=(None, 299, 299, 3), name='resized_input_tensor')
            # base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))(input_tensor)
            base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
            self._keras_resized_input_handle_ = base_model.input
            self._session = tf.keras.backend.get_session()
            self._graph = self._session.graph
            # add a global spatial average pooling layer:
            x = base_model.output
            bottlenecks = GlobalAveragePooling2D()(x)
            # let's add a fully-connected layer output shape 1024
            logits = Dense(1024, activation='relu')(bottlenecks)
            # and a fully connected logistic layer for self.num_classes
            predictions = Dense(self.num_classes, activation='softmax')(logits)

            # this is the model we will train
            self._keras_model = Model(inputs=base_model.input, outputs=predictions)

            # first: train only the top layers (which were randomly initialized)
            # i.e. freeze all convolutional InceptionV3 layers
            for layer in base_model.layers:
                layer.trainable = False

    def call(self, resized_input_tensors):
        """
        call: The SKLearn fit method will invoke this equivalent Keras method (if necessary) when called. For additional
            information see: https://www.tensorflow.org/guide/keras#model_subclassing
        :param resized_input_tensors: A Tensor of shape (None, 299, 299, 3) for imagenet, where None = batch size.
        :return:
        """
        # Define forward pass here, using layers defined in _build_model_and_graph
        return self._keras_model(inputs=resized_input_tensors)

    # def compute_output_shape(self, input_shape):
    #     # TODO: Multiple inheritance issue with Keras and SKLearn super classes.
    #     #   For a possible resolution, see: https://stackoverflow.com/a/9575426/3429090
    #     super(InceptionV3Estimator, self).compute_output_shape(input_shape)

    def fit(self, X_train, y_train, bottlenecks=False, num_epochs=1000):
        """
        fit:
        :param X_train: What if this was a list of images? then could perform dataset conversions here...
        :param y_train:
        :param precomputed: True if X_train is
        :return:
        """
        self._build_model_and_graph_def()

        if not bottlenecks:
            # X_train is a list of image paths, y_train is the associated one-hot-encoded labels.
            num_images = len(X_train)
            if self.train_batch_size == -1:
                self.train_batch_size = num_images
            train_path_ds = tf.data.Dataset.from_tensor_slices(X_train)
            train_image_ds = train_path_ds.map(
                InceptionV3Estimator._load_and_preprocess_image,
                num_parallel_calls=tf.contrib.data.AUTOTUNE
            )
            # Convert to categorical format for keras (see bottom of page: https://keras.io/losses/)
            categorical_labels = to_categorical(y_train, num_classes=self.num_classes)
            train_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(categorical_labels, tf.int64))
            train_image_label_ds = tf.data.Dataset.zip((train_image_ds, train_label_ds))
            tf.logging.info(msg='Training Batch Size: %d' % self.train_batch_size)
            steps_per_epoch = math.ceil(len(X_train)/self.train_batch_size)
            train_ds = train_image_label_ds.cache()
            # Data will not have been shuffled:
            train_ds = train_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=num_images))
            train_ds = train_ds.batch(self.train_batch_size).prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

            if self.is_fixed_feature_extractor:
                self._keras_model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                self._keras_model.fit(train_ds, epochs=num_epochs, steps_per_epoch=steps_per_epoch)
        else:
            num_images = X_train.shape[0]
            if self.train_batch_size == -1:
                train_batch_size = num_images
            # X_train is bottleneck data of size (?, 2048):
            if self.is_fixed_feature_extractor:
                self._keras_model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                self._keras_model.fit(X_train, y_train)

        # TODO: When to kill session?
        # self._session.close()
        return self
