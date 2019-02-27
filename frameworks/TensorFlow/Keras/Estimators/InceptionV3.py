from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf
import numpy as np

from frameworks.DataAcquisition.BottleneckExecutor import BottleneckExecutor


class InceptionV3Estimator(BaseEstimator, ClassifierMixin, tf.keras.Model):

    def __init__(self, num_classes, activation=tf.nn.elu, optimizer=tf.train.AdamOptimizer, is_fixed_feature_extractor=True, random_state=None):
        super(InceptionV3Estimator, self).__init__(name='inception_v3_estimator')
        self.num_classes = num_classes
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
            x = GlobalAveragePooling2D()(x)
            # let's add a fully-connected layer
            x = Dense(1024, activation='relu')(x)
            # and a fully connected logistic layer for self.num_classes
            predictions = Dense(self.num_classes, activation='softmax')(x)

            # this is the model we will train
            self._keras_model = Model(inputs=base_model.input, outputs=predictions)

            # first: train only the top layers (which were randomly initialized)
            # i.e. freeze all convolutional InceptionV3 layers
            for layer in base_model.layers:
                layer.trainable = False



    def call(self, resized_input_tensors):
        # Define forward pass here, using layers defined in _build_model_and_graph
        return self._keras_model(inputs=resized_input_tensors)

    def fit(self, X_train, y_train):
        # self._session = tf.keras.backend.get_session()
        self._build_model_and_graph_def()
        if self.is_fixed_feature_extractor:
            self._keras_model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            self._keras_model.fit(X_train, y_train)
        return self
