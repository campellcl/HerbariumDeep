from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from keras import backend as K
import tensorflow as tf
import numpy as np
import math

from frameworks.DataAcquisition.BottleneckExecutor import BottleneckExecutor
from frameworks.TensorFlow.Keras.Callbacks.CustomCallbacks import FileWritersTensorBoardCallback


class InceptionV3Estimator(BaseEstimator, ClassifierMixin, tf.keras.Model):

    def __init__(self, num_classes, train_batch_size=-1, val_batch_size=-1, activation=tf.nn.elu,
                 optimizer=tf.train.AdamOptimizer, initializer=tf.variance_scaling_initializer(),
                 is_fixed_feature_extractor=True, random_state=None, tb_log_dir=None):
        super(InceptionV3Estimator, self).__init__(name='inception_v3_estimator')
        self.num_classes = num_classes
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.activation = activation
        self.optimizer = optimizer
        self.initializer = initializer
        self.random_state = random_state
        self.tb_log_dir = tb_log_dir

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
            # self._session = tf.Session()
            base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
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

            self._keras_resized_input_handle_ = base_model.input
            # self._session = tf.keras.backend.get_session()
            tf.logging.info(msg='self._session set to: %s' % self._session)
            # self._graph = self._session.graph
        else:
            raise NotImplementedError

    def call(self, inputs, training=None, mask=None):
        """
        call: The SKLearn fit method will invoke this equivalent Keras method (if necessary) when called. For additional
            information see: https://www.tensorflow.org/guide/keras#model_subclassing
        :param inputs: In parent method, a tensor or list of tensors. For this subclass, a re-sized input tensor of
            shape (None, 299, 299, 3) for InceptionV3, where None = batch size.
        :param training: Boolean or boolean scalar tensor, indicating whether to run
          the `Network` in training mode or inference mode.
        :param mask: A mask or list of masks. A mask can be
            either a tensor or None (no mask).
        :return:
        """
        # Define forward pass here, using layers defined in _build_model_and_graph
        return self._keras_model(inputs=inputs)

    # def compute_output_shape(self, input_shape):
    #     # TODO: Multiple inheritance issue with Keras and SKLearn super classes.
    #     #   For a possible resolution, see: https://stackoverflow.com/a/9575426/3429090
    #     super(InceptionV3Estimator, self).compute_output_shape(input_shape)

    def _tf_data_generator(self, image_file_paths, image_one_hot_encoded_labels, is_training):
        """

        :param image_file_paths:
        :param image_one_hot_encoded_labels:
        :param is_training: <boolean> True if this method is being invoked to yield a dataset during
        :return:
        :source url: https://gist.github.com/datlife/abfe263803691a8864b7a2d4f87c4ab8#file-mnist_tfdata-py-L30
        """
        def _preprocess_image(image, height=299, width=299, num_channels=3):
            image = tf.image.decode_jpeg(image, channels=num_channels)
            image = tf.image.resize_images(image, [height, width])
            image /= 255.0  # normalize to [0,1] range
            return image

        def _load_and_preprocess_image(path):
            image = tf.read_file(path)
            return _preprocess_image(image)

        path_ds = tf.data.Dataset.from_tensor_slices(image_file_paths)
        image_ds = path_ds.map(
            _load_and_preprocess_image,
            num_parallel_calls=tf.contrib.data.AUTOTUNE
        )
        # Convert to categorical format for keras (see bottom of page: https://keras.io/losses/):
        categorical_labels = to_categorical(image_one_hot_encoded_labels, num_classes=self.num_classes)
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(categorical_labels, tf.int64))
        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
        num_images = len(image_file_paths)
        # 1. Cache the pre-processed and decoded image files:
        ds = image_label_ds.cache()
        # 2. Shuffle all the entire dataset:
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

    # def _numpy_array_tensorflow_dataset_generator(self, sample_image_file_paths, sample_image_one_hot_encoded_class_labels, is_train_data):
    #     """
    #     _convert_to_tensorflow_dataset_generator: Converts the provided X_train, y_train or X_val, y_val array-like in-memory
    #         file paths to a generator which yields TensorFlow.data.Dataset's for use with Keras models with prefetch buffer allocation.
    #     :param sample_image_file_paths: X_train or X_valid a list of file paths.
    #     :param sample_image_one_hot_encoded_class_labels: y_train or y_valid, a list of one-hot encoded
    #         class labels which correspond to the provided sample_image_file_paths_array_like.
    #     :return ds: <TensorFlow.data.Dataset> Returns the provided sample images and one-hot encoded labels as a
    #         TensorFlow Dataset object.
    #     """
    #     path_ds = tf.data.Dataset.from_tensor_slices(sample_image_file_paths)
    #     image_ds = path_ds.map(
    #         InceptionV3Estimator._load_and_preprocess_image,
    #         num_parallel_calls=tf.contrib.data.AUTOTUNE
    #     )
    #     # Convert to categorical format for keras (see bottom of page: https://keras.io/losses/):
    #     categorical_labels = to_categorical(sample_image_one_hot_encoded_class_labels, num_classes=self.num_classes)
    #     label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(categorical_labels, tf.int64))
    #     image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    #     num_images = len(sample_image_file_paths)
    #     # if is_train_data:
    #     #     steps_per_epoch = math.ceil(num_images/self.train_batch_size)
    #     # else:
    #     #     steps_per_epoch = math.ceil(num_images/self.val_batch_size)
    #     ds = image_label_ds.cache()
    #     # Shuffle the data:
    #     ds = ds.shuffle(buffer_size=num_images)
    #     # ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=num_images))
    #     if is_train_data:
    #
    #         # ds = ds.batch(self.train_batch_size).prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    #         # ds = ds.batch(self.train_batch_size).prefetch(buffer_size=tf.contrib.data.AUTOTUNE).repeat()
    #     else:
    #         # ds = ds.batch(self.val_batch_size).prefetch(buffer_size=tf.contrib.data.AUTOTUNE).repeat()
    #     ds_iterator = ds.make_one_shot_iterator()
    #     next_batch = ds_iterator.get_next()
    #     # self._session.close()
    #     with tf.Session() as sess:
    #         tf.logging.info(msg='self._session is now: %s' % sess)
    #         while True:
    #             yield sess.run(next_batch)
    #             # yield tf.keras.backend.get_session().run(next_batch)

    # def _convert_to_tensorflow_dataset(self, sample_image_file_paths, sample_image_one_hot_encoded_class_labels, is_train_data):
    #     path_ds = tf.data.Dataset.from_tensor_slices(sample_image_file_paths)
    #     image_ds = path_ds.map(
    #         InceptionV3Estimator._load_and_preprocess_image,
    #         num_parallel_calls=tf.contrib.data.AUTOTUNE
    #     )
    #     # Convert to categorical format for keras (see bottom of page: https://keras.io/losses/):
    #     categorical_labels = to_categorical(sample_image_one_hot_encoded_class_labels, num_classes=self.num_classes)
    #     label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(categorical_labels, tf.int64))
    #     image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    #     num_images = len(sample_image_file_paths)
    #     # if is_train_data:
    #     #     steps_per_epoch = math.ceil(num_images/self.train_batch_size)
    #     # else:
    #     #     steps_per_epoch = math.ceil(num_images/self.val_batch_size)
    #     ds = image_label_ds.cache()
    #     # Shuffle the data:
    #     ds = ds.shuffle(buffer_size=num_images).repeat()
    #     # ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=num_images))
    #     if is_train_data:
    #         # ds = ds.batch(self.train_batch_size).prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    #         ds = ds.batch(self.train_batch_size).prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    #     else:
    #         ds = ds.batch(self.val_batch_size).prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    #     # ds_iterator = ds.make_one_shot_iterator()
    #     # next_batch = ds_iterator.get_next()
    #     # while True:
    #     #     yield self._session.run(next_batch)
    #     return ds

    def fit(self, X_train, y_train, fed_bottlenecks=False, num_epochs=1000, eval_freq=1, ckpt_freq=0, X_val=None, y_val=None):
        """
        fit:
        :param X_train: What if this was a list of images? then could perform dataset conversions here...
        :param y_train:
        :param precomputed: True if X_train is
        :return:
        """
        train_ds, val_ds = None, None
        num_train_images, num_val_images = None, None
        if X_val is not None and y_val is not None:
            has_validation_data = True
        else:
            has_validation_data = False

        self._build_model_and_graph_def()

        # Was a DataFrame with pre-computed bottlenecks fed to this method from memory, or just a list of image paths?
        if not fed_bottlenecks:
            # X_train is a list of image paths, y_train is the associated one-hot encoded labels.
            num_train_images = len(X_train)
            if self.train_batch_size == -1:
                self.train_batch_size = num_train_images

            # train_ds = self._convert_to_tensorflow_dataset(
            #     sample_image_file_paths=X_train,
            #     sample_image_one_hot_encoded_class_labels=y_train,
            #     is_train_data=True
            # )
            train_ds = self._tf_data_generator(image_file_paths=X_train, image_one_hot_encoded_labels=y_train, is_training=True)

            if has_validation_data:
                num_val_images = len(X_val)
                if self.val_batch_size == -1:
                    self.val_batch_size = num_val_images

                # val_ds = self._convert_to_tensorflow_dataset(
                #     sample_image_file_paths=X_val,
                #     sample_image_one_hot_encoded_class_labels=y_val,
                #     is_train_data=False
                # )
                val_ds = self._tf_data_generator(image_file_paths=X_val, image_one_hot_encoded_labels=y_val, is_training=False)

            if self.is_fixed_feature_extractor:
                steps_per_epoch = math.ceil(num_train_images/self.train_batch_size)
                self._keras_model.compile(
                    optimizer=self.optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                if has_validation_data:
                    # Has validation data, train with that in mind.
                    val_steps_per_epoch = math.ceil(num_val_images/self.val_batch_size)

                    # self._keras_model.fit_generator(
                    #     train_ds,
                    #     validation_data=val_ds,
                    #     epochs=num_epochs,
                    #     steps_per_epoch=steps_per_epoch,
                    #     validation_steps=val_steps_per_epoch,
                    #     callbacks=[
                    #         FileWritersTensorBoardCallback(
                    #             log_dir=self.tb_log_dir,
                    #             hyperparameter_string_repr=self.__repr__(),
                    #             write_graph=False
                    #         )
                    #     ]
                    # )

                    # self._keras_model.fit_generator(
                    #     self._numpy_array_tensorflow_dataset_generator(sample_image_file_paths=X_train, sample_image_one_hot_encoded_class_labels=y_train, is_train_data=True),
                    #     validation_data=self._numpy_array_tensorflow_dataset_generator(sample_image_file_paths=X_val, sample_image_one_hot_encoded_class_labels=y_val, is_train_data=False),
                    #     epochs=num_epochs,
                    #     steps_per_epoch=steps_per_epoch,
                    #     validation_steps=val_steps_per_epoch,
                    #     callbacks=[
                    #         FileWritersTensorBoardCallback(
                    #             log_dir=self.tb_log_dir,
                    #             hyperparameter_string_repr=self.__repr__(),
                    #             write_graph=False
                    #         )
                    #     ]
                    # )

                    self._keras_model.fit(
                        train_ds.make_one_shot_iterator(),
                        validation_data=val_ds.make_one_shot_iterator(),
                        epochs=num_epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=val_steps_per_epoch,
                        callbacks=[
                            FileWritersTensorBoardCallback(log_dir=self.tb_log_dir,hyperparameter_string_repr=self.__repr__(), write_graph=False)
                        ]
                    )

                else:
                    # No validation data, just train on the training data:
                    tf.logging.error(msg='Not implemented yet.')
                    raise NotImplementedError
                    # self._keras_model.fit_generator(
                    #     train_ds,
                    #     epochs=num_epochs,
                    #     steps_per_epoch=steps_per_epoch,
                    #     callbacks=[
                    #         FileWritersTensorBoardCallback(
                    #             log_dir=self.tb_log_dir,
                    #             hyperparameter_string_repr=self.__repr__(),
                    #             write_graph=False
                    #         )
                    #     ]
                    # )
        else:
            # X_train is an array of bottlenecks, y_train is the associated one-hot encoded labels.
            tf.logging.error(msg='Not implemented yet.')
            raise NotImplementedError
            # num_images = X_train.shape[0]
            # if self.train_batch_size == -1:
            #     train_batch_size = num_images
            # # X_train is bottleneck data of size (?, 2048):
            # if self.is_fixed_feature_extractor:
            #     self._keras_model.compile(
            #         optimizer=self.optimizer,
            #         loss='categorical_crossentropy',
            #         metrics=['accuracy']
            #     )
            #     self._keras_model.fit(X_train, y_train)

        # TODO: When to kill session?
        # self._session.close()
        return self

    @staticmethod
    def _get_initializer_repr(initializer):
        function_repr = str(initializer)
        if 'random_uniform' in function_repr:
            return 'INIT_UNIFORM'
        elif 'random_normal' in function_repr:
            return 'INIT_NORMAL'
        elif 'init_ops.TruncatedNormal' in function_repr:
            return 'INIT_NORMAL_TRUNCATED'
        elif 'he_normal' in function_repr or 'init_ops.VarianceScaling' in function_repr:
            if initializer.distribution == 'uniform':
                # He uniform
                return 'INIT_HE_UNIFORM'
            else:
                # He normal
                return 'INIT_HE_NORMAL'
        else:
            return 'INIT_UNKNOWN'

    @staticmethod
    def _get_optimizer_repr(optimizer):
        class_repr = str(optimizer)
        if 'GradientDescentOptimizer' in class_repr:
            return 'OPTIM_GRAD_DESCENT'
        elif 'AdamOptimizer' in class_repr:
            return 'OPTIM_ADAM'
        elif 'MomentumOptimizer' in class_repr:
            if optimizer._use_nesterov:
                return 'OPTIM_NESTEROV'
            else:
                return 'OPTIM_MOMENTUM'
        elif 'AdaDelta' in class_repr:
            return 'OPTIM_ADADELTA'
        else:
            return 'OPTIM_UNKNOWN'

    @staticmethod
    def _get_activation_repr(activation):
        function_repr = str(activation)
        if 'leaky_relu' in function_repr:
            return 'ACTIVATION_LEAKY_RELU'
        elif 'elu' in function_repr:
            return 'ACTIVATION_ELU'
        else:
            return 'ACTIVATION_UNKNOWN'

    @staticmethod
    def _get_transfer_learning_type_repr(is_fixed_feature_extractor):
        if is_fixed_feature_extractor:
            return 'FFE_TRUE'
        else:
            return 'FFE_FALSE'

    def __repr__(self):
        inception_v3_estimator_repr = '%s,%s,%s,%s,TRAIN_BATCH_SIZE__%d' % (
            self._get_transfer_learning_type_repr(self.is_fixed_feature_extractor),
            self._get_initializer_repr(self.initializer),
            self._get_optimizer_repr(self.optimizer),
            self._get_activation_repr(self.activation),
            self.train_batch_size
        )
        return inception_v3_estimator_repr
