from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from keras import backend as K
import tensorflow as tf
import numpy as np
import math
# from memory_profiler import profile

from frameworks.DataAcquisition.BottleneckExecutor import BottleneckExecutor
from frameworks.TensorFlow.Keras.Callbacks.CustomCallbacks import FileWritersTensorBoardCallback


class InceptionV3Estimator(BaseEstimator, ClassifierMixin, tf.keras.Model):

    def __init__(self, dataset, num_classes, class_labels, train_from_bottlenecks, train_batch_size=-1, val_batch_size=-1, activation=tf.nn.elu,
                 optimizer=tf.train.AdamOptimizer, initializer=tf.variance_scaling_initializer(),
                 is_fixed_feature_extractor=True, random_state=None, tb_log_dir=None, is_refit=False):
        super(InceptionV3Estimator, self).__init__(name='inception_v3_estimator')
        self.dataset = dataset
        self.num_classes = num_classes
        self.class_labels = class_labels
        self.train_from_bottlenecks = train_from_bottlenecks
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.activation = activation
        self.optimizer = optimizer
        self.initializer = initializer
        self.random_state = random_state
        self.tb_log_dir = tb_log_dir
        self._y_proba = None
        self.eval_freq = None
        self._is_trained = False

        self.is_fixed_feature_extractor = is_fixed_feature_extractor
        self._keras_model = None

        self._session = None
        self._graph = None
        # TensorBoard setup:
        self._train_writer = None
        self._val_writer = None
        self.is_refit = is_refit

        # if self._keras_model is None:
        #     self._build_model_and_graph_def()

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

    @staticmethod
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

    def _build_model_and_graph_def(self):
        if self._session is None:
            session = tf.keras.backend.get_session()
            tf.logging.info(msg='current backend session: %s' % session)
        # ON RESUME: The memory leak is happening where the TODO statement is below. Confirmed with debugger and printing growing length of
        # traininable variables in the session graph. To fix this, I will need to pull the tensor handles for this instance of the gridsearch
        # object directly from the backend session computational graph, as the backend session appears to persist across instances of this class.
        # Note that attempting to clear the backend session would require multiple graph re-instantiations. Grid search is driving however, so no
        # control over this unless I want to attempt a subclass.

        # This is a sanity check, the backend session is never None:
        if session is not None:
            # Note: backed session exists with no graph on init, so we need to actually check the graph for collections:
            if not session.graph.collections:
                # Ok the graph has no collections, go ahead and initialize:
                should_initialize_base_model_graph = True
            else:
                # The graph has collections, it was previously initialized and retained in the persistent backend sess.
                should_initialize_base_model_graph = False
        if not should_initialize_base_model_graph:
            # Have to initialize base model graph due to Graph disconnected errors preventing tensor handle retrieval.
            tf.keras.backend.clear_session()
            session = tf.keras.backend.get_session()
            tf.logging.warning(msg='Cleared Keras\' backend session to attempt to free memory.')
            ''' Get the tensor handles from the already initialized graph and return: '''
            # nodes = [node.name for node in session.graph.as_graph_def().node]
            # input_nodes = [node.name for node in session.graph.as_graph_def().node if 'input' in node.name]
            # base_model_input = session.graph.get_tensor_by_name('input_1:0')
            # x = session.graph.get_tensor_by_name('mixed10/concat:0')
            # bottlenecks = session.graph.get_tensor_by_name('bottleneck:0')
            # Now that we avoided re-initializing the base_model; initialize what actually changed this run:
            # logits = Dense(self.num_classes, activation=self.activation)(bottlenecks)
            # y_proba = Dense(self.num_classes, activation='softmax')(logits)
            # self._keras_model = Model(inputs=base_model_input, outputs=y_proba)
            # self._keras_resized_input_handle_ = base_model_input
            # self._y_proba = y_proba
            # return

        # K.clear_session()
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        if self.is_fixed_feature_extractor:
            # input_tensor = tf.placeholder(dtype=tf.float32, shape=(None, 299, 299, 3), name='resized_input_tensor')
            # base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))(input_tensor)
            # self._session = tf.Session()

            # TODO: Huge jump in memory at initialization here, contributing to memory leak.
            base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
            # print('base_model input names: %s' % base_model.input)
            # print('base_model output names: %s' % base_model.output_names)
            # base_model.name = 'base_model'

            # first: train only the top layers (which were randomly initialized)
            # i.e. freeze all convolutional InceptionV3 layers
            for layer in base_model.layers:
                layer.trainable = False

            if not self.train_from_bottlenecks:
                # add a global spatial average pooling layer:
                x = base_model.output
                # spatial = tf.keras.backend.squeeze(x, axis=0)
                bottlenecks = GlobalAveragePooling2D()(x)
                # let's add a fully-connected layer output shape 1024
                logits = Dense(self.num_classes, activation=self.activation)(bottlenecks)
                # and a fully connected logistic layer for self.num_classes
                y_proba = Dense(self.num_classes, activation='softmax')(logits)

                # This is the model that is actually trained, if raw images are being fed from drfive:
                self._keras_model = Model(inputs=base_model.input, outputs=y_proba)
                self._keras_resized_input_handle_ = self._keras_model.input
                self._y_proba = self._keras_model.output
            else:
                bottlenecks = Input(shape=(base_model.output_shape[-1],), name='bottleneck')
                # bottlenecks = Dense(self.num_classes, input_shape=(base_model.output_shape[-1],))
                logits = Dense(self.num_classes, activation=self.activation, name='logits')(bottlenecks)
                y_proba = Dense(self.num_classes, activation='softmax', name='y_proba')(logits)
                # This is the model that is actually trained, if bottlenecks are being fed from memory:
                self._keras_model = Model(inputs=bottlenecks, outputs=y_proba)
                self._keras_resized_input_handle_ = self._keras_model
                self._y_proba = self._keras_model.output
        else:
            # Not a fixed feature extractor, this logic hasn't been implemented yet for fine-tuning.
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
        # super(InceptionV3Estimator, self).call()
        return self._keras_model(inputs=inputs)

    # def compute_output_shape(self, input_shape):
    #     # TODO: Multiple inheritance issue with Keras and SKLearn super classes.
    #     #   For a possible resolution, see: https://stackoverflow.com/a/9575426/3429090
    #     super(InceptionV3Estimator, self).compute_output_shape(input_shape)

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

    def _tf_data_generator_from_disk(self, image_file_paths, image_one_hot_encoded_labels, is_training):
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

    def _fit_images(self, X_train, y_train, num_epochs, eval_freq, ckpt_freq, early_stopping_eval_freq, X_val, y_val, has_validation_data):
        """

        :param X_train:
        :param y_train:
        :param num_epochs:
        :param eval_freq:
        :param ckpt_freq:
        :param early_stopping_eval_freq:
        :param X_val:
        :param y_val:
        :param has_validation_data:
        :return:
        """
        # X_train is a list of image paths, y_train is the associated one-hot encoded labels.
        num_train_images = len(X_train)
        if self.train_batch_size == -1:
            self.train_batch_size = num_train_images

        # train_ds = self._convert_to_tensorflow_dataset(
        #     sample_image_file_paths=X_train,
        #     sample_image_one_hot_encoded_class_labels=y_train,
        #     is_train_data=True
        # )
        train_ds = self._tf_data_generator_from_disk(image_file_paths=X_train, image_one_hot_encoded_labels=y_train, is_training=True)

        if has_validation_data:
            num_val_images = len(X_val)
            if self.val_batch_size == -1:
                self.val_batch_size = num_val_images

            # val_ds = self._convert_to_tensorflow_dataset(
            #     sample_image_file_paths=X_val,
            #     sample_image_one_hot_encoded_class_labels=y_val,
            #     is_train_data=False
            # )
            val_ds = self._tf_data_generator_from_disk(image_file_paths=X_val, image_one_hot_encoded_labels=y_val, is_training=False)

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
                        FileWritersTensorBoardCallback(
                            log_dir=self.tb_log_dir, hyperparameter_string_repr=self.__repr__(),
                            write_graph=False, is_refit=self.is_refit, write_freq=self.eval_freq
                        ),
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss', min_delta=0, patience=early_stopping_eval_freq, verbose=0,
                            mode='min', baseline=None, restore_best_weights=False
                        )
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

    # @profile
    def fit(self, X_train, y_train, fed_bottlenecks=False, num_epochs=1000, eval_freq=1, ckpt_freq=0, early_stopping_eval_freq=1, X_val=None, y_val=None):
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

        self.eval_freq = eval_freq

        self._build_model_and_graph_def()

        # Was a DataFrame with pre-computed bottlenecks fed to this method from memory, or just a list of image paths?
        if not fed_bottlenecks:
            # X_train is a list of image paths, y_train is the associated one-hot encoded labels.
            self._fit_images(
                X_train=X_train, y_train=y_train, num_epochs=num_epochs, eval_freq=eval_freq, ckpt_freq=ckpt_freq,
                early_stopping_eval_freq=early_stopping_eval_freq, X_val=X_val, y_val=y_val,
                has_validation_data=has_validation_data
            )
        else:
            # X_train is an array of bottlenecks, y_train is the associated one-hot encoded labels.
            num_train_bottlenecks = len(X_train)
            if self.train_batch_size == -1:
                self.train_batch_size = num_train_bottlenecks
            steps_per_epoch = math.ceil(num_train_bottlenecks/self.train_batch_size)

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
            train_ds = self._tf_data_generator_from_memory(image_bottlenecks=X_train, image_encoded_labels=y_train, is_training=True)

            if has_validation_data:
                val_ds = self._tf_data_generator_from_memory(image_bottlenecks=X_val, image_encoded_labels=y_val, is_training=False)

                self._keras_model.fit(
                    train_ds.make_one_shot_iterator(),
                    validation_data=val_ds.make_one_shot_iterator(),
                    epochs=num_epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=val_steps_per_epoch,
                    callbacks=[
                        FileWritersTensorBoardCallback(
                            log_dir=self.tb_log_dir, hyperparameter_string_repr=self.__repr__(),
                            write_graph=False, is_refit=self.is_refit, write_freq=self.eval_freq
                        ),
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss', min_delta=0, patience=early_stopping_eval_freq, verbose=0,
                            mode='min', baseline=None, restore_best_weights=False
                        )
                    ]
                )

            # for batch_num, (X_batch, y_batch) in enumerate(self._shuffle_batch(X_train, y_train, batch_size=self.train_batch_size)):
            #     if self.dataset == 'SERNEC':
            #         raise NotImplementedError
            #     else:
            #         # Dataset small enough to run training summaries on entire dataset, so run train op on the minibatch level, but don't capture it:
            #         if X_val is not None:
            #             self._keras_model.fit(
            #                 X_batch,
            #                 validation_data=X_val,
            #                 epochs=num_epochs,
            #                 steps_per_epoch=steps_per_epoch,
            #                 validation_steps=val_steps_per_epoch,
            #                 callbacks=[
            #                     FileWritersTensorBoardCallback(
            #                         log_dir=self.tb_log_dir, hyperparameter_string_repr=self.__repr__(),
            #                         write_graph=False, is_refit=self.is_refit, write_freq=self.eval_freq
            #                     ),
            #                     tf.keras.callbacks.EarlyStopping(
            #                         monitor='val_loss', min_delta=0, patience=early_stopping_eval_freq, verbose=0,
            #                         mode='min', baseline=None, restore_best_weights=False
            #                     )]
            #             )

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
        self._is_trained = True
        return self

    def predict(self, X):
        print('predict called with X: %s' % (X.shape,))
        if not self._is_trained:
            class_indices = np.argmax(self.predict_proba(X, batch_size=self.train_batch_size), axis=1)
        else:
            class_indices = np.argmax(self.predict_proba(X, batch_size=self.val_batch_size), axis=1)

        # Prevent memory leaks by clearing session (see: https://stackoverflow.com/questions/50331201/memory-leak-keras-tensorflow1-8-0/50331508)
        # K.clear_session()
        return np.array(class_indices, np.int32)
        # tf.logging.error(msg='Not implemented yet.')
        # raise NotImplementedError

    def predict_proba(self, X, batch_size):
        # if not self._session:
        #     raise NotFittedError('This %s instance is not fitted yet' % self.__class__.__name__)
        if not self._is_trained:
            raise NotFittedError('This %s instance is not fitted yet' % self.__class__.__name__)
        if not self.train_from_bottlenecks:
            num_images = len(X)
            path_ds = tf.data.Dataset.from_tensor_slices(X)
            image_ds = path_ds.map(
                InceptionV3Estimator._load_and_preprocess_image,
                num_parallel_calls=tf.contrib.data.AUTOTUNE
            )
            ds = image_ds.shuffle(buffer_size=num_images)
            ds = ds.batch(batch_size=batch_size)
            ds = ds.repeat()
            ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
            steps_per_epoch = math.ceil(num_images/batch_size)
            y_proba = self._keras_model.predict(ds.make_one_shot_iterator(), batch_size=batch_size, verbose=0, steps=steps_per_epoch)
            # y_proba is a [1 x 3] vector for each image where each column is: p(class) for class in all_classes;
            # y_proba = np.argmax(y_proba, axis=1)
            # return self.call(inputs=ds, training=self._is_trained, mask=None)
            return y_proba

            # with self._session.as_default() as sess:
            #     return self._y_proba.eval(feed_dict={self._keras_resized_input_handle_: X})

            # tf.logging.error(msg='Not implemented yet')
            # raise NotImplementedError
        else:
            num_bottlenecks = len(X)
            # bottleneck_ds = tf.data.Dataset.from_tensor_slices(X)
            # ds = bottleneck_ds.shuffle(buffer_size=num_bottlenecks)
            # ds.batch(batch_size=batch_size)
            # ds = ds.repeat()
            # ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
            steps_per_epoch = math.ceil(num_bottlenecks/batch_size)
            y_proba = self._keras_model.predict(X, batch_size=batch_size, verbose=0, steps=steps_per_epoch)
            return y_proba

    def _get_model_params(self):
        with self._session.graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}
        # raise NotImplementedError

    def _restore_model_params(self, model_params):
        gvar_names = list(model_params.keys())
        assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + '/Assign') for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)

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
