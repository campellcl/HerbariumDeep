import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from frameworks.DataAcquisition.BottleneckExecutor import BottleneckExecutor
import numpy as np
import math
import gc

def _tf_data_generator_from_memory(num_classes, train_batch_size, val_batch_size, image_bottlenecks, image_encoded_labels, is_training):
    # Convert to categorical format for keras (see bottom of page: https://keras.io/losses/):
    bottleneck_ds = tf.data.Dataset.from_tensor_slices(image_bottlenecks)
    categorical_labels = to_categorical(image_encoded_labels, num_classes=num_classes)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(categorical_labels, tf.int64))
    bottleneck_label_ds = tf.data.Dataset.zip((bottleneck_ds, label_ds))
    num_images = len(image_bottlenecks)
    # 1. Cache dataset:
    ds = bottleneck_label_ds.cache()
    # 2. Shuffle entire dataset:
    ds = ds.shuffle(buffer_size=num_images)
    # 3. Apply the shuffle operation immediately:
    ds = ds.repeat()
    # 4. Partition into batches:
    if is_training:
        ds = ds.batch(batch_size=train_batch_size)
    else:
        ds = ds.batch(batch_size=val_batch_size)
    # 5. Apply the batch operation immediately:
    ds = ds.repeat()
    # 6. Allocate prefetch buffer:
    ds = ds.prefetch(tf.contrib.data.AUTOTUNE)
    return ds

def main(run_config):
    num_params = 100
    num_epochs = 100

    train_from_bottlenecks = True
    activations = ['elu', 'relu', 'tanh']
    optimizers = [tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08), tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9, use_nesterov=True)]
    train_batch_sizes = [20, 60, 100]

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
    num_classes = len(np.unique(train_bottleneck_ground_truth_labels))

    X_train, y_train = train_bottleneck_values, train_bottleneck_ground_truth_indices
    X_valid, y_valid = val_bottleneck_values, val_bottleneck_ground_truth_indices

    for i in range(num_params):
        tf.keras.backend.clear_session()
        tf.logging.warning('Cleared Keras\' back-end session.')

        gc.collect()
        tf.logging.warning('Ran garbage collector.')

        current_activation = activations[i % len(activations)]
        current_optimizer = optimizers[i % len(optimizers)]
        current_train_batch_size = train_batch_sizes[i % len(train_batch_sizes)]

        base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))

        for layer in base_model.layers:
            layer.trainable = False

        if not train_from_bottlenecks:
            x = base_model.output
            bottlenecks = GlobalAveragePooling2D()(x)
            logits = Dense(num_classes, activation=current_activation, name='logits')(bottlenecks)
            y_proba = Dense(num_classes, activation='softmax')(logits)
            _keras_model = Model(inputs=base_model.input, outputs=y_proba)
        else:
            bottlenecks = Input(shape=(base_model.output_shape[-1],), name='bottleneck')
            # bottlenecks = Dense(self.num_classes, input_shape=(base_model.output_shape[-1],))
            logits = Dense(num_classes, activation=current_activation, name='logits')(bottlenecks)
            y_proba = Dense(num_classes, activation='softmax', name='y_proba')(logits)
            # This is the model that is actually trained, if bottlenecks are being fed from memory:
            _keras_model = Model(inputs=bottlenecks, outputs=y_proba)

        _keras_model.compile(
            optimizer=current_optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        tf.logging.info(msg='Compiled Keras model.')
        train_ds = _tf_data_generator_from_memory(image_bottlenecks=X_train, image_encoded_labels=y_train, is_training=True, num_classes=num_classes, train_batch_size=current_train_batch_size, val_batch_size=num_val_samples)
        val_ds = _tf_data_generator_from_memory(image_bottlenecks=X_valid, image_encoded_labels=y_valid, is_training=False, num_classes=num_classes, train_batch_size=current_train_batch_size, val_batch_size=num_val_samples)

        train_steps_per_epoch = math.ceil(num_train_samples/current_train_batch_size)
        val_steps_per_epoch = math.ceil(num_val_samples/num_val_samples)

        _keras_model.fit(
            train_ds.make_one_shot_iterator(),
            validation_data=val_ds.make_one_shot_iterator(),
            epochs=num_epochs,
            steps_per_epoch=train_steps_per_epoch,
            validation_steps=val_steps_per_epoch,
            callbacks=[]
        )


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
