import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from frameworks.DataAcquisition.BottleneckExecutor import BottleneckExecutor
import numpy as np


def main(run_config):
    num_params = 100

    train_from_bottlenecks = True
    activations = ['elu', 'relu', 'tanh']

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

    for i in range(num_params):
        tf.logging.warning('Cleared Keras\' back-end session.')
        tf.keras.backend.clear_session()
        current_activation = activations[i % len(activations)]
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
