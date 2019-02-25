import tensorflow as tf
import time
import math
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import metrics
from frameworks.DataAcquisition.BottleneckExecutor import BottleneckExecutor
AUTOTUNE = tf.contrib.data.AUTOTUNE

BATCH_SIZE = 20

"""
Source: https://keras.io/applications/#usage-examples-for-image-classification-models
Source: https://www.tensorflow.org/tutorials/load_data/images#basic_methods_for_training
"""


def preprocess_image(image, height=299, width=299, num_channels=3):
    image = tf.image.decode_jpeg(image, channels=num_channels)
    image = tf.image.resize_images(image, [height, width])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


def time_shuffle_and_repeat(ds, batches):
    overall_start = time.time()
    # Fetch a single batch to prime the pipeline (fill the shuffle buffer), before starting the timer:
    it = iter(ds.take(batches+1))
    next(it)

    start = time.time()
    for i, (images, labels) in enumerate(it):
        if i % 10 == 0:
            print('.', end='')
    print()
    end = time.time()
    duration = end - start
    print("{} batches: {} s".format(batches, duration))
    print("{:0.5f} Images/s".format(BATCH_SIZE*batches/duration))
    print("Total time: {}s".format(end - overall_start))


def main(run_config):
    bottleneck_executor = BottleneckExecutor(
        image_dir=run_config['image_dir'],
        tfhub_module_url='https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1',
        compressed_bottleneck_file_path=run_config['bottleneck_path']
    )
    bottlenecks = bottleneck_executor.get_bottlenecks()
    image_count = bottlenecks.shape[0]
    num_classes = len(bottlenecks['class'].unique())
    all_image_paths = bottlenecks['path'].values
    all_image_labels = bottlenecks['class'].values
    label_to_index = dict((name, index) for index, name in enumerate(bottlenecks['class'].unique()))
    all_image_labels_one_hot = [label_to_index[label] for label in all_image_labels]
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels_one_hot, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    steps_per_epoch = math.ceil(len(all_image_paths)/BATCH_SIZE)
    ds = image_label_ds.cache()
    ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
    ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    # time_shuffle_and_repeat(ds, batches=2*steps_per_epoch+1)
    ''' Fixed-Feature Extractor InceptionV3 with Transfer Learning:'''
    base_model = InceptionV3(include_top=False, weights=None, input_shape=(299, 299, 3))
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(num_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=[metrics.categorical_accuracy])

    # train the model on the new data for a few epochs
    model.fit(ds, epochs=10, steps_per_epoch=steps_per_epoch)

    # return the loss value and metric values for the model in test mode:
    # print(model.evaluate(ds, batch_size=BATCH_SIZE, steps=steps_per_epoch))


if __name__ == '__main__':
    run_configs = {
        'debug': {
            'image_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\GoingDeeper\\images',
            'bottleneck_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\TensorFlow\\TFHub\\bottlenecks.pkl'
        },
        'BOON': {
            'image_dir': 'D:\\data\\BOON\\images',
            'bottleneck_path': 'D:\\data\\BOON\\bottlenecks.pkl'
        },
        'GoingDeeper': {
            'image_dir': 'D:\\data\\GoingDeeperData\\images',
            'bottleneck_path': 'D:\\data\\GoingDeeperData\\bottlenecks.pkl'
        },
        'SERNEC': {
            'image_dir': 'D:\\data\\SERNEC\\images',
            'bottleneck_path': 'D:\\data\\SERNEC\\bottlenecks.pkl'
        }
    }
    print('TensorFlow Version: ', tf.VERSION)
    print('TensorFlow.Keras Version: ', tf.keras.__version__)
    main(run_config=run_configs['BOON'])
