import tensorflow as tf
import time
import math
from frameworks.DataAcquisition.BottleneckExecutor import BottleneckExecutor
AUTOTUNE = tf.contrib.data.AUTOTUNE

BATCH_SIZE = 20


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
    all_image_paths = bottlenecks['path'].values
    all_image_labels = bottlenecks['class'].values
    label_to_index = dict((name, index) for index, name in enumerate(bottlenecks['class'].unique()))
    all_image_labels_one_hot = [label_to_index[label] for label in all_image_labels]
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels_one_hot, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    steps_per_epoch=math.ceil(len(all_image_paths)/BATCH_SIZE)
    # Setting a shuffle buffer size as large as the dataset ensures that the data is completely shuffled:
    # ds = image_label_ds.shuffle(buffer_size=image_count)
    # ds = ds.repeat()
    # ds = ds.batch(BATCH_SIZE)
    ds = image_label_ds.cache()
    ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
    ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    # 'prefetch' lets the dataset fetch batches, in the background while the model is training.
    # ds = ds.prefetch(buffer_size=AUTOTUNE)
    time_shuffle_and_repeat(ds, batches=2*steps_per_epoch+1)


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
    tf.enable_eager_execution()
    main(run_config=run_configs['BOON'])
