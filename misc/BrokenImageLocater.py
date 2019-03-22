import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
import os
from collections import OrderedDict
import json
import math


# class DebugEstimator(tf.keras.Model):
#
#     def __init__(self):
#         super(tf.keras.Model, self).__init__(name='debug_estimator')
#         a = Input(shape=(299, 299, 3))
#         b = Dense(32)(a)
#         self._keras_model = Model(inputs=a, outputs=b)
#
#     def call(self, inputs, training=None, mask=None):
#         return self._keras_model(inputs=inputs)

class BrokenImageLocator:
    def __init__(self, image_root_dir, blacklisted_image_file_export_dir, accepted_extensions, batch_size):
        self.image_root_dir = image_root_dir
        self.image_blacklist_dir = blacklisted_image_file_export_dir
        self.accepted_extensions = accepted_extensions
        self.batch_size = batch_size


    def _get_tf_data_generator(self, image_file_paths):
        def _preprocess_image(image, height=299, width=299, num_channels=3):
            try:
                image = tf.image.decode_jpeg(image, channels=num_channels)
            except tf.errors.InvalidArgumentError as err:
                tf.logging.error(msg='Could not decode image: \'%s\'' % image)
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
        image_path_ds = tf.data.Dataset.zip((image_ds, path_ds))
        ds = image_path_ds.cache()
        ds = ds.batch(batch_size=self.batch_size)
        ds = ds.repeat()
        ds = ds.prefetch(tf.contrib.data.AUTOTUNE)
        return ds

    def _get_image_file_paths(self):
        file_paths = []
        sub_dirs = sorted(x[0] for x in tf.gfile.Walk(self.image_root_dir))
        for i, sub_dir in enumerate(sub_dirs):
            file_list = []
            dir_name = os.path.basename(sub_dir)
            if i == 0:
                # skip root dir
                continue
            tf.logging.info(msg='Locating images in: \'%s\'' % dir_name)
            for extension in self.accepted_extensions:
                file_glob = os.path.join(self.image_root_dir, dir_name, '*.' + extension)
                file_list.extend(tf.gfile.Glob(file_glob))
            if not file_list:
                tf.logging.info(msg='\tNo files found in \'%s\'. Class label omitted from data sets.' % dir_name)
            for file in file_list:
                file_paths.append(file)
        return file_paths

    def _create_data_generator(self):
        img_file_paths = self._get_image_file_paths()
        self.num_images = len(img_file_paths)
        tf_data_set = self._get_tf_data_generator(image_file_paths=img_file_paths)
        return tf_data_set

    def test_run_generator(self):
        ds = self._create_data_generator()
        iterator = ds.make_one_shot_iterator()
        image, path = iterator.get_next()
        with tf.Session() as sess:
            # sess.run(iterator.initializer)
            while True:
                try:
                    sess.run([image, path])
                except tf.errors.InvalidArgumentError as err:
                    tf.logging.warning(msg='Could not open image: \'%s\'. Encountered error: %s' % (path, err))

        # a = Input(shape=(299, 299, 3))
        # b = Dense(32)(a)
        # model = Model(inputs=a, outputs=b)
        # model.compile(optimizer='adam', loss='mean_squared_error')
        #
        # steps_per_epoch = math.ceil(self.num_images/self.batch_size)
        # # ds.make_one_shot_iterator()
        # try:
        #     model.fit(ds.make_one_shot_iterator(), steps_per_epoch=steps_per_epoch)
        # except tf.errors.InvalidArgumentError as err:
        #     tf.logging.warning(msg='Could not open image! Enountered err: %s' % err)




if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    image_root_dir = 'C:\\Users\\ccamp\Documents\\GitHub\\HerbariumDeep\\data\\SERNEC\\images'
    blacklisted_image_file_export_dir = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\misc'
    # image_root_dir = 'D:\\data\\SERNEC\\images\\'
    broken_image_locator = BrokenImageLocator(
        image_root_dir=image_root_dir,
        blacklisted_image_file_export_dir=blacklisted_image_file_export_dir,
        accepted_extensions=['jpg', 'jpeg'],
        batch_size=20
    )
    broken_image_locator.test_run_generator()

# # image_dir = image_path = 'C:\\Users\\ccamp\Documents\\GitHub\\HerbariumDeep\\data\\SERNEC\\images'
# image_dir = 'D:\\data\\SERNEC\\images\\'
# # image_dir = 'D:\\data\\SERNEC\\images\\Echinochloa muricata var. microstachya'
# image_blacklist_dir = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\misc'
# accepted_extensions = ['jpg', 'jpeg']
#
# tf.logging.set_verbosity(tf.logging.INFO)
#
# sub_dirs = sorted(x[0] for x in tf.gfile.Walk(image_dir))
#
# # Last tested sub-directory:
# last_tested_dir = 'D:\\data\\SERNEC\\images\\Echinochloa muricata'
#
# resume_index = sub_dirs.index(last_tested_dir)
# tf.logging.info('Resuming at user specified directory: \'%s\'' % last_tested_dir)
# sub_dirs = sub_dirs[resume_index:]
#
# file_paths = []
#
# for i, sub_dir in enumerate(sub_dirs):
#     file_list = []
#     dir_name = os.path.basename(sub_dir)
#     if i == 0:
#         # skip root dir
#         continue
#     tf.logging.info(msg='Locating images in: \'%s\'' % dir_name)
#     for extension in accepted_extensions:
#         file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
#         file_list.extend(tf.gfile.Glob(file_glob))
#     if not file_list:
#         tf.logging.info(msg='\tNo files found in \'%s\'. Class label omitted from data sets.' % dir_name)
#     label_name = dir_name.lower()
#     for file in file_list:
#         file_paths.append(file)
#
# blacklist = OrderedDict()
#
# with tf.Graph().as_default():
#     file_name_tensor = tf.placeholder(tf.string, shape=())
#     # tf.logging.info(msg='Opening \'%s\'' % file_name_tensor)
#     image_contents = tf.read_file(file_name_tensor)
#     image = tf.image.decode_jpeg(image_contents, channels=3)
#     init_op = tf.tables_initializer()
#
#     with tf.Session() as sess:
#         sess.run(init_op)
#         report_freq = 100
#         for i, file in enumerate(file_paths):
#             species_dir = os.path.abspath(os.path.join(file, os.pardir))
#             species = species_dir.split('\\')[-1]
#             if i % report_freq == 0:
#                 tf.logging.info(msg='BlackList size: %d' % len(blacklist))
#             tf.logging.info(msg='Opening [%d/%d] \'%s\'' % (i+1, len(file_paths), file))
#             try:
#                 image_result_tensor = sess.run(image, feed_dict={file_name_tensor: file})
#             except tf.errors.InvalidArgumentError as err:
#                 if species not in blacklist:
#                     blacklist[species] = [file]
#                 else:
#                     blacklist[species].append(file)
#                 tf.logging.warning(msg='Failed to JPEG decode image: \'%s\'. Blacklisted image.' % file)
#
# with open(os.path.join(image_blacklist_dir, 'blacklist.json'), 'w') as fp:
#     json.dump(blacklist, fp)
# tf.logging.info(msg='Finished running decoding checks. Blacklist exported to: \'%s\'' % image_blacklist_dir)


# Class label to resume at: 'Echinochloa muricata variety microstachya' at 'D:\data\SERNEC\images\Echinochloa muricata variety microstachya\1199967.jpg'

# TODO:
# ON RESUME: Use the keras branch to autotune maximum forward propagation for image decoding errors.
