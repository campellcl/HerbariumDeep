import tensorflow as tf
import os

# image_dir = image_path = 'C:\\Users\\ccamp\Documents\\GitHub\\HerbariumDeep\\data\\SERNEC\\images'
image_dir = 'D:\\data\\SERNEC\\images'
accepted_extensions = ['jpg', 'jpeg']

sub_dirs = sorted(x[0] for x in tf.gfile.Walk(image_dir))
file_paths = []

for i, sub_dir in enumerate(sub_dirs):
    file_list = []
    dir_name = os.path.basename(sub_dir)
    if i == 0:
        # skip root dir
        continue
    tf.logging.info('Locating images in: \'%s\'' % dir_name)
    for extension in accepted_extensions:
        file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
        file_list.extend(tf.gfile.Glob(file_glob))
    if not file_list:
        tf.logging.info(msg='\tNo files found in \'%s\'. Class label omitted from data sets.' % dir_name)
    label_name = dir_name.lower()
    for file in file_list:
        file_paths.append(file)

tf.logging.set_verbosity(tf.logging.INFO)

with tf.Graph().as_default():
    file_name_tensor = tf.placeholder(tf.string, shape=())
    # tf.logging.info(msg='Opening \'%s\'' % file_name_tensor)
    image_contents = tf.read_file(file_name_tensor)
    try:
        image = tf.image.decode_jpeg(image_contents, channels=3)
    except tf.errors.InvalidArgumentError:
        tf.logging.warning(msg='Blacklisting \'%s\'' % file)
    init_op = tf.tables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        for i, file in enumerate(file_paths):
            tf.logging.info(msg='Opening [%d/%d] \'%s\'' % (i, len(file_paths), file))
            image_result_tensor = sess.run(image, feed_dict={file_name_tensor: file})

# Class label to resume at: 'Echinochloa muricata variety microstachya' at 'D:\data\SERNEC\images\Echinochloa muricata variety microstachya\1199967.jpg'
