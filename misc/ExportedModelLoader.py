import os
import numpy as np
import tensorflow as tf


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def read_tensor_from_image_file(file_name, input_height=299, input_width=299, input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)
    return result


class TFHClassifier:
    relative_path = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\TensorFlow\\TFHub\\tmp\\summaries'
    model_path = os.path.join(relative_path, 'trained_model')


    def __init__(self, model_path, model_label_file_path):
        self.model_path = model_path
        self.model_label_file_path = model_label_file_path
        self.img_input_height = 299
        self.img_input_width = 299
        self.img_input_mean = 0
        self.img_input_std = 255

    def classify_image(self, image_path):
        image_tensor = read_tensor_from_image_file(file_name=image_path, input_height=self.img_input_height, input_width=self.img_input_width, input_mean=self.img_input_mean, input_std=self.img_input_std)
        final_tensor_name = 'y_proba'
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], self.model_path)
            input_operation = sess.graph.get_operation_by_name('source_model/resized_input')
            output_operation = sess.graph.get_operation_by_name('eval_graph/retrain_ops/final_retrain_ops/y_proba')
            results = sess.run(output_operation.outputs[0], feed_dict={
                input_operation.outputs[0]: image_tensor
            })
        results = np.squeeze(results)
        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(self.model_label_file_path)
        for i in top_k:
            tf.logging.info('label: %s, %.2f%%' % (labels[i], results[i] * 100))
        return labels, results

if __name__ == '__main__':
    model_label_file_path = os.path.join('C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\TensorFlow\\TFHub\\tmp', 'class_labels.txt')
    tfh_classifier = TFHClassifier(
        model_path='C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\TensorFlow\\TFHub\\tmp\\summaries\\trained_model',
        model_label_file_path=model_label_file_path
    )
    # labels, results = tfh_classifier.classify_image(image_path='C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\GoingDeeper\\images\\aconitum napellus l\\1441326927238znOeMSdai1MWUE2N.jpg')
    # labels, results = tfh_classifier.classify_image(image_path='C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\GoingDeeper\\images\\aconitum napellus l\\1441326927238znOeMSdai1MWUE2N.JPG')
    labels, results = tfh_classifier.classify_image(image_path="C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\GoingDeeper\\images\\aconitum napellus l\\1441326927238znOeMSdai1MWUE2N.jpg")
    print('labels: %s' % labels)
    print('y_proba: %s' % results)
