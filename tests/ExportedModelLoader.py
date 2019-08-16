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
    sess.close()
    return result


class TrainedTFHClassifier:
    relative_path = 'C:\\tmp'
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


def main(run_config):
    model_path = 'C:\\tmp\\summaries\\gs_winner\\train'
    model_path = os.path.join(model_path, os.listdir(model_path)[0])
    model_path = os.path.join(model_path, 'trained_model')
    model_label_file_path = os.path.join(model_path, 'class_labels.txt')
    tfh_classifier = TrainedTFHClassifier(
        model_path=os.path.join(model_path, 'inference'),
        model_label_file_path=model_label_file_path
    )
    ''' For BOONE trained classifiers: '''
    labels, results = tfh_classifier.classify_image(image_path="D:\\data\\BOON\\images\\Adiantum pedatum\\o3nDj8h7LZbGBYp874WCuB.jpeg")
    print('True Sample Class Label: \'Adiantum pedatum\'')
    print('Predicted Class Label: %s (%.2f%%)' % (labels[np.argmax(results)], results[np.argmax(results)]*100))

    labels, results = tfh_classifier.classify_image(image_path='D:\\data\\BOON\\images\\Solidago bicolor\\fToebpoXH3t8NeXta94qyf.jpeg')
    print('True Sample Class Label: \'Solidago bicolor\'')
    print('Predicted Class Label: %s (%.2f%%)' % (labels[np.argmax(results)], results[np.argmax(results)]*100))

    labels, results = tfh_classifier.classify_image(image_path='D:\\data\\BOON\\images\\Trillium undulatum\\kdtEranKBEGLnmYXBuviLP.jpeg')
    print('True Sample Class Label: \'Trillium undulatum\'')
    print('Predicted Class Label: %s (%.2f%%)' % (labels[np.argmax(results)], results[np.argmax(results)]*100))

    # print('labels: %s' % labels)
    # print('y_proba: %s' % results)
    # print('sanity check (sum y_proba): %s' % sum(results))

    ''' For GoingDeeper trained classifiers: '''
    # labels, results = tfh_classifier.classify_image(image_path="C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\GoingDeeper\\images\\aconitum napellus l\\1441326927238znOeMSdai1MWUE2N.jpg")
    # print('True Sample Class Label: \'aconitum napellus l\'')
    # print('Predicted Class Label: %s (%.2f%%)' % (labels[np.argmax(results)], results[np.argmax(results)]*100))

    # labels, results = tfh_classifier.classify_image(image_path="C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\GoingDeeper\\images\\anemone coronaria l\\1441326217337cmDHzGcGpgMbbZ1H.jpg")
    # print('True Sample Class Label: \'anemone coronaria l\'')
    # print('Predicted Class Label: %s (%.2f%%)' % (labels[np.argmax(results)], results[np.argmax(results)]*100))
    
    # labels, results = tfh_classifier.classify_image(image_path="C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\GoingDeeper\\images\\ajuga iva (l.) schreb\\1441351317032Va9XmHMbLugbA56p.jpg")
    # print('True Sample Class Label: \'ajuga iva\'')
    # print('Predicted Class Label: %s (%.2f%%)' % (labels[np.argmax(results)], results[np.argmax(results)]*100))


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
        'SERNEC': {
            'dataset': 'SERNEC',
            'image_dir': 'D:\\data\\SERNEC\\images',
            'bottleneck_path': 'D:\\data\\SERNEC\\bottlenecks.pkl',
            'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeepKeras\\frameworks\\DataAcquisition\\CleaningResults\\SERNEC'
        }
    }
    main(run_configs['BOON'])
