import os
import numpy as np
import tensorflow as tf
from frameworks.DataAcquisition.BottleneckExecutor import BottleneckExecutor
from frameworks.DataAcquisition.ImageExecutor import ImageExecutor


class TrainedTFHClassifier:
    model_path = None

    def __init__(self, model_path, model_label_file_path):
        self.model_path = model_path
        self.model_label_file_path = model_label_file_path
        self.img_input_height = 299
        self.img_input_width = 299
        self.img_input_mean = 0
        self.img_input_std = 255
        self.session = None
        self.input_raw_image_op = None    # Input operation for the network takes an image tensor as input.
        self.output_op = None   # Output operation for the network holds the output y_pred.

    def load_model(self):
        self.session = tf.Session(graph=tf.Graph())
        with self.session as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], self.model_path)
            self.input_raw_image_op = sess.graph.get_operation_by_name('source_model/resized_input')
            self.input_bottleneck_op = sess.graph.get_operation_by_name('source_model/pre_trained_hub_module/inception_v3_hub_apply_default/hub_output/feature_vector/SpatialSqueeze')
            self.input_bottleneck_tensor = sess.graph.get_tensor_by_name('source_model/pre_trained_hub_module/inception_v3_hub_apply_default/hub_output/feature_vector/SpatialSqueeze:0')
            self.output_op = sess.graph.get_operation_by_name('eval_graph/retrain_ops/final_retrain_ops/y_proba')

    # def calculate_class_top_1_acc(self, bottleneck_values, bottleneck_ground_truth_indices, bottleneck_ground_truth_labels, dataset='test'):
    #     total_num_unique_classes = len(bottleneck_ground_truth_indices)
    #
    #     for i, ground_truth_index in enumerate(bottleneck_ground_truth_indices):
    #         pass
    #     raise NotImplementedError

    def calculate_class_top_1_accuracies(self, bottlenecks, class_labels):
        class_top_1_accuracies = {}
        for i, class_label in enumerate(class_labels):
            print('Computing accuracy for class \'%s (%d)\':' % (class_label, i))
            # Subset the dataframe by the class label:
            bottlenecks_class_subset = bottlenecks[bottlenecks['class'] == class_label]
            total_num_class_samples = bottlenecks_class_subset.shape[0]

            bottleneck_subset_values = bottlenecks_class_subset['bottleneck'].tolist()
            bottleneck_subset_values = np.array(bottleneck_subset_values)
            # All class samples have the same ground truth index:
            bottleneck_subset_ground_truth_indices = np.array([j for j in range(total_num_class_samples)])

            # Pass all class bottlenecks through at once:
            with tf.Session(graph=tf.Graph()) as sess:
                tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], self.model_path)
                input_raw_image_op = sess.graph.get_operation_by_name('source_model/resized_input')
                input_bottleneck_op = sess.graph.get_operation_by_name('source_model/pre_trained_hub_module/inception_v3_hub_apply_default/hub_output/feature_vector/SpatialSqueeze')
                input_bottleneck_tensor = sess.graph.get_tensor_by_name('source_model/pre_trained_hub_module/inception_v3_hub_apply_default/hub_output/feature_vector/SpatialSqueeze:0')
                output_op = sess.graph.get_operation_by_name('eval_graph/retrain_ops/final_retrain_ops/y_proba')
                class_results = sess.run(output_op.outputs[0], feed_dict={
                    input_bottleneck_op.outputs[0]: bottleneck_subset_values
                })
            class_samples_correct = 0
            class_samples_incorrect = 0
            # print('class_results pre-squeeze shape: %s' % (class_results.shape,))
            class_results = np.squeeze(class_results)
            # print('class_results shape post-squeeze: %s' % (class_results.shape,))
            for j, class_result in enumerate(class_results):
                # Obtain the indices corresponding to a sorting of y_proba in ascending order:
                pred_class_index = class_result.argsort()[-1]       # The predicted class index is the most probable prediction for this sample (last in argsort() array).
                pred_prob = class_result[pred_class_index]          # This is the probability of belonging to the predicted class.
                pred_class_label = class_labels[pred_class_index]   # This is the predicted class label (human readable)
                ground_truth_class_label = class_label              # This is the ground truth class label.
                print('\tClass sample [%d/%d] predicted to be class: \'%s (%d)\' with %.2f%% probability. The real class was: \'%s (%d)\'' % (j+1, total_num_class_samples, pred_class_label, pred_class_index, pred_prob*100, class_label, i))
                if pred_class_index == i:
                    class_samples_correct += 1
                else:
                    class_samples_incorrect += 1
            assert class_samples_correct + class_samples_incorrect == total_num_class_samples
            print('Class \'%s (%d)\'\'s accuracy is: %.2f%%' % (class_label, i, (class_samples_correct / total_num_class_samples)*100))
            exit(0)
            for j, bottleneck in enumerate(bottleneck_subset_values):
                pass

        bottleneck_values = bottlenecks['bottleneck'].tolist()
        bottleneck_values = np.array(bottleneck_values)
        bottleneck_ground_truth_labels = bottlenecks['class'].values
        # Convert the labels into indices (one hot encoding by index):
        train_bottleneck_ground_truth_indices = np.array([class_labels.index(ground_truth_label)
                                                          for ground_truth_label in bottleneck_ground_truth_labels])


    def classify_image(self, image_path):
        raise NotImplementedError


def main(run_config):
    model_path = run_config['saved_model_path']
    model_path = os.path.join(model_path, os.listdir(model_path)[0])
    model_path = os.path.join(model_path, 'trained_model')
    model_label_file_path = os.path.join(model_path, 'class_labels.txt')

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

    test_bottleneck_values = test_bottlenecks['bottleneck'].tolist()
    test_bottleneck_values = np.array(test_bottleneck_values)
    test_bottleneck_ground_truth_labels = test_bottlenecks['class'].values
    # Convert the labels into indices (one hot encoding by index):
    test_bottleneck_ground_truth_indices = np.array([class_labels.index(ground_truth_label)
                                                    for ground_truth_label in test_bottleneck_ground_truth_labels])

    tfh_classifier = TrainedTFHClassifier(model_path=os.path.join(model_path, 'inference'), model_label_file_path=model_label_file_path)
    # tfh_classifier.load_model()
    tfh_classifier.calculate_class_top_1_accuracies(bottlenecks=test_bottlenecks, class_labels=class_labels)


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
        'BOONE': {
            'dataset': 'BOON',
            'image_dir': 'D:\\data\\BOON\\images',
            'bottleneck_path': 'D:\\data\\BOON\\bottlenecks.pkl',
            'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeepKeras\\frameworks\\DataAcquisition\\CleaningResults\\BOON',
            'saved_model_path': 'D:\\data\\BOON\\training summaries\\8-16-2019\\gs_winner\\train'
        },
        'GoingDeeper': {
            'dataset': 'GoingDeeper',
            'image_dir': 'D:\\data\\GoingDeeperData\\images',
            'bottleneck_path': 'D:\\data\\GoingDeeperData\\bottlenecks.pkl',
            'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeepKeras\\frameworks\\DataAcquisition\\CleaningResults\\GoingDeeper',
            'saved_model_path': 'D:\\data\\GoingDeeperData\\training summaries\\8-17-2019\\gs_winner\\train'
        },
        'SERNEC': {}
    }
    main(run_config=run_configs['BOONE'])
