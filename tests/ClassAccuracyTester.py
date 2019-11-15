import os
import numpy as np
import tensorflow as tf
from frameworks.DataAcquisition.BottleneckExecutor import BottleneckExecutor
import json
from frameworks.DataAcquisition.ImageExecutor import ImageExecutor
from sklearn.metrics import confusion_matrix


class TrainedTFHClassifier:
    preceding_model_path = None

    def __init__(self, preceding_model_path, preceding_model_label_file_path, current_model_label_file_path):
        self.preceding_model_path = preceding_model_path
        self.current_model_label_file_path = current_model_label_file_path
        self.preceding_model_label_file_path = preceding_model_label_file_path
        self.img_input_height = 299
        self.img_input_width = 299
        self.img_input_mean = 0
        self.img_input_std = 255
        self.session = None
        self.input_raw_image_op = None    # Input operation for the network takes an image tensor as input.
        self.output_op = None   # Output operation for the network holds the output y_pred.

    def load_preceding_model(self):
        self.session = tf.Session(graph=tf.Graph())
        with self.session as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], self.preceding_model_path)
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

    def calculate_class_top_1_accuracies(self, current_process_bottlenecks, class_labels):
        class_top_1_accuracies = {}
        # Pass all class bottlenecks through at once:
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], self.preceding_model_path)
            input_raw_image_op = sess.graph.get_operation_by_name('source_model/resized_input')
            input_bottleneck_op = sess.graph.get_operation_by_name('source_model/pre_trained_hub_module/inception_v3_hub_apply_default/hub_output/feature_vector/SpatialSqueeze')
            input_bottleneck_tensor = sess.graph.get_tensor_by_name('source_model/pre_trained_hub_module/inception_v3_hub_apply_default/hub_output/feature_vector/SpatialSqueeze:0')
            output_op = sess.graph.get_operation_by_name('eval_graph/retrain_ops/final_retrain_ops/y_proba')

            for i, class_label in enumerate(class_labels):
                print('Computing accuracy for class \'%s (%d)\':' % (class_label, i))
                # Subset the dataframe by the class label:
                bottlenecks_class_subset = current_process_bottlenecks[current_process_bottlenecks['class'] == class_label]
                total_num_class_samples = bottlenecks_class_subset.shape[0]

                bottleneck_subset_values = bottlenecks_class_subset['bottleneck'].tolist()
                bottleneck_subset_values = np.array(bottleneck_subset_values)
                # All class samples have the same ground truth index:
                bottleneck_subset_ground_truth_indices = np.array([j for j in range(total_num_class_samples)])

                if total_num_class_samples == 0:
                    # There are no class samples in the chosen dataset's partition of (train, test, val):
                    print('\tWARNING: There are no samples for class \'%s\' in the chosen dataset partition of (train/val/test)' % class_label)
                    print('Class \'%s\' (%d)\'s accuracy is UNDEFINED' % (class_label, i))
                    class_top_1_accuracies[i] = {'class': class_label, 'top_1_acc': np.NaN}
                    continue
                # Run the bottleneck subset through the computational graph:
                class_results = sess.run(output_op.outputs[0], feed_dict={
                    input_bottleneck_op.outputs[0]: bottleneck_subset_values
                })

                class_samples_correct = 0
                class_samples_incorrect = 0
                # print('class_results pre-squeeze shape: %s' % (class_results.shape,))
                class_results = np.squeeze(class_results)
                # print('class_results shape post-squeeze: %s' % (class_results.shape,))

                # This is for single samples only in the validation set:
                if class_results.ndim == 1:
                    pred_class_index = class_results.argsort()[-1]      # The predicted class index is the most probable prediction for this sample (last in argsort() array).
                    pred_prob = class_results[pred_class_index]         # This is the probability of belonging to the predicted class.
                    pred_class_label = class_labels[pred_class_index]   # This is the predicted class label (human readable)
                    ground_truth_class_label = class_label              # This is the ground truth class label.
                    print('\tClass sample [%d/%d] predicted to be class: \'%s (%d)\' with %.2f%% probability. The real class was: \'%s (%d)\'' % (j+1, total_num_class_samples, pred_class_label, pred_class_index, pred_prob*100, class_label, i))
                    if pred_class_index == i:
                        class_samples_correct += 1
                    else:
                        class_samples_incorrect += 1
                    print('Class \'%s (%d)\'\'s accuracy is: %.2f%%' % (class_label, i, (class_samples_correct / total_num_class_samples)*100))
                    class_top_1_accuracies[i] = {'class': class_label, 'top_1_acc': (class_samples_correct / total_num_class_samples)*100}
                else:
                    # This is for multiple class samples in the validation dataset:
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
                    class_top_1_accuracies[i] = {'class': class_label, 'top_1_acc': (class_samples_correct / total_num_class_samples)*100}
        print('class_top_1_accuracies: %s' % class_top_1_accuracies)
        return class_top_1_accuracies

    def calculate_class_top_1_positive_predictive_values(self, current_process_bottlenecks, class_labels):
        # https://stackoverflow.com/a/50671617/3429090
        y_true = current_process_bottlenecks['class'].values
        y_pred = []

        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], self.preceding_model_path)
            input_raw_image_op = sess.graph.get_operation_by_name('source_model/resized_input')
            input_bottleneck_op = sess.graph.get_operation_by_name('source_model/pre_trained_hub_module/inception_v3_hub_apply_default/hub_output/feature_vector/SpatialSqueeze')
            input_bottleneck_tensor = sess.graph.get_tensor_by_name('source_model/pre_trained_hub_module/inception_v3_hub_apply_default/hub_output/feature_vector/SpatialSqueeze:0')
            output_op = sess.graph.get_operation_by_name('eval_graph/retrain_ops/final_retrain_ops/y_proba')

            print('Accumulating Prediction Counts for Each Class...')
            for i, bottleneck_sample in enumerate(current_process_bottlenecks.itertuples()):
                # print('Accumulating Positive Predictive Value (PPV) class counts for sample [%d/%d]:' % (i+1, current_process_bottlenecks.shape[0]))
                bottleneck_sample_value = bottleneck_sample.bottleneck
                ground_truth_class_label = bottleneck_sample[1]           # This is the ground truth class label.
                sample_result = sess.run(output_op.outputs[0], feed_dict={
                    input_bottleneck_op.outputs[0]: np.expand_dims(bottleneck_sample_value, axis=0)
                })
                sample_result = np.squeeze(sample_result)
                # Obtain the indices corresponding to a sorting of y_proba in ascending order:
                pred_class_index = sample_result.argsort()[-1]      # The predicted class index is the most probable prediction for this sample (last in argsort() array).
                pred_prob = sample_result[pred_class_index]         # This is the probability of belonging to the predicted class.
                pred_class_label = class_labels[pred_class_index]   # This is the predicted class label (human readable)
                y_pred.append(pred_class_label)
        assert len(y_pred) == len(y_true)
        cnf_matrix = confusion_matrix(y_true, y_pred, labels=class_labels)
        print(cnf_matrix)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP)
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        FDR = FP/(TP+FP)
        # Overall accuracy
        # ACC = (TP+TN)/(TP+FP+FN+TN)
        ACC = (TP+TN)/(TP+TN+FP+FN)
        # Per-Class Accuracy:
        per_class_acc = np.diag(cnf_matrix) / np.sum(cnf_matrix, axis=1)
        # Per-Class PPV:
        per_class_ppv = np.diag(cnf_matrix) / np.sum(cnf_matrix, axis=0)
        print('PPV: %s' % PPV)
        print('ACC: %s' % ACC)

        for i, class_label in enumerate(current_process_bottlenecks['class'].values):
            print('Top-1 Acc of Class \'%s\' (%d): %.2f%%' % (class_label, class_labels.index(class_label), per_class_acc[class_labels.index(class_label)]*100))
            print('PPV of Class \'%s\' (%d): %.2f%%' % (class_label, class_labels.index(class_label), per_class_ppv[class_labels.index(class_label)]*100))

        exit(0)

        class_positive_predictive_values = {}
        for i, class_label in enumerate(class_labels):
            df_subset = current_process_bottlenecks[current_process_bottlenecks['class'] == class_label]
            num_times_actual_label = df_subset.shape[0]
            class_positive_predictive_values[class_label] = {'class': class_label, 'num_times_predicted_label': 0, 'num_times_actual_label': num_times_actual_label}

        # Pass all class bottlenecks through at once:
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], self.preceding_model_path)
            input_raw_image_op = sess.graph.get_operation_by_name('source_model/resized_input')
            input_bottleneck_op = sess.graph.get_operation_by_name('source_model/pre_trained_hub_module/inception_v3_hub_apply_default/hub_output/feature_vector/SpatialSqueeze')
            input_bottleneck_tensor = sess.graph.get_tensor_by_name('source_model/pre_trained_hub_module/inception_v3_hub_apply_default/hub_output/feature_vector/SpatialSqueeze:0')
            output_op = sess.graph.get_operation_by_name('eval_graph/retrain_ops/final_retrain_ops/y_proba')

            print('Accumulating Prediction Counts for Each Class...')
            for i, bottleneck_sample in enumerate(current_process_bottlenecks.itertuples()):
                print('Accumulating Positive Predictive Value (PPV) class counts for sample [%d/%d]:' % (i+1, current_process_bottlenecks.shape[0]))
                bottleneck_sample_value = bottleneck_sample.bottleneck
                ground_truth_class_label = bottleneck_sample[1]           # This is the ground truth class label.
                sample_result = sess.run(output_op.outputs[0], feed_dict={
                    input_bottleneck_op.outputs[0]: np.expand_dims(bottleneck_sample_value, axis=0)
                })
                sample_result = np.squeeze(sample_result)
                # Obtain the indices corresponding to a sorting of y_proba in ascending order:
                pred_class_index = sample_result.argsort()[-1]      # The predicted class index is the most probable prediction for this sample (last in argsort() array).
                pred_prob = sample_result[pred_class_index]         # This is the probability of belonging to the predicted class.
                pred_class_label = class_labels[pred_class_index]   # This is the predicted class label (human readable)

                class_positive_predictive_values[pred_class_label]['num_times_predicted_label'] += 1

                # if ground_truth_class_label == pred_class_label:
                #     if ground_truth_class_label not in class_positive_predictive_values.keys():
                #         class_positive_predictive_values[ground_truth_class_label] = {'class': ground_truth_class_label, 'num_times_predicted_label': 1, 'num_times_actual_label': 1, 'top_1_ppv': None}
                #     else:
                #         class_positive_predictive_values[ground_truth_class_label]['num_times_predicted_label'] += 1
                #         class_positive_predictive_values[ground_truth_class_label]['num_times_actual_label'] += 1
                # else:
                #     if ground_truth_class_label not in class_positive_predictive_values.keys():
                #         class_positive_predictive_values[ground_truth_class_label] = {'class': ground_truth_class_label, 'num_times_predicted_label': 1, 'num_times_actual_label': 0, 'top_1_ppv': None}
                #     else:
                #         class_positive_predictive_values[ground_truth_class_label]['num_times_actual_label'] += 1
                #         if pred_class_label not in class_positive_predictive_values.keys():
                #             class_positive_predictive_values[pred_class_label] = {'class': pred_class_label, 'num_times_predicted_label': 1, 'num_times_actual_label': 0, 'top_1_ppv': None}
                #         else:
                #             class_positive_predictive_values[pred_class_label]['num_times_predicted_label'] += 1
            # Now we have for every class label the number of times it was predicted, and the number of times it actually was the correct label:
            for i, class_label in enumerate(class_labels):
                assert current_process_bottlenecks[current_process_bottlenecks['class'] == class_label].shape[0] == class_positive_predictive_values[class_label]['num_times_actual_label']
                class_positive_predictive_values[class_label]['top_1_ppv'] = \
                    class_positive_predictive_values[class_label]['num_times_actual_label'] \
                    / class_positive_predictive_values[class_label]['num_times_predicted_label']
                as_percent = (class_positive_predictive_values[class_label]['num_times_actual_label'] * 100) / class_positive_predictive_values[class_label]['num_times_predicted_label']
                print('Computed Positive Predictive Value (PPV) for class \'%s (%d)\': %.2f%%' % (class_label, i, as_percent))
        return class_positive_predictive_values
            # for i, class_label in enumerate(class_labels):
            #     print('Computing Positive Predictive Value (PPV) for class \'%s (%d)\':' % (class_label, i))
            #     # Subset the dataframe by the actual class label:
            #     bottlenecks_class_subset = current_process_bottlenecks[current_process_bottlenecks['class'] == class_label]
            #     total_num_class_samples = bottlenecks_class_subset.shape[0]
            #
            #     bottleneck_subset_values = bottlenecks_class_subset['bottleneck'].tolist()
            #     bottleneck_subset_values = np.array(bottleneck_subset_values)
            #     # All class samples have the same ground truth index:
            #     bottleneck_subset_ground_truth_indices = np.array([j for j in range(total_num_class_samples)])
            #
            #     if total_num_class_samples == 0:
            #         # There are no class samples in the chosen dataset's partition of (train, test, val):
            #         print('\tWARNING: There are no samples for class \'%s\' in the chosen dataset partition of (train/val/test)' % class_label)
            #         print('Class \'%s\' (%d)\'s Positive Predictive Value (PPV) is UNDEFINED' % (class_label, i))
            #         class_positive_predictive_values[i] = {'class': class_label, 'top_1_acc': np.NaN}
            #         continue
            #
            #     # Run the bottleneck subset through the computational graph:
            #     class_results = sess.run(output_op.outputs[0], feed_dict={
            #         input_bottleneck_op.outputs[0]: bottleneck_subset_values
            #     })
            #
            #     class_samples_correct = 0
            #     class_samples_incorrect = 0
            #     class_results = np.squeeze(class_results)
            #
            #     # If there is only a single sample in the validation set:
            #     if class_results.ndim == 1:
            #         raise NotImplementedError("TODO: fix this.")
            #     else:
            #         # There are multiple (one or more) samples of the corresponding class in the current dataset:
            #         for j, class_result in enumerate(class_results):
            #             # Obtain the indices corresponding to a sorting of y_proba in ascending order:
            #             pred_class_index = class_result.argsort()[-1]       # The predicted class index is the most probable prediction for this sample (last in argsort() array).
            #             pred_prob = class_result[pred_class_index]          # This is the probability of belonging to the predicted class.
            #             pred_class_label = class_labels[pred_class_index]   # This is the predicted class label (human readable)
            #             ground_truth_class_label = class_label              # This is the ground truth class label.
            #
            #             if ground_truth_class_label == pred_class_label:
            #                 if ground_truth_class_label not in class_positive_predictive_values.values():
            #                     top_1_positive_predictive_value = total_num_class_samples
            #                     class_positive_predictive_values[ground_truth_class_label] = {'class': ground_truth_class_label, 'top_1_ppv': 1}
            #             print('\tClass sample [%d/%d] predicted to be class: \'%s (%d)\' with %.2f%% probability. The real class was: \'%s (%d)\'' % (j+1, total_num_class_samples, pred_class_label, pred_class_index, pred_prob*100, class_label, i))
            #             if pred_class_index == i:
            #                 class_samples_correct += 1
            #             else:
            #                 class_samples_incorrect += 1
            #         # How many samples did we predict belonged to this class:
            #         # class_samples_correct?
            #         assert class_samples_correct + class_samples_incorrect == total_num_class_samples


    def classify_image(self, image_path):
        raise NotImplementedError

    def calculate_class_top_5_accuracies(self, bottlenecks, class_labels):
        class_top_5_accuracies = {}
        # Pass all class bottlenecks through at once:
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], self.preceding_model_path)
            input_raw_image_op = sess.graph.get_operation_by_name('source_model/resized_input')
            input_bottleneck_op = sess.graph.get_operation_by_name('source_model/pre_trained_hub_module/inception_v3_hub_apply_default/hub_output/feature_vector/SpatialSqueeze')
            input_bottleneck_tensor = sess.graph.get_tensor_by_name('source_model/pre_trained_hub_module/inception_v3_hub_apply_default/hub_output/feature_vector/SpatialSqueeze:0')
            output_op = sess.graph.get_operation_by_name('eval_graph/retrain_ops/final_retrain_ops/y_proba')

            for i, class_label in enumerate(class_labels):
                print('Computing top-5 accuracy for class \'%s (%d)\':' % (class_label, i))
                # Subset the dataframe by the class label:
                bottlenecks_class_subset = bottlenecks[bottlenecks['class'] == class_label]
                total_num_class_samples = bottlenecks_class_subset.shape[0]

                bottleneck_subset_values = bottlenecks_class_subset['bottleneck'].tolist()
                bottleneck_subset_values = np.array(bottleneck_subset_values)
                # All class samples have the same ground truth index:
                bottleneck_subset_ground_truth_indices = np.array([j for j in range(total_num_class_samples)])

                # Run the bottleneck subset through the computational graph:
                class_results = sess.run(output_op.outputs[0], feed_dict={
                    input_bottleneck_op.outputs[0]: bottleneck_subset_values
                })

                class_top_5_samples_correct = 0
                class_top_5_samples_incorrect = 0
                # print('class_results pre-squeeze shape: %s' % (class_results.shape,))
                class_results = np.squeeze(class_results)
                # print('class_results shape post-squeeze: %s' % (class_results.shape,))

                for j, class_result in enumerate(class_results):
                    # Obtain the indices corresponding to a sorting of y_proba in ascending order:
                    top_k = class_result.argsort()[-5:][::-1]           # k = 5
                    pred_class_indices = top_k
                    pred_class_probs = [class_results[j][pred_class_index] for pred_class_index in pred_class_indices]
                    pred_class_labels = [class_labels[pred_class_index] for pred_class_index in pred_class_indices]
                    ground_truth_class_labels = [class_label for i in range(total_num_class_samples)]
                    print('\tClass sample [%d/%d] top-5 predicted classes: %s with %s probabilities. The real class was: \'%s (%d)\'' % (j+1, total_num_class_samples, pred_class_labels, pred_class_probs, class_label, i))
                    if i in pred_class_indices:
                        class_top_5_samples_correct += 1
                    else:
                        class_top_5_samples_incorrect += 1
                assert class_top_5_samples_correct + class_top_5_samples_incorrect == total_num_class_samples
                print('Class \'%s (%d)\'\'s top-5 accuracy is: %.2f%%' % (class_label, i, (class_top_5_samples_correct / total_num_class_samples)*100))
                class_top_5_accuracies[i] = {'class': class_label, 'top_5_acc': (class_top_5_samples_correct / total_num_class_samples)*100}
        print('class_top_5_accuracies: %s' % class_top_5_accuracies)
        return class_top_5_accuracies


def main(run_config):
    if run_config['process'] == 'Training':
        preceding_process = None
        raise NotImplementedError('This is not a fair evaluation')
    elif run_config['process'] == 'Validation':
        preceding_process = 'train'
    elif run_config['process'] == 'Testing':
        preceding_process = 'val'
    else:
        preceding_process = None

    preceding_process_run_config = run_configs[run_config['dataset']][preceding_process]
    preceding_process_model_path = preceding_process_run_config['saved_model_path']
    current_process_model_path = run_config['saved_model_path']

    preceding_process_model_path = os.path.join(preceding_process_model_path, os.listdir(preceding_process_model_path)[0])
    current_process_model_path = os.path.join(current_process_model_path, os.listdir(current_process_model_path)[0])

    preceding_process_model_path = os.path.join(preceding_process_model_path, 'trained_model')
    current_process_model_path = os.path.join(current_process_model_path, 'trained_model')

    preceding_process_model_label_file_path = os.path.join(preceding_process_model_path, 'class_labels.txt')
    current_process_model_label_file_path = os.path.join(current_process_model_path, 'class_labels.txt')

    bottleneck_executor = BottleneckExecutor(
        image_dir=run_config['image_dir'],
        logging_dir=run_config['logging_dir'],
        tfhub_module_url='https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1',
        compressed_bottleneck_file_path=run_config['bottleneck_path']
    )
    all_bottlenecks = bottleneck_executor.get_bottlenecks()
    class_labels = list(all_bottlenecks['class'].unique())
    train_bottlenecks, val_bottlenecks, test_bottlenecks = bottleneck_executor.get_partitioned_bottlenecks()
    # assert len(class_labels) == len(train_bottlenecks['class'].unique())
    # print('Different class label: %s' % list(set(train_bottlenecks['class'].unique()) ^ set(class_labels)))
    # print('Different class labels: %s' % np.setdiff1d(train_bottlenecks['class'].unique(), val_bottlenecks['class'].unique()))
    # assert len(class_labels) == len(val_bottlenecks['class'].unique())
    # assert len(class_labels) == len(test_bottlenecks['class'].unique())

    # Load trained classifier:
    tfh_classifier = TrainedTFHClassifier(preceding_model_path=os.path.join(preceding_process_model_path, 'inference'), preceding_model_label_file_path=preceding_process_model_label_file_path, current_model_label_file_path=current_process_model_label_file_path)
    # tfh_classifier.load_preceding_model()

    if run_config['process'] == 'Training':
        train_bottleneck_values = train_bottlenecks['bottleneck'].tolist()
        train_bottleneck_values = np.array(train_bottleneck_values)
        train_bottleneck_ground_truth_labels = train_bottlenecks['class'].values
        # Convert the labels into indices (one hot encoding by index):
        train_bottleneck_ground_truth_indices = np.array([class_labels.index(ground_truth_label)
                                                          for ground_truth_label in train_bottleneck_ground_truth_labels])
        class_top_1_accuracies = tfh_classifier.calculate_class_top_1_accuracies(current_process_bottlenecks=train_bottlenecks, class_labels=class_labels)
        class_top_1_positive_predictive_values = tfh_classifier.calculate_class_top_1_positive_predictive_values(current_process_bottlenecks=train_bottlenecks, class_labels=class_labels)
        class_top_5_accuracies = tfh_classifier.calculate_class_top_5_accuracies(bottlenecks=train_bottlenecks, class_labels=class_labels)
        top_1_accs = [value['top_1_acc'] for (key, value) in class_top_1_accuracies.items()]
        top_5_accs = [value['top_5_acc'] for (key, value) in class_top_5_accuracies.items()]
        print('Average top-1 Accuracy (training set): %.2f%%' % (sum(top_1_accs)/len(top_1_accs)))
        print('Average top-5 Accuracy (training set): %.2f%%' % (sum(top_5_accs)/len(top_5_accs)))
        with open('top_1_accuracies_by_class_train_set.json', 'w') as fp:
            json.dump(class_top_1_accuracies, fp, indent=4, separators=(',', ': '))
        with open('top_5_accuracies_by_class_train_set.json', 'w') as fp:
            json.dump(class_top_5_accuracies, fp, indent=4, separators=(',', ': '))
    elif run_config['process'] == 'Validation':
        val_bottleneck_values = val_bottlenecks['bottleneck'].tolist()
        val_bottleneck_values = np.array(val_bottleneck_values)
        val_bottleneck_ground_truth_labels = val_bottlenecks['class'].values
        # Convert the labels into indices (one hot encoding by index):
        val_bottleneck_ground_truth_indices = np.array([class_labels.index(ground_truth_label)
                                                        for ground_truth_label in val_bottleneck_ground_truth_labels])
        class_top_1_accuracies = tfh_classifier.calculate_class_top_1_accuracies(current_process_bottlenecks=val_bottlenecks, class_labels=class_labels)
        class_top_1_positive_predictive_values = tfh_classifier.calculate_class_top_1_positive_predictive_values(current_process_bottlenecks=val_bottlenecks, class_labels=class_labels)
        class_top_5_accuracies = tfh_classifier.calculate_class_top_5_accuracies(bottlenecks=val_bottlenecks, class_labels=class_labels)
        top_1_accs = [value['top_1_acc'] for (key, value) in class_top_1_accuracies.items()]
        top_5_accs = [value['top_5_acc'] for (key, value) in class_top_5_accuracies.items()]
        print('Average top-1 Accuracy (validation set): %.2f%%' % (sum(top_1_accs)/len(top_1_accs)))
        print('Average top-5 Accuracy (validation set): %.2f%%' % (sum(top_5_accs)/len(top_5_accs)))
        with open('top_1_accuracies_by_class_val_set.json', 'w') as fp:
            json.dump(class_top_1_accuracies, fp, indent=4, separators=(',', ': '))
        with open('top_5_accuracies_by_class_val_set.json', 'w') as fp:
            json.dump(class_top_5_accuracies, fp, indent=4, separators=(',', ': '))
    elif run_config['process'] == 'Testing':
        test_bottleneck_values = test_bottlenecks['bottleneck'].tolist()
        test_bottleneck_values = np.array(test_bottleneck_values)
        test_bottleneck_ground_truth_labels = test_bottlenecks['class'].values
        # Convert the labels into indices (one hot encoding by index):
        test_bottleneck_ground_truth_indices = np.array([class_labels.index(ground_truth_label)
                                                        for ground_truth_label in test_bottleneck_ground_truth_labels])
        class_top_1_accuracies = tfh_classifier.calculate_class_top_1_accuracies(current_process_bottlenecks=test_bottlenecks, class_labels=class_labels)
        class_top_1_positive_predictive_values = tfh_classifier.calculate_class_top_1_positive_predictive_values(current_process_bottlenecks=test_bottlenecks, class_labels=class_labels)
        class_top_5_accuracies = tfh_classifier.calculate_class_top_5_accuracies(bottlenecks=test_bottlenecks, class_labels=class_labels)
        top_1_accs = [value['top_1_acc'] for (key, value) in class_top_1_accuracies.items()]
        top_5_accs = [value['top_5_acc'] for (key, value) in class_top_5_accuracies.items()]
        print('Average top-1 Accuracy (testing set): %.2f%%' % (sum(top_1_accs)/len(top_1_accs)))
        print('Average top-5 Accuracy (testing set): %.2f%%' % (sum(top_5_accs)/len(top_5_accs)))
        with open('top_1_accuracies_by_class_test_set.json', 'w') as fp:
            json.dump(class_top_1_accuracies, fp, indent=4, separators=(',', ': '))
        with open('top_5_accuracies_by_class_test_set.json', 'w') as fp:
            json.dump(class_top_5_accuracies, fp, indent=4, separators=(',', ': '))
    else:
        print('ERROR: Could not identify process designation')


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
            'val': {
                'dataset': 'BOONE',
                'process': 'Validation',
                'image_dir': 'D:\\data\\BOON\\images',
                'bottleneck_path': 'D:\\data\\BOON\\bottlenecks.pkl',
                'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeepKeras\\frameworks\\DataAcquisition\\CleaningResults\\BOON',
                'hyperparam_df_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\Boone\\gs_val_hyperparams.pkl',
                'top_1_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\Boone\\top_1_accuracies_by_class_val_set.json',
                'top_5_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\Boone\\top_5_accuracies_by_class_val_set.json',
                'saved_model_path': 'D:\\data\\BOON\\training summaries\\10-25-2019\\gs_winner\\train'
            },
            'train': {
                'dataset': 'BOONE',
                'process': 'Training',
                'image_dir': 'D:\\data\\BOON\\images',
                'bottleneck_path': 'D:\\data\\BOON\\bottlenecks.pkl',
                'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeepKeras\\frameworks\\DataAcquisition\\CleaningResults\\BOON',
                'hyperparam_df_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\Boone\\gs_train_hyperparams.pkl',
                'top_1_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\Boone\\top_1_accuracies_by_class_train_set.json',
                'top_5_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\Boone\\top_5_accuracies_by_class_train_set.json',
                'saved_model_path': 'D:\\data\\BOON\\training summaries\\10-25-2019\\gs_winner\\train'
            },
            'test': {
                    'dataset': 'BOONE',
                    'process': 'Testing',
                    'image_dir': 'D:\\data\\BOON\\images',
                    'bottleneck_path': 'D:\\data\\BOON\\bottlenecks.pkl',
                    'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeepKeras\\frameworks\\DataAcquisition\\CleaningResults\\BOON',
                    'hyperparam_df_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\Boone\\gs_test_hyperparams.pkl',
                    'top_1_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\Boone\\top_1_accuracies_by_class_test_set.json',
                    'top_5_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\Boone\\top_5_accuracies_by_class_test_set.json',
                    'saved_model_path': 'D:\\data\\BOON\\training summaries\\10-25-2019\\gs_winner\\train'
            }
        },
        'GoingDeeper': {
            'train': {
                'dataset': 'GoingDeeper',
                'process': 'Training',
                'image_dir': 'D:\\data\\GoingDeeperData\\images',
                'bottleneck_path': 'D:\\data\\GoingDeeperData\\bottlenecks.pkl',
                'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeepKeras\\frameworks\\DataAcquisition\\CleaningResults\\GoingDeeper',
                'saved_model_path': 'D:\\data\\GoingDeeperData\\training summaries\\10-28-2019\\gs_winner\\train',
                'hyperparam_df_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\GoingDeeper\\gs_train_hyperparams.pkl',
                'top_1_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\GoingDeeper\\top_1_accuracies_by_class_train_set.json',
                'top_5_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\GoingDeeper\\top_5_accuracies_by_class_train_set.json'
            },
            'val': {
                'dataset': 'GoingDeeper',
                'process': 'Validation',
                'image_dir': 'D:\\data\\GoingDeeperData\\images',
                'bottleneck_path': 'D:\\data\\GoingDeeperData\\bottlenecks.pkl',
                'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeepKeras\\frameworks\\DataAcquisition\\CleaningResults\\GoingDeeper',
                'saved_model_path': 'D:\\data\\GoingDeeperData\\training summaries\\10-28-2019\\gs_winner\\train',
                'hyperparam_df_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\GoingDeeper\\gs_val_hyperparams.pkl',
                'top_1_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\GoingDeeper\\top_1_accuracies_by_class_val_set.json',
                'top_5_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\GoingDeeper\\top_5_accuracies_by_class_val_set.json'
            },
            'test': {
                'dataset': 'GoingDeeper',
                'process': 'Testing',
                'image_dir': 'D:\\data\\GoingDeeperData\\images',
                'bottleneck_path': 'D:\\data\\GoingDeeperData\\bottlenecks.pkl',
                'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeepKeras\\frameworks\\DataAcquisition\\CleaningResults\\GoingDeeper',
                'saved_model_path': 'D:\\data\\GoingDeeperData\\training summaries\\10-28-2019\\gs_winner\\train',
                'hyperparam_df_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\GoingDeeper\\gs_test_hyperparams.pkl',
                'top_1_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\GoingDeeper\\top_1_accuracies_by_class_test_set.json',
                'top_5_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\GoingDeeper\\top_5_accuracies_by_class_test_set.json'
            }},
        'SERNEC': {}
    }
    main(run_config=run_configs['BOONE']['val'])
