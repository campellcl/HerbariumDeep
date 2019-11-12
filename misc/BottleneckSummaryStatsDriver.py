import numpy as np
import os
from frameworks.DataAcquisition.BottleneckExecutor import BottleneckExecutor
import statistics
from collections import Counter


def main(run_config):
    bottleneck_executor = BottleneckExecutor(
        image_dir=run_config['image_dir'],
        logging_dir=run_config['logging_dir'],
        tfhub_module_url='https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1',
        compressed_bottleneck_file_path=run_config['bottleneck_path']
    )
    all_bottlenecks = bottleneck_executor.get_bottlenecks()
    class_labels = list(all_bottlenecks['class'].unique())
    train_bottlenecks, val_bottlenecks, test_bottlenecks = bottleneck_executor.get_partitioned_bottlenecks(
        train_percent=.80, val_percent=.20, test_percent=.20, random_state=42
    )
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
    X_train, y_train = train_bottleneck_values, train_bottleneck_ground_truth_indices
    X_valid, y_valid = val_bottleneck_values, val_bottleneck_ground_truth_indices

    num_train_bottlenecks = train_bottlenecks.shape[0]
    num_val_bottlenecks = val_bottlenecks.shape[0]
    num_test_bottlenecks = test_bottlenecks.shape[0]

    class_labels_and_bottleneck_vector_counts = {}
    for class_label in class_labels:
        class_labels_and_bottleneck_vector_counts[class_label] = all_bottlenecks[all_bottlenecks['class'] == class_label].shape[0]

    # Export summary stats:
    with open(os.path.join(run_config['logging_dir'], 'bottleneck_executor_summary_stats.txt'), 'w') as fp:
        fp.write('Number of Training Bottlenecks: %d\n' % num_train_bottlenecks)
        fp.write('Number of Validation Bottlenecks: %d\n' % num_val_bottlenecks)
        fp.write('Number of Testing Bottlenecks: %d\n' % num_test_bottlenecks)
        fp.write('\n')
        fp.write('Number of Unique Bottleneck Class Labels: %d\n' % len(class_labels))
        fp.write('Total Number of Bottleneck Vectors: %d\n' % all_bottlenecks.shape[0])
        fp.write('Enforced Minimum Number of Bottleneck Vectors per Class: %d\n' % 20)
        fp.write('Actual Minimum Number of Bottleneck Vectors per Class: %d\n' % min(class_labels_and_bottleneck_vector_counts.values()))
        fp.write('Enforced Maximum Number of Bottleneck Vectors per Class: 2^{27} - 1 ~= 134 M\n')
        fp.write('Actual Maximum Number of Bottleneck Vectors per Class: %d\n' % max(class_labels_and_bottleneck_vector_counts.values()))
        fp.write('Mean Number of Bottleneck Vectors per Class: %.2f\n' % np.mean(list(class_labels_and_bottleneck_vector_counts.values())))
        fp.write('Median Number of Bottleneck Vectors per Class: %.2f\n' % np.median(list(class_labels_and_bottleneck_vector_counts.values())))
        try:
            mode_bottleneck_vectors_per_class = statistics.mode(list(class_labels_and_bottleneck_vector_counts.values()))
            mode_count = len([bottleneck_count for clss_label, bottleneck_count in class_labels_and_bottleneck_vector_counts.items() if bottleneck_count == mode_bottleneck_vectors_per_class])
            fp.write('Mode Number of Bottleneck Vectors per Class: %d (%d counts)\n' % (statistics.mode(list(class_labels_and_bottleneck_vector_counts.values())), mode_count))
        except statistics.StatisticsError as err:
            data = Counter(list(class_labels_and_bottleneck_vector_counts.values()))
            print('Note: Two most equally common values: %s' % data.most_common(2))
            fp.write('Mode_1 Number of Bottleneck Vectors per Class: %d (%d counts)\n' % (data.most_common(2)[0][0], data.most_common(2)[0][1]))
            fp.write('Mode_2 Number of Bottleneck Vectors per Class: %d (%d counts)\n' % (data.most_common(2)[1][0], data.most_common(2)[1][1]))

    print('Wrote Bottleneck Executor summary statistics to \'%s\'' % os.path.join(run_config['logging_dir'], 'bottleneck_executor_summary_stats.txt'))

    # First we need to calculate the prior probabilities using the distribution in the training dataset:
    classes_and_counts_series = train_bottlenecks['class'].value_counts()
    num_samples = train_bottlenecks.shape[0]

    print()


if __name__ == '__main__':
    run_configs = {
        'DEBUG': {
            'dataset': 'DEBUG',
            'image_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\GoingDeeper\\images',
            'bottleneck_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\GoingDeeper\\images\\bottlenecks.pkl',
            'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\DataAcquisition\\CleaningResults\\DEBUG'
        },
        'BOON': {
            'dataset': 'BOON',
            'image_dir': 'D:\\data\\BOON\\images',
            'bottleneck_path': 'D:\\data\\BOON\\bottlenecks.pkl',
            'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\DataAcquisition\\CleaningResults\\BOON'
        },
        'GoingDeeper': {
            'dataset': 'GoingDeeper',
            'image_dir': 'D:\\data\\GoingDeeperData\\images',
            'bottleneck_path': 'D:\\data\\GoingDeeperData\\bottlenecks.pkl',
            'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\DataAcquisition\\CleaningResults\\GoingDeeper'
        },
        'SERNEC': {
            'dataset': 'SERNEC',
            'image_dir': 'D:\\data\\SERNEC\\images',
            'bottleneck_path': 'D:\\data\\SERNEC\\bottlenecks.pkl',
            'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\DataAcquisition\\CleaningResults\\SERNEC'
        }
    }
    main(run_configs['GoingDeeper'])
