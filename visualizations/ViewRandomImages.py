import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from frameworks.DataAcquisition.BottleneckExecutor import BottleneckExecutor


def main(run_config):
    bottleneck_executor = BottleneckExecutor(
        image_dir=run_config['image_dir'],
        logging_dir=run_config['logging_dir'],
        tfhub_module_url='https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1',
        compressed_bottleneck_file_path=run_config['bottleneck_path']
    )
    all_bottlenecks = bottleneck_executor.get_bottlenecks()
    class_labels = list(all_bottlenecks['class'].unique())
    image_executor = bottleneck_executor.image_executor
    image_lists = image_executor.get_image_lists()
    num_images_to_show = 4
    random_keys = [list(image_lists.keys())[np.random.randint(0, len(image_lists.keys()))] for _ in range(num_images_to_show)]
    random_samples = []
    for random_key in random_keys:
        random_sample_index = np.random.randint(0, len(image_lists[random_key]))
        random_samples.append(image_lists[random_key][random_sample_index])
    print('Displaying sample images: %s' % random_samples)
    image_executor.display_sample_images(sample_image_paths=random_samples)

    pass



if __name__ == '__main__':
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
                'saved_model_path': 'D:\\data\\BOON\\training summaries\\8-16-2019\\gs_winner\\train'
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
                'saved_model_path': 'D:\\data\\BOON\\training summaries\\8-16-2019\\gs_winner\\train'
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
                    'saved_model_path': 'D:\\data\\BOON\\training summaries\\8-16-2019\\gs_winner\\train'
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
    main(run_config=run_configs['GoingDeeper']['val'])
