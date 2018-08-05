"""
GoingDeeperTransferLearn.py
Implementation of transfer learning for the Going Deeper dataset.
"""

from torchvision import models
from torch.autograd import Variable
import argparse
import torch as pt
import torch.nn as nn
import os
import pandas as pd
import numpy as np
from sklearn import model_selection
import shutil
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import time
import copy

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Transfer Learning Demo on Going Deeper Herbaria 1K Dataset')
parser.add_argument('STORE', metavar='DIR', help='Data storage directory.')
parser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true',
                    help='Enable verbose print statements (yes, no)?')
parser.add_argument('--arch', '-a', metavar='ARCH', default='inception_v3', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: inception_v3)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')


def get_metadata_properties():
    """
    get_metadata_properties: Returns a dictionary containing various properties about the datasets relevant to the
        training of machine learning models.
    :returns metadata_prop: A dictionary composed of the following metadata about datasets relevant to the training process:
        :return has_test_set: A boolean variable indicating if a test set folder is present (DataLoader context).
        :return has_train_set: A boolean variable indicating if a training set folder is present (DataLoader context).
        :return has_val_set: A boolean variable indicating if a validation set folder is present (DataLoader context).
        :return num_cols: An integer variable representing the number of columns in all datasets.
        :return num_samples: A dictionary housing the number of samples for all of the datasets.
            :return num_samples['test_set']: The number of cleaned samples in the testing dataset (None if not present).
            :return num_samples['train_set']: The number of cleaned samples in the training dataset (None if not present).
            :return num_samples['val_set']: The number of cleaned samples in the validation dataset (None if not present).
        :return class_names: A list of classes found across all datasets.
        :return data_loader_prop: A dictionary containing the following properties relevant to dataloader instances:
            :return data_loader_prop['num_workers']: The number of threads to instantiate DataLoaders with.
            :return data_loader_prop['batch_sizes']: How many images the data loader grabs during one call to
                next(iter(data_loader)).
                :return data_loader_prop['batch_sizes']['train']: The batch size for DataLoader instances loading,
                    training data.
                :return data_loader_prop['batch_sizes']['test']: The batch size for DataLoader instances loading,
                    testing data.
                :return data_loader_prop['batch_sizes']['val']: The batch size for DataLoader instances loading,
                    validation data.
            :return img_load_size: The size (in pixels) which images should be re-sized to when loaded by DataLoader,
                instances.
            :return receptive_field_size: The size (in pixels) which images will be center cropped to when loaded by,
                DataLoader instances during training.
    """
    return NotImplementedError


def sort_by_target_then_collector(df_meta):
    """
    sort_by_target_then_collector: Sorts the provided metadata in ascending order first by the target label
    (species/dwc:scientificName), and second by the collector attribute (dwc:recordedBy).
    :param df_meta:
    :return:
    """
    df_meta_sorted = df_meta.sort_values(by=['dwc:scientificName', 'dwc:recordedBy'], axis=0,
                                         ascending=True, kind='quicksort', na_position='last')
    return df_meta_sorted


# def partition_data(df_meta):
#     """
#     partition_data: Partitions the metadata into training, testing, and validation datasets. Partitioning is performed
#         secondarily by target label (dwc:scientificName) in an attempt to keep classes equally represented in both
#         training and testing sets. Partitioned is performed primarily by collector so that no single collector has
#         entries in both the training and testing datasets. This is done to avoid bias in the classifier. It is not
#         desired that the classifier learns to predict samples only collected by a certain type of collector (i.e. those
#         that follow a particular style of sample mounting). By isolating collectors' samples to either training or
#         test sets in their entirely the algorithm has less of a chance to pick up on patterns among collectors and can
#         instead focus on learning patterns among target labels (species). This method will update the respective,
#         metadata dataframe's 'file_path' attribute to house the new location of the samples after partitioning.
#     :param df_meta: The global metadata dataframe.
#     :returns Up to 3 instances of the metadata partitioned into separate dataframes:
#         :return df_meta_train:
#         :return df_meta_test:
#         :return df_meta_val:
#     """
#     species_coll_counts = {}
#     # Sort the dataframe first by the class label and then by collector:
#     df_meta = sort_by_target_then_collector(df_meta)
#     total_num_target_labels = len(df_meta['dwc:scientificName'].unique())
#     for i, target_label in enumerate(df_meta['dwc:scientificName'].unique()):
#         if target_label is not np.nan:
#
#             coll_counts = {}
#             target_subset = df_meta[df_meta['dwc:scientificName'] == target_label]
#             # Drop entries with a collector of np.nan from the target label subset:
#             target_subset = target_subset.dropna(axis=0, how='any', subset=['dwc:recordedBy'])
#             for collector in target_subset['dwc:recordedBy'].unique():
#                 collector_samples = target_subset[target_subset['dwc:recordedBy'] == collector]
#                 coll_counts[collector] = collector_samples.shape[0]
#             species_coll_counts[target_label] = coll_counts
#             # print('collector counts for target label [%s]: %s' % (target_label, coll_counts))
#             print('Finished partitioning samples by collector for target label <%d/%d>: [%s]'
#                   % (i+1, total_num_target_labels, target_label))
#             print('\tThere are %d samples for this target label with a non-NaN collector.'
#                   % target_subset.shape[0])
#             print('\tThere are %d collectors for this target label.' % len(coll_counts))
#             collection_counts = list(coll_counts.values())
#             if collection_counts:
#                 print('\tThe min number of samples recorded by an individual collector was: %d'
#                       % min(collection_counts))
#                 print('\tThe max number of samples recorded by an individual collector was: %d.'
#                       % max(collection_counts))
#                 print('\tThe mean number of samples recorded by an individual collector was: %d'
#                       % np.mean(collection_counts))
#                 print('\tThe median number of samples recorded by an individual collector was: %d'
#                       % np.median(collection_counts))
#                 # print('\tThe mode number of samples recorded by an individual collector was: %d'
#                 #       % mode(list(coll_counts.values())))
#                 ''' partition collectors into training or testing based on number of samples '''
#                 # Sort the target_subset by collector and number of observations:
#                 coll_counts = target_subset['dwc:recordedBy'].value_counts(dropna=True)
#                 coll_counts = coll_counts[coll_counts.values != 0]
#                 total_num_samples = target_subset.shape[0]
#                 percent_test = 20
#                 ideal_num_test_samples = ceil((percent_test * total_num_samples)/100)
#                 ideal_num_train_samples = total_num_samples - ideal_num_test_samples
#
#                 # Get every possible permutation of the collectors and their samples (order matters):
#                 train_test_perms = permutations(coll_counts.items(), r=coll_counts.shape[0])
#                 # train_test_perms = [dict(zip(coll_counts, v)) for v in permutations(coll_counts.values(), r=2)]
#                 for p, perm in enumerate(train_test_perms):
#                     print('permutation: %s' % (perm, ))
#                 #     ratio = coll_counts[perm[0]]/coll_counts[perm[1]]
#                 #     pass
#                 pass
#                 # sorted_coll_counts = [(coll, coll_counts[coll]) for coll in sorted(coll_counts, key=coll_counts.get, reverse=)]
#     pass
#
#
#             # TODO: how to divide the collector subsets to maintain 80/20 ratio?
#     # Collect statistics on the number of viable class labels:
#     # TODO: Partition so that collector is isolated and target labels equally represented in both train and test splits.
#     # TODO: Actually move the data on the hard drive to the appropriate folders.
#     # TODO: Update the columns of the respective dataframe to the new sample locations
#     # TODO: Update the global metadata dataframe if necessary.
#     # TODO: Return df_meta_train, df_meta_test, df_meta_val.
#     return NotImplementedError


def update_metadata_with_file_paths(df_meta):
    """
    update_metadata_with_file_paths: Updates the global metadata dataframe by appending a column with the file path of
        every sample on the hard drive. This is done in order to move files after partitioning into training and test
        datasets. This method provides a way to link the physical location of image files on the hard drive to the
        metadata of the specimen.
    :param df_meta: The global metadata dataframe (with no currently existing 'file_path' column).
    :return df_meta: The global metadata dataframe augmented with a file_path column which holds the physical location
        on the hard drive of each sample.
    """
    print('Updating the metadata with file paths for every sample. This code is sequential and will take a while...')
    # Add a new column of None to df_meta:
    df_meta = df_meta.assign(file_path=pd.Series([None for i in range(df_meta.shape[0])]).values)
    root_folder = args.STORE + '\\images\\'
    dir_num = 0
    for item in os.listdir(root_folder):
        if os.path.isdir(os.path.join(root_folder, item)):
            the_folder_path = os.path.join(root_folder, item)
            # f_num = 0
            for the_file in os.listdir(the_folder_path):
                # Find the sample in df_meta that matches this image:
                urls = df_meta['ac:accessURI'].tolist()
                filenames = [url.split('/')[-1] for url in urls]
                for i, f_name in enumerate(filenames):
                    if not '.jpg' in f_name.lower():
                        f_name = f_name + '.jpg'
                    if f_name == the_file:
                        # Now have the index in df_meta of this file.
                        # Append this samples file_path to the global metadata dataframe:
                        df_meta.at[i, 'file_path'] = os.path.join(the_folder_path, the_file)
                        # print('Just matched the %d\'th file.' % f_num)
                        # f_num += 1
                        break
                # query_view = df_meta[os.path.basename(df_meta['ac:accessURI'].str) == os.path.basename(the_file)]
                # pass
                # if os.path.basename(the_file) is in df_meta['dwc:accessURI'
            print('Just finished the %d\'th directory' % dir_num)
            dir_num += 1
    return df_meta


def partition_data():
    """
    partition_data: Partitions df_meta into df_train and df_test using a 20% test split and 80% train. Shuffles the
        samples in df_meta prior to partitioning. This method then attempts to move every sample to its respective
        folder (either train or test). If the move was successful the sample's file_path attribute is updated
        accordingly.
    :return:
    """
    # Data needs to be split into train, test, validate groups.
    # Data should be split so that no collector ends up in both the training and testing sets.
    # Load the metadata dataframe holding the collector of each sample:
    if os.path.isfile(args.STORE + '\\images\\df_meta.pkl'):
        # Has the data already been partitioned?
        if not os.path.isfile(args.STORE + '\\images\df_meta_train.pkl'):
            df_meta = pd.read_pickle(args.STORE + '\\images\\df_meta.pkl')
            # Check to see if the metadata has been updated with the file path of its samples:
            if 'file_path' not in df_meta.columns:
                # Update the df_meta dataframe with the file paths of it's samples:
                df_meta = update_metadata_with_file_paths(df_meta)
                # Pickle the dataframe so this code isn't run again:
                df_meta.to_pickle(path=args.STORE + '\\images\\df_meta.pkl')
            # Drop rows with a file_path of None (hopefully b/c http 404 and not string matching errors during reconstruct):
            df_meta = df_meta.mask(df_meta.eq('None')).dropna(axis=0, how='all', subset=['file_path'])
            ''' The exciting part! '''
            # Drop everything but the file path and the target label from df_meta:
            # df_meta = df_meta[['dwc:scientificName', 'file_path']]
            # Partition into train and test splits:
            df_train, df_test = model_selection.train_test_split(df_meta, test_size=0.2, shuffle=True)
            ''' now actually move the files '''
            if not os.path.isdir(args.STORE + '\\images\\train'):
                os.mkdir(args.STORE + '\\images\\train')
            if not os.path.isdir(args.STORE + '\\images\\test'):
                os.mkdir(args.STORE + '\\images\\test')
            # iterate through test data and move all files:
            df_test_copy = df_test.copy(deep=True)
            for i, row in df_test_copy.iterrows():
                split_path = str.split(row['file_path'], sep='\\')
                split_path.insert(-2, 'test')
                desired_new_dir = '\\'.join(split_path[0:-1])
                desired_file_path = '\\'.join(split_path)
                if not os.path.isdir(desired_new_dir):
                    os.mkdir(desired_new_dir)
                try:
                    # Attempt to move the file:
                    shutil.move(row['file_path'], desired_file_path)
                    # Update the metadata with the new file_path if successful:
                    df_test.at[i, 'file_path'] = desired_file_path
                except Exception as err:
                    print(err)
            df_test.to_pickle(path=args.STORE + '\\images\\df_test.pkl')
            del df_test_copy
            df_train_copy = df_train.copy(deep=True)
            for i, row in df_train_copy.iterrows():
                split_path = str.split(row['file_path'], sep='\\')
                split_path.insert(-2, 'train')
                desired_new_dir = '\\'.join(split_path[0:-1])
                desired_file_path = '\\'.join(split_path)
                if not os.path.isdir(desired_new_dir):
                    os.mkdir(desired_new_dir)
                try:
                    # Attempt to move the file:
                    shutil.move(row['file_path'], desired_file_path)
                    # Update the metadata with the new file_path if successful:
                    df_train.at[i, 'file_path'] = desired_file_path
                except Exception as err:
                    print(err)
            df_train.to_pickle(path=args.STORE + '\\images\\df_train.pkl')
            del df_train_copy
    else:
            print('df_meta.pkl not present, run the GoingDeeperImageDownloader to re-aggregate data.')
            exit(-1)


def get_image_channel_means_and_std_deviations(df_train, df_test):
    """
    get_image_means: Returns the means of each color channel for all images, as well as the standard deviations of each
        color channel across all images.
    :source url: http://forums.fast.ai/t/image-normalization-in-pytorch/7534/7
    :return:
    """

    train_img_folder = args.STORE + '\\images\\train'
    test_img_folder = args.STORE + '\\images\\test'
    img_pop_means = []
    img_pop_std_devs = []
    train_img_pop_means = []
    test_img_pop_means = []
    train_img_pop_std_devs = []
    test_img_pop_std_devs = []

    # Test images:
    for item in os.listdir(test_img_folder):
        if os.path.isdir(os.path.join(test_img_folder, item)):
            for the_file in os.listdir(os.path.join(test_img_folder, item)):
                # shape (width, height)
                test_img = Image.open(os.path.join(test_img_folder, item, the_file))
                # print('test_img [%s] (width, height): %s' % (the_file, test_img.size))
                # Reshape:
                test_img.thumbnail((1024, 1024))
                # shape after conversion to numpy: (height, width, channels)
                np_test_img = np.array(test_img, dtype="uint8")
                # Refresher on numpy image methods: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#color-images
                # NOTE when using plt.imshow to verify: http://www.degeneratestate.org/posts/2016/Oct/23/image-processing-with-numpy/#Colours
                # fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
                # plt.suptitle('Sample Image %s Channels and Composite' % the_file)
                sample_means = np.zeros(shape=(3,), dtype="uint8")
                sample_std_devs = np.zeros(shape=(3,), dtype="uint8")
                # for c, ax in zip(range(4), axs):
                for c in range(3):
                    # if c == 3:
                    #     ax.imshow(np_test_img)
                    #     ax.set_axis_off()
                    # else:
                    #     tmp_im = np.zeros(np_test_img.shape, dtype="uint8")
                    #     tmp_im[:, :, c] = np_test_img[:, :, c]
                    #     ax.set_title('mean: %.4f std: %.4f'
                    #                  % (np.mean(np_test_img[:, :, c]), np.std(np_test_img[:, :, c])))
                    sample_means[c] = np.mean(np_test_img[:, :, c])
                    sample_std_devs[c] = np.std(np_test_img[:, :, c])
                        # ax.imshow(tmp_im)
                        # ax.set_axis_off()
                # update population mean's:
                test_img_pop_means.append(sample_means)
                test_img_pop_std_devs.append(sample_std_devs)
                img_pop_means.append(sample_means)
                img_pop_std_devs.append(sample_std_devs)
                ''' Uncomment the following line to display the results of this code in matplotlib for each sample '''
                # plt.show()
            print('Finished calculating channel means and channel standard deviations for testing dir: %s' % item)
    print('test_img_pop_means shape: %s' % (np.array(test_img_pop_means).shape,))
    print('test_img_pop_mean (along first axis): %s' % np.mean(np.array(test_img_pop_means), axis=0))
    print('test_img_pop_std (along first axis): %s' % np.std(np.array(test_img_pop_std_devs), axis=0))

    # Train images:
    for item in os.listdir(train_img_folder):
        if os.path.isdir(os.path.join(train_img_folder, item)):
            for the_file in os.listdir(os.path.join(train_img_folder, item)):
                # shape (width, height)
                train_img = Image.open(os.path.join(train_img_folder, item, the_file))
                # print('train_img [%s] (width, height): %s' % (the_file, train_img.size))
                # Reshape:
                train_img.thumbnail((1024, 1024))
                # shape after conversion to numpy: (height, width, channels)
                np_train_img = np.array(train_img, dtype="uint8")
                # Refresher on numpy image methods: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#color-images
                # NOTE when using plt.imshow to verify: http://www.degeneratestate.org/posts/2016/Oct/23/image-processing-with-numpy/#Colours
                # fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
                # plt.suptitle('Sample Image %s Channels and Composite' % the_file)
                sample_means = np.zeros(shape=(3,), dtype="uint8")
                sample_std_devs = np.zeros(shape=(3,), dtype="uint8")
                # for c, ax in zip(range(4), axs):
                for c in range(3):
                    # if c == 3:
                    #     ax.imshow(np_train_img)
                    #     ax.set_axis_off()
                    # else:
                    #     tmp_im = np.zeros(np_train_img.shape, dtype="uint8")
                    #     tmp_im[:, :, c] = np_train_img[:, :, c]
                    #     ax.set_title('mean: %.4f std: %.4f'
                    #                  % (np.mean(np_train_img[:, :, c]), np.std(np_train_img[:, :, c])))
                    sample_means[c] = np.mean(np_train_img[:, :, c])
                    sample_std_devs[c] = np.std(np_train_img[:, :, c])
                        # ax.imshow(tmp_im)
                        # ax.set_axis_off()
                # update population mean's:
                train_img_pop_means.append(sample_means)
                train_img_pop_std_devs.append(sample_std_devs)
                img_pop_means.append(sample_means)
                img_pop_std_devs.append(sample_std_devs)
                ''' Uncomment the following line to display the results of this code in matplotlib for each sample '''
                # plt.show()
            print('Finished calculating channel means and channel standard deviations for training dir: %s' % item)
    print('train_img_pop_means shape: %s' % (np.array(train_img_pop_means).shape,))
    print('train_img_pop_means (along first axis): %s' % np.mean(np.array(train_img_pop_means), axis=0))
    print('test_img_pop_std (along first axis): %s' % np.std(np.array(train_img_pop_std_devs), axis=0))

    print('img_pop_means shape: %s' % (np.array(img_pop_means).shape,))
    print('img_pop_means (along first axis): %s' % np.mean(np.array(img_pop_means), axis=0))
    print('img_pop_std_devs (along first axis): %s' % np.std(np.array(img_pop_std_devs), axis=0))
    return train_img_pop_means, test_img_pop_means, train_img_pop_std_devs, test_img_pop_std_devs


def get_data_loaders_and_properties(df_train, df_test):
    """
    get_data_loaders: Creates either two or three instances of torch.utils.data.DataLoader depending on the datasets
        present in the storage directory provided via command line argument 'args.STORE' at runtime. Instantiates and
        returns a DataLoader for the training dataset, test dataset, and validation dataset (if present).
    :returns data_loaders, data_props:
        :return data_loaders: A dictionary of torch.utils.DataLoader instances.
        :return data_props: A dictionary of properties relating to the data sets comprised of the following:
            :return has_test_set: A boolean variable indicating if a test set folder is present (DataLoader context).
            :return has_train_set: A boolean variable indicating if a training set folder is present (DataLoader context).
            :return has_val_set: A boolean variable indicating if a validation set folder is present (DataLoader context).
            :return class_names['train']: A list of the class names (folder names in training directory).
            :return class_names['test']: A list of the class names (folder names in the testing directory).
            :return num_classes['train']: The number of classes in the training dataset.
            :return num_classes['test']: The number of classes in the testing dataset.
            :return dataset_sizes['train']: The number of samples in the training dataset.
            :return dataset_sizes['test']: The number of samples in the testing dataset.
    """
    ''' Specified in the Research Paper: '''
    img_pxl_load_size = 1024
    receptive_field_pxl_size = 299
    # How many images the DataLoader will grab during one call to next(iter(data_loader)):
    batch_sizes = {'train': 16, 'test': 16}
    ''' Hyperparameters specified by me: '''
    # Declare number of asynchronous threads per data loader (I chose number of CPU cores):
    num_workers = 6
    shuffle = True
    '''
    Training Data and Validation Data Input Pipeline:
        Data Augmentation and Normalization as described here: http://pytorch.org/docs/master/torchvision/models.html
    '''
    # train_pop_means, test_pop_means, train_pop_std_devs, test_pop_std_devs = \
    #     get_image_channel_means_and_std_deviations(df_train=df_train, df_test=df_test)
    # print('train_pop_mean shape: %s' % np.array(train_pop_means).shape)
    # train_img_pop_means = [0.74535418, 0.70882273, 0.61583241]
    # train_img_pop_std_devs = [0.04480927, 0.04685673, 0.05492202]
    # See: https://pytorch.org/docs/stable/torchvision/models.html
    train_img_pop_means_imgnet = [0.485, 0.456, 0.406]
    train_img_pop_std_devs_imgnet = [0.229, 0.224, 0.225]
    data_transforms = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_pxl_load_size),
            torchvision.transforms.CenterCrop(receptive_field_pxl_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(train_img_pop_means_imgnet, train_img_pop_std_devs_imgnet)
        ]),
        'test': torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_pxl_load_size),
            torchvision.transforms.CenterCrop(receptive_field_pxl_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(train_img_pop_means_imgnet, train_img_pop_std_devs_imgnet)
        ])
    }
    data_loaders = {}
    data_props = {'class_names': {}, 'num_classes': {}, 'dataset_sizes': {}}

    # Training set image folder:
    if os.path.isdir(args.STORE + '\\images\\train'):
        data_props['has_train_set'] = True
        train_img_folder = torchvision.datasets.ImageFolder(args.STORE + '\\images\\train',
                                                            transform=data_transforms['train'])
        # Classes present in the training image set:
        data_props['class_names']['train'] = train_img_folder.classes
        # Number of classes present in the training image set:
        data_props['num_classes']['train'] = len(train_img_folder.classes)
        # Number of samples present in the training image set:
        data_props['dataset_sizes']['train'] = len(train_img_folder)
        # Instantiate the training dataset DataLoader:
        train_loader = pt.utils.data.DataLoader(train_img_folder, batch_size=batch_sizes['train'], shuffle=shuffle,
                                            num_workers=num_workers)
        data_loaders['train'] = train_loader
        if args.verbose:
            print('Training data loader instantiated with:'
                  '\n\tshuffle data: %s'
                  '\n\tnumber of workers (async threads): %d'
                  '\n\tbatch size (during iteration):%d'
                  % (shuffle, num_workers, batch_sizes['train']))
    else:
        data_props['has_train_set'] = False

    # Testing set image folder:
    if os.path.isdir(args.STORE + '\\images\\test'):
        data_props['has_test_set'] = True
        test_img_folder = torchvision.datasets.ImageFolder(args.STORE + '\\images\\test',
                                                           transform=data_transforms['test'])
        # Classes present in the testing image set:
        data_props['class_names']['test'] = test_img_folder.classes
        # Number of classes present in the testing image set:
        data_props['num_classes']['test'] = len(test_img_folder.classes)
        # Number of samples present in the testing image set:
        data_props['dataset_sizes']['test'] = len(test_img_folder)
        # Instantiate the testing dataset DataLoader:
        test_loader = pt.utils.data.DataLoader(test_img_folder, batch_size=batch_sizes['test'], shuffle=shuffle,
                                               num_workers=num_workers)
        data_loaders['test'] = test_loader
        if args.verbose:
            print('Testing data loader instantiated with:'
                      '\n\tshuffle data: %s'
                      '\n\tnumber of workers (async threads): %d'
                      '\n\tbatch size (during iteration):%d'
                      % (shuffle, num_workers, batch_sizes['test']))
    else:
        data_props['has_test_set'] = False
    return data_loaders, data_props


def imshow_tensor(input, title=None):
    """
    imshow_tensor: Matplotlib imshow function for PyTorch Tensor Objects.
    :source URL: http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    :param input: The input image as a Tensor.
    :param title: The title for the image.
    :return:
    """
    # Note: not sure what the point of this transposition is:
    input = input.numpy().transpose((1, 2, 0))
    # Normalize the input Tensor:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    # Restrict to [0, 1] interval:
    input = np.clip(input, a_min=0, a_max=1)
    fig = plt.figure()
    has_title = title is not None
    # fig.add_suplot(Rows,Cols,Pos)
    # Below code does not work because we are dealing with a tensor object.
    # for position in range(num_image):
    #     sub_plt = fig.add_subplot(1, num_image, position+1)
    #     if has_title:
    #         sub_plt.set_title(title[position])
    #         plt.imshow(input)
    # fig.show()
    # a = fig.add_subplot(1, num_image, 0)
    # if has_title:
    #     a.set_title(title[0])
    # b = fig.add_subplot(1, num_image, 1)
    # if has_title:
    #     b.set_title(title[1])
    plt.imshow(input)
    if title is not None:
        plt.title(title)
    # plt.pause(0.001)    # pause a second so that plots are updated?
    # plt.figure(num='Training Data and Ground Truth Labels')
    plt.show()


def visualize_model(data_loaders, class_names, model, num_images=6):
    """
    visualize_model: Generic function to display the models predictions for a few images.
    :source URL: http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    :param model:
    :param num_images:
    :return:
    """
    train_class_names = class_names['train']
    was_training = model.training
    model.train(False)
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    for i, data in enumerate(data_loaders['train']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = model(inputs)
        else:
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = model(inputs)

        _, preds = pt.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(train_class_names[preds[j]]))
            imshow_tensor(inputs.cpu().data[j])

            if images_so_far == num_images:
                model.train(mode=was_training)
                return
    model.train(mode=was_training)


def get_accuracy(model, data_loaders):
    """
    get_accuracy: Computes the overall accuracy of the provided pre-trained model. In this case the accuracy is the
        number of times the network is correct in its prediction for all the test samples.
    :param model: A classification model that has already been trained (or pre-trained).
    :param data_loaders: The torch.utils.data.DataLoader instances which provide access to the training, testing, and
        validation data sets.
    :return accuracy: The overall accuracy of the provided model on the testing data set read by the provided DataLoader.
    """
    if 'test' in data_loaders:
        data_loader = data_loaders['test']
    elif 'val' in data_loaders:
        data_loader = data_loaders['val']
    else:
        print('Error: No test or validation set provided. Can\'t compute the accuracy of the provided model.')
        return NotImplementedError
    # If the model wasn't finished training and the accuracy is being computed on the fly we need to know:
    original_model_state_is_training = model.training
    # If the model is currently in training mode in order to evaluate we need to switch this mode off:
    if model.training:
        model.train(False)
    correct = 0
    total = 0
    for i, data in enumerate(data_loader):
        images, labels = data
        if use_gpu:
            outputs = model(Variable(images.cuda(), volatile=True))
        else:
            outputs = model(Variable(images), volatile=True)
        _, predicted = pt.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()
        print('Current Accuracy (Batch %d): %.2f Percent' % (i, (100 * correct / total)))
    # Now that accuracy is computed if the model was originally in training mode, restore it to training mode:
    if original_model_state_is_training:
        model.train(True)
    return 100 * correct / total

def train_model(data_loaders, model, criterion, optimizer, scheduler, num_epochs=25):
    """
    train_model: TODO: method header
    :param data_loaders:
    :param model:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param num_epochs:
    :return:
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        print('Epoch {%d}/{%d}' % (epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                # Modify learning rate according to schedule:
                scheduler.step()
                # Set the model to training mode (see: http://pytorch.org/docs/master/nn.html#torch.nn.Module.train):
                model.train()
            elif phase == 'test':
                '''
                WARNING: DEBUG PURPOSES ONLY. 
                The model should never be exposed to test data during training.
                '''
                # Set the model to eval mode (see: http://pytorch.org/docs/master/torchvision/models.html):
                model.eval()
            elif phase == 'val':
                # Here it is appropriate to use part of the training set to modify hyperparameters.
                pass

            running_loss = 0.0
            running_num_correct = 0

            # Iterate over the data in minibatches:
            for data in data_loaders[phase]:
                inputs, labels = data
                if use_gpu:
                    inputs = Variable(inputs.cuda(), volatile=False)
                    labels = Variable(labels.cuda(), volatile=False)
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                # Zero paramater gradients:
                optimizer.zero_grad()

                # Compute Forward pass:
                outputs = model(inputs)
                _, preds = pt.max(outputs[0].data, 1)
                loss = criterion(outputs[0], labels)

                # If in the training phase then backprop and optimize by taking a step in the dir of gradient:
                if phase == 'train':
                    # Backprop:
                    loss.backward()
                    # Gradient step:
                    optimizer.step()

                # Update loss and accuracy statistics:
                running_loss += loss.data[0] * inputs.size(0)
                running_num_correct += pt.sum(preds == labels.data)

            epoch_loss = running_loss / data_props['num_samples'][phase]
            losses.append(epoch_loss)
            epoch_acc = running_num_correct / data_props['num_samples'][phase]
            accuracies.append(epoch_acc)

            print('[{}]:\t Epoch Loss: {:.4f} Epoch Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model's weights if this epoch was the best performing:
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Checkpoint: This epoch had the best accuracy. The model weights SHOULD be saved.')
                # create checkpoint:
                # TODO: Implement checkpoints
                # save_checkpoint({
                #     'epoch': epoch + 1,
                #     'arch': args.arch,
                #     'state_dict': copy.deepcopy(model.state_dict()),
                #     'best_prec_1': best_acc,
                #     'optimizer': optimizer.state_dict()
                # }, is_best=True, filename='../../data/PTCheckpoints/model_best.pth.tar')
        print('Accuracy (Top-1 Error or Precision at 1) of the network on %d %s images: %.2f%%\n'
              % (data_props['num_samples'][phase], phase, epoch_acc * 100))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # Load best model weights:
    model.load_state_dict(best_model_wts)
    return model


def main():
    # Declare globals:
    global args, use_gpu, data_props
    args = parser.parse_args()
    use_gpu = pt.cuda.is_available()
    if args.verbose:
        print('CUDA is enabled?: %s' % use_gpu)
    # Check if the data has already been partitioned into train, test, and validation datasets:
    if not os.path.isdir(args.STORE + '\\images\\train'):
        # Partition the metadata and move the files if necessary.
        partition_data()
    # Load the dataframes into memory from the hard drive:
    df_train = pd.read_pickle(args.STORE + '\\images\\df_train.pkl')
    df_test = pd.read_pickle(args.STORE + '\\images\\df_test.pkl')
    if not df_train.empty and not df_test.empty:
        print('Loaded both training and testing metadata into memory. Images already physically partitioned on HDD.')
    else:
        print('Could not load either df_train or df_test with data. Exiting...')
        exit(-1)
    data_loaders, data_props = get_data_loaders_and_properties(df_train, df_test)
    print('Instantiated Lazy DataLoaders.')
    pretrained_model = True
    model = models.__dict__[args.arch](pretrained=pretrained_model)
    if use_gpu:
        model = model.cuda()
    print('Loaded %s source model. CUDA suppport?: %s. Pre-trained?: %s.' % (args.arch, use_gpu, pretrained_model))
    # Freeze the entire network except for the final two layers (as detailed in Going Deeper in Automated Id.)
    # Parameters of newly constructed modules have requires_grad=True by default:
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    # Use training classes because we can't predict what the network hasn't seen:
    model.fc = nn.Linear(in_features=num_ftrs, out_features=data_props['num_classes']['train'], bias=True)
    if use_gpu:
        model = model.cuda()

    # Define loss function (criterion):
    if use_gpu:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    # Define optimizer (observe only params of final layer are being optimized):
    optimizer = pt.optim.SGD(model.fc.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Decay the learning rate by a factor of 0.1 every 7 epochs:
    exp_lr_scheduler = pt.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=7, gamma=0.1)

    print('=> Checkpoints are not currently supported.')
    print('==' * 15 + 'Begin Training' + '==' *15)

    # Accuracy after a single backprop:
    model = train_model(data_loaders=data_loaders, model=model, criterion=criterion, optimizer=optimizer,
                        scheduler=exp_lr_scheduler, num_epochs=25)

    print('==' * 15 + 'Finished Training' + '==' * 15)
    # Visualize model predictions:
    # visualize_model(data_loaders=data_loaders, class_names=class_names['train'], model=model, num_images=6)

    # Initial accuracy before training:
    # top_1_err_before_training = get_accuracy(model, data_loaders)
    # print('Overall Accuracy before training: %.2f' % top_1_err_before_training)


if __name__ == '__main__':
    main()
