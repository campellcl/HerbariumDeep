"""
MergeDirectories.py

Merges the separate training and testing directories into a single directory. Then creates a meta-dataframe to be
    utilized to perform training, validation, and testing, partitions on the fly. This dataframe contains the location
    of each image on the hard drive, and the class of the image.
"""

import os
import pandas as pd
import shutil


def merge_train_test_into_single_dir():
    df_train = pd.read_pickle('D:\\data\\GoingDeeperData\\images\\df_train.pkl')
    df_test = pd.read_pickle('D:\\data\\GoingDeeperData\\images\\df_test.pkl')
    df_meta = pd.concat([df_train, df_test])
    df_meta_updated = df_meta.copy(deep=True)
    for i, sample in df_meta.iterrows():
        if '.jp2' not in sample['file_path']:
            old_path = sample['file_path']
            split_path = str(old_path).split(sep='\\')
            if 'train' in split_path:
                split_path.remove('train')
            elif 'test' in split_path:
                split_path.remove('test')
            else:
                print('Uh oh. No train or test in this filepath. This really shouldn\'t happen.')
            new_path = ''
            for j, sub_dir in enumerate(split_path):
                if j != len(split_path) - 1:
                    new_path = new_path + sub_dir + '\\'
                else:
                    new_path = new_path + sub_dir
            # Get the file path up to the containing folder
            new_dir_name = '//'.join(new_path.split(sep='\\')[0:5])
            # First create the destination directory if it doesn't exist:
            if not os.path.isdir(new_dir_name):
                os.mkdir(new_dir_name)
            # See if this has already been moved:
            if not os.path.isfile(path=new_path):
                # Then move the file to the new directory:
                try:
                    shutil.move(src=old_path, dst=new_path)
                except OSError as err:
                    split_new_path = new_path.split('\\')
                    file_name = split_new_path[-1]
                    target_label = split_new_path[-2]
                    # Did we fail to get the original path because it was deleted do to too few samples?
                    if os.path.isfile('D:\\data\\GoingDeeperData\\rejected train\\too few samples\\' + target_label + '\\' + file_name):
                        print('Removing \'%s\' from updated metadata dataframe as \'%s\' has too few samples.' % (new_path, target_label))
                        df_meta_updated = df_meta_updated.drop(labels=[i], axis=0, errors='raise')
                    elif os.path.isfile('D:\\data\GoingDeeperData\\rejected test\\too few samples\\' + target_label + '\\' + file_name):
                        print('Removing \'%s\' from updated metadata dataframe as \'%s\' has too few samples.' % (new_path, target_label))
                        df_meta_updated = df_meta_updated.drop(labels=[i], axis=0, errors='raise')
                    elif '00-' in file_name:
                        print('Removing \'%s\' from updated metadata dataframe as \'%s\' is one of the weird 00-... samples that didn\'t download correctly.' % (new_path, file_name))
                        df_meta_updated = df_meta_updated.drop(labels=[i], axis=0, errors='raise')
                    else:
                        print(err)
                        exit(-1)
                    # df_meta_updated.to_pickle(path='D:\\data\\GoingDeeperData\\images\\df_meta.pkl')
            df_meta_updated.at[i, 'file_path'] = new_path
        else:
            df_meta_updated = df_meta_updated.drop(labels=[i], axis=0, errors='raise')

    df_meta_updated.to_pickle(path='D:\\data\\GoingDeeperData\\images\\df_meta.pkl')


def main():
    merge_train_test_into_single_dir()


if __name__ == '__main__':
    main()


