"""
ImageExecutor.py
Performs aggregation across image directories and the cleaning of class labels. Use this class to obtain image_lists
that can then be fed into the BottleneckExecutor. Image lists returned by this classes' get_image_lists() method are
guaranteed to be cleaned prior to return.
"""
import pandas as pd
import tensorflow as tf
AUTOTUNE = tf.contrib.data.AUTOTUNE
import os
from collections import OrderedDict
import numpy as np
import copy


class ImageExecutor:

    img_root_dir = None
    accepted_extensions = None
    min_num_imgs_per_class = 20
    df_images = None
    image_lists = None
    class_labels_one_hot_encoded_list = None
    train_image_lists = None
    val_image_lists = None
    test_image_lists = None

    def __init__(self, img_root_dir, accepted_extensions, min_num_imgs_per_class=20):
        self.img_root_dir = img_root_dir
        self.accepted_extensions = accepted_extensions
        self.df_images = None
        if min_num_imgs_per_class:
            self.min_num_imgs_per_class = min_num_imgs_per_class
        self._clean_images()

    def get_class_labels_one_hot_encoded(self):
        if self.cleaned_images:
            return self.class_labels_one_hot_encoded_list
        else:
            self._clean_images()
            return self.class_labels_one_hot_encoded_list

    def _get_raw_image_lists_df(self):
        col_names = {'class', 'path', 'bottleneck'}
        df_images = pd.DataFrame(columns=col_names)
        sub_dirs = sorted(x[0] for x in tf.gfile.Walk(self.img_root_dir))
        for i, sub_dir in enumerate(sub_dirs):
            file_list = []
            dir_name = os.path.basename(sub_dir)
            if i == 0:
                # skip root dir
                continue
            tf.logging.info('Locating images in: \'%s\'' % dir_name)
            for extension in self.accepted_extensions:
                file_glob = os.path.join(self.img_root_dir, dir_name, '*.' + extension)
                file_list.extend(tf.gfile.Glob(file_glob))
            if not file_list:
                tf.logging.warning(msg='\tNo files found in \'%s\'. Class label omitted from data sets.' % dir_name)
            label_name = dir_name.lower()
            for file in file_list:
                series = pd.Series({'class': label_name, 'path': file, 'bottleneck': None})
                df_images = df_images.append(series, ignore_index=True)
        df_images['class'] = df_images['class'].astype('category')
        return df_images

    def _get_raw_image_lists(self):
        image_lists = OrderedDict()
        sub_dirs = sorted(x[0] for x in tf.gfile.Walk(self.img_root_dir))
        for i, sub_dir in enumerate(sub_dirs):
            file_list = []
            dir_name = os.path.basename(sub_dir)
            if i == 0:
                # skip root dir
                continue
            tf.logging.info('Locating images in: \'%s\'' % dir_name)
            for extension in self.accepted_extensions:
                if extension == '':
                    tf.logging.warning(msg='\tOverrode your allowed extensions. Ignoring files with extension \'\' due '
                                           'to high possibility of corrupted files.')
                    pass
                else:
                    file_glob = os.path.join(self.img_root_dir, dir_name, '*.' + extension)
                    file_list.extend(tf.gfile.Glob(file_glob))
            if not file_list:
                tf.logging.warning(msg='\tNo files found in \'%s\'. Class label omitted from data sets.' % dir_name)
            label_name = dir_name.lower()
            if label_name not in image_lists:
                image_lists[label_name] = file_list
            else:
                image_lists[label_name].extend(file_list)
        return image_lists

    def _remove_taxon_ranks_and_remap_categoricals(self):
        """
        _remove_taxon_rank_and_remap_categoricals: Renames species whose ScientificName contain 'var.' (short for
         varietas) as in the example: 'Andropogon glomeratus var. glomeratus' which would be remapped to:
         'Andropogon glomeratus'. Since an existing 'Andropogon glomeratus' category exists, this method will
         re-label every class with 'Andropogon glomeratus var. glomeratus' as simply: 'Andropogon glomeratus',
         thereby merging the categorical data types. This method additionally removes species whose scientificName
         contain a subspecies designation: 'subsp.'.
         For additional information (see: https://en.wikipedia.org/wiki/Infraspecific_name and
                http://rs.tdwg.org/dwc/terms/taxonRank).
         subspecies, varietas, forma, species, genus
        :return:
        """
        unique_labels = self.df_images.unique()
        # Remove Infraspecific Name (see: https://en.wikipedia.org/wiki/Infraspecific_name)
        species_with_variety_info = []
        species_with_subspecies_info = []
        for label in unique_labels:
            if 'var.' in label:
                species_with_variety_info.append(label)
            elif 'subsp.' in label:
                species_with_subspecies_info.append(label)

        ''' Remove 'var.' varietas from target label and merge categorical data types: '''
        new_categorical_mapping = {}
        for label in species_with_variety_info:
            label_sans_var = label.split('var.')[0].strip()
            if label_sans_var in unique_labels:
                # This class should be merged with an existing one:
                tf.logging.warning(msg='Conflicting classes after dropping designation varietas:'
                                       '\n\tOriginal label: \'%s\' New label: \'%s\'' % (label, label_sans_var))
                if label not in new_categorical_mapping:
                    new_categorical_mapping[label] = label_sans_var
        # Update the dataframe:
        self.df_images['class'] = self.df_images['class'].rename(columns=new_categorical_mapping)
        self.df_images['class'] = self.df_images['class'].cat.remove_unused_categories()

        ''' Remove 'var.' varietas from target label and merge categorical data types: '''
        new_categorical_mapping = {}
        for label in species_with_subspecies_info:
            label_sans_subsp = label.split('subsp.')[0].strip()
            if label_sans_subsp in unique_labels:
                # This class should be merged with an existing one:
                tf.logging.warning(msg='Conflicting classes after dropping designation subspecies:'
                                       '\n\tOriginal label: \'%s\' New label: \'%s\'' % (label, label_sans_subsp))
                if label not in new_categorical_mapping:
                    new_categorical_mapping[label] = label_sans_subsp
        # Update the dataframe:
        self.df_images['class'] = self.df_images['class'].rename(columns=new_categorical_mapping)
        self.df_images['class'] = self.df_images['class'].cat.remove_unused_categories()

    def _remove_taxon_ranks_and_merge_keys(self):
        unique_labels = np.unique(list(self.image_lists.keys()))
        species_with_variety_info = []
        species_with_subspecies_info = []
        for label in unique_labels:
            if 'var.' in label:
                species_with_variety_info.append(label)
            elif 'subsp.' in label:
                species_with_subspecies_info.append(label)
        ''' Remove 'var.' varietas from target label and merge keys: '''
        for label in species_with_variety_info:
            label_sans_var = label.split('var.')[0].strip()
            if label_sans_var in unique_labels:
                # This class should be merged with an existing one:
                tf.logging.warning(msg='Conflicting classes after dropping designation varietas:'
                                       '\n\tOriginal label: \'%s\' New label: \'%s\'' % (label, label_sans_var))
                self.image_lists[label_sans_var].extend(self.image_lists[label])
                self.image_lists.pop(label)
            else:
                # No need to merge class with existing, but shorten the name:
                self.image_lists[label_sans_var] = self.image_lists[label]
                self.image_lists.pop(label)

        ''' Remove 'subsp.' subspecies designation from target label and merge keys: '''
        for label in species_with_subspecies_info:
            label_sans_subsp = label.split('subsp.')[0].strip()
            if label_sans_subsp in unique_labels:
                # This class should be merged with an existing one:
                tf.logging.warning(msg='Conflicting classes after dropping designation subspecies:'
                                       '\n\tOriginal label: \'%s\' New label: \'%s\'' % (label, label_sans_subsp))
                self.image_lists[label_sans_subsp].extend(self.image_lists[label])
                self.image_lists.pop(label)
            else:
                # No need to merge class with existing, but shorten the name:
                self.image_lists[label_sans_subsp] = self.image_lists[label]
                self.image_lists.pop(label)
        return self.image_lists

    def _remove_genus_level_scientific_names(self):
        genus_level_labels = []
        for label in self.image_lists.keys():
            if len(label.split(' ')) == 1:
                genus_level_labels.append(label)
        for genus in genus_level_labels:
            tf.logging.warning(msg='Removing species with genus level designation: %s' % genus)
            self.image_lists.pop(genus)
        return self.image_lists

    def _remove_authorship_info_and_merge_keys(self):
        labels_with_authorship_info = []
        for label in self.image_lists.keys():
            if '(' in label or ')' in label:
                labels_with_authorship_info.append(label)
            elif '.' in label and len(label.split(' ')) > 1:
                labels_with_authorship_info.append(label)
        for label in labels_with_authorship_info:
            label_sans_authorship = ''
            if '(' in label:
                label_sans_authorship = label.split('(')[0].strip()
            elif '.' in label:
                label_delimited = label.split(' ')
                for sub_label in label_delimited:
                    if '.' in sub_label:
                        break
                    else:
                        label_sans_authorship = label_sans_authorship + ' %s' % sub_label
                label_sans_authorship = label_sans_authorship.strip()
            if label_sans_authorship in self.image_lists.keys():
                tf.logging.warning(msg='Conflicting classes after dropping designation taxonomic-authorship:'
                                       '\n\tOriginal label: \'%s\' New label: \'%s\'' % (label, label_sans_authorship))
                # Merge classes
                self.image_lists[label_sans_authorship].extend(self.image_lists[label])
                self.image_lists.pop(label)
            else:
                # Update class name
                self.image_lists[label_sans_authorship] = self.image_lists[label]
                self.image_lists.pop(label)
        return self.image_lists

    def _remove_hybrid_genera_scientific_names(self):
        labels_with_hybrid_genera = []
        for label in self.image_lists.keys():
            for sub_label in label.split(' '):
                if sub_label == 'x':
                    labels_with_hybrid_genera.append(label)
                    break
        for label in labels_with_hybrid_genera:
            self.image_lists.pop(label)
            tf.logging.warning(msg='Dropping label with designation hybrid-genus: %s' % label)
        return self.image_lists

    def _remove_user_identified_abnormalities(self):
        problem_labels = ['asclepias incarnata pulchra']
        for label in problem_labels:
            if label in self.image_lists.keys():
                if label == 'asclepias incarnata pulchra':
                    if 'asclepias incarnata' in self.image_lists.keys():
                        self.image_lists['asclepias incarnata'].extend(self.image_lists[label])
                        self.image_lists.pop(label)
                    else:
                        self.image_lists['asclepias incarnata'] = self.image_lists[label]
                        self.image_lists.pop(label)
        return self.image_lists

    def _clean_scientific_name(self):
        # DwC scientificName: 'Genus + specificEpithet
        # Rename categoricals with 'genus species var. subspecies' to just 'genus species'. Also,
        #   rename 'genus species subsp. subspecie' to just 'genus species':
        self.image_lists = self._remove_taxon_ranks_and_merge_keys()
        # Remove 'Genus' level scientific names, it is assumed there will be too much variation within one genus:
        self.image_lists = self._remove_genus_level_scientific_names()
        # Remove taxonomic authorship information (this is not pertinent to this method of classification):
        self.image_lists = self._remove_authorship_info_and_merge_keys()
        # Remove hybrid genera (genera derived from cross breeding plants of two different genuses) as this violates
        #   the assumption of statistical independence in class labels:
        self.image_lists = self._remove_hybrid_genera_scientific_names()
        # Remove user-identified abnormalities:
        self.image_lists = self._remove_user_identified_abnormalities()
        return self.image_lists

    def _clean_images(self):
        # self.df_images = self._get_raw_image_lists_df()
        self.image_lists = self._get_raw_image_lists()
        self.image_lists = self._clean_scientific_name()
        image_lists = copy.deepcopy(self.image_lists)
        for label in self.image_lists:
            if len(self.image_lists[label]) <= self.min_num_imgs_per_class:
                image_lists.pop(label)
        self.image_lists = image_lists
        self.class_labels_one_hot_encoded_list = list(image_lists.keys())
        self.cleaned_images = True

    def get_image_lists(self):
        if not self.cleaned_images:
            self._clean_images()
        return self.image_lists

    # def _partition_image_lists(image_lists, train_percent, val_percent, test_percent, random_state):
    #     """
    #     _partition_image_lists: Partitions the provided dict of class labels and file paths into training, validation, and
    #         testing datasets.
    #     :param image_lists: <collections.OrderedDict> A dictionary indexed by class label, which provides as its value a
    #         list of file paths for images belonging to the chosen key/species/class-label.
    #     :param train_percent: What percentage of the training data is to remain in the training set.
    #     :param val_percent: What percentage of the remaining training data (after removing test set) is to be allocated
    #         for a validation set.
    #     :param test_percent: What percentage of the training data is to be allocated to a testing set.
    #     :param random_state: A seed for the random number generator controlling the stratified partitioning.
    #     :return partitioned_image_lists: <collections.OrderedDict> A dictionary indexed by class label which returns another
    #         dictionary indexed by the dataset type {'train','val','test'} which in turn, returns the list of image file
    #         paths that correspond to the chosen class label and that reside in the chosen dataset.
    #     """
    #     partitioned_image_lists = OrderedDict()
    #     for class_label, image_paths in image_lists.items():
    #         class_label_train_images, class_label_test_images = model_selection.train_test_split(
    #             image_paths, train_size=train_percent,
    #             test_size=test_percent, shuffle=True,
    #             random_state=random_state
    #         )
    #         class_label_train_images, class_label_val_images = model_selection.train_test_split(
    #             class_label_train_images, train_size=train_percent,
    #             test_size=val_percent, shuffle=True,
    #             random_state=random_state
    #         )
    #         partitioned_image_lists[class_label] = {
    #             'train': class_label_train_images,
    #             'val': class_label_val_images,
    #             'test': class_label_test_images
    #         }
    #     return partitioned_image_lists

    def get_image_lists_as_df(self):
        """
        get_cleaned_image_lists_as_df: Designed to operate only on cleaned image lists returned by public getter method
            self.get_image_lists(). This method aggregates as a dataframe and attempts to load all images into said
            dataframe for persistence in memory.
        :param image_lists:
        :return:
        """
        if self.cleaned_images:
            image_lists = self.image_lists
        else:
            image_lists = self.get_image_lists()
        df_images_empty = pd.DataFrame(columns=['class', 'path', 'img'])
        for clss in image_lists.keys():
            for image_path in image_lists[clss]:
                new_img_entry = pd.Series([clss, image_path, None], index=['class', 'path', 'img'])
                df_images_empty = df_images_empty.append(new_img_entry, ignore_index=True)
        df_images_empty['class'] = df_images_empty['class'].astype('category')
        df_images = df_images_empty.copy(deep=True)
        num_imgs_loaded = 0
        for i, (clss, series) in enumerate(df_images_empty.iterrows()):
            img_path = series['path']
            if not tf.gfile.Exists(img_path):
                tf.logging.fatal('Image file does not exist: \'%s\'' % img_path)
                exit(-1)
            img_data = tf.gfile.GFile(img_path, 'rb').read()
            df_images.at[i, 2] = img_data
            num_imgs_loaded += 1
            if num_imgs_loaded % 100 == 0:
                tf.logging.info(msg='Loaded %d img vectors into dataframe.' % num_imgs_loaded)
            if num_imgs_loaded % 1000 == 0:
                tf.logging.info(msg='Loaded %d img vectors into dataframe. Backing up dataframe to: \'%s\'' % (num_imgs_loaded, '?'))
        tf.logging.info(msg='Finished generating image dataframe. Saving dataframe to: \'%s\'' % '?')
        return df_images

    @staticmethod
    def _preprocess_image(raw_image, height=299, width=299, num_channels=3):
        """

        :param raw_image:
        :param height:
        :param width:
        :param num_channels:
        :source: https://www.tensorflow.org/tutorials/load_data/images
        :return:
        """
        image = tf.image.decode_jpeg(raw_image, channels=num_channels)
        image = tf.image.resize_images(image, [height, width])
        image /= 255.0  # normalize to [0,1] range
        return image

    def _load_and_preprocess_image(self, img_path):
        img_raw = tf.read_file(img_path)
        return self._preprocess_image(img_raw)

    @staticmethod
    def get_partitioned_image_lists_same_as_dataframes(df_train_bottlenecks, df_val_bottlenecks, df_test_bottlenecks):
        train_image_lists = OrderedDict()
        val_image_lists = OrderedDict()
        test_image_lists = OrderedDict()
        for i, series in enumerate(df_train_bottlenecks.iterrows()):
            clss = series[1]['class']
            path = series[1]['path']
            if clss not in train_image_lists:
                train_image_lists[clss] = [path]
            else:
                train_image_lists[clss].append(path)
        for i, series in enumerate(df_val_bottlenecks.iterrows()):
            clss = series[1]['class']
            path = series[1]['path']
            if clss not in val_image_lists:
                val_image_lists[clss] = [path]
            else:
                val_image_lists[clss].append(path)
        for i, series in enumerate(df_test_bottlenecks.iterrows()):
            clss = series[1]['class']
            path = series[1]['path']
            if clss not in test_image_lists:
                test_image_lists[clss] = [path]
            else:
                test_image_lists[clss].append(path)
        return train_image_lists, val_image_lists, test_image_lists

    def get_partitioned_image_list_as_tensor_flow_dataset(self, image_lists_subset):
        subset_image_paths = []
        subset_image_labels_one_hot = []
        for clss, clss_img_paths in image_lists_subset.items():
            for img_path in clss_img_paths:
                subset_image_paths.append(img_path)
                subset_image_labels_one_hot.append(self.class_labels_one_hot_encoded_list.index(clss))
        path_tf_ds = tf.data.Dataset.from_tensor_slices(subset_image_paths)
        subset_image_tf_ds = path_tf_ds.map(self._load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
        subset_label_tf_ds = tf.data.Dataset.from_tensor_slices(tf.cast(subset_image_labels_one_hot, tf.int64))
        subset_image_label_tf_ds = tf.data.Dataset.zip((subset_image_tf_ds, subset_label_tf_ds))
        return subset_image_label_tf_ds

    def get_image_lists_as_tensor_flow_dataset(self):
        """
        get_image_lists_as_tensor_flow_dataset:
        :source: https://www.tensorflow.org/tutorials/load_data/images
        :return:
        """
        if self.cleaned_images:
            image_lists = self.image_lists
        else:
            image_lists = self.get_image_lists()
        all_image_paths = []
        all_image_label_indices = []
        for i, (clss_label, img_paths) in enumerate(image_lists.items()):
            for img_path in img_paths:
                all_image_paths.append(img_path)
                all_image_label_indices.append(i)
        tf_ds_paths = tf.data.Dataset.from_tensor_slices(all_image_paths)
        tf_ds_images = tf_ds_paths.map(self._load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
        label_names = list(image_lists.keys())
        tf_ds_labels = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_label_indices, tf.int64))
        tf_ds_images_and_labels = tf.data.Dataset.zip((tf_ds_images, tf_ds_labels))
        tf.logging.info(msg='Created TensorFlow datset from labels and images:')
        tf.logging.info(msg='\timage shape: %s' % tf_ds_images_and_labels.output_shapes[0])
        tf.logging.info(msg='\tlabel shape: %s' % tf_ds_images_and_labels.output_shapes[1])
        tf.logging.info(msg='\ttypes: {}'.format(tf_ds_images_and_labels.output_types))
        # print()
        tf.logging.info(msg='\t {}'.format(tf_ds_images_and_labels))
        # label_to_index = dict((name, index) for index, name in enumerate(label_names))
        # all_image_labels = [label_to_index[label_name] for label in ]
        return tf_ds_images_and_labels



def main(root_dir):
    img_executor = ImageExecutor(img_root_dir=root_dir, accepted_extensions=['jpg', 'jpeg'])
    image_lists = img_executor.get_image_lists()
    # TensorFlow Dataset support for Keras integration:
    # tf_ds = img_executor.get_image_lists_as_tensor_flow_dataset()
    # df_images = img_executor.get_image_lists_as_df()


if __name__ == '__main__':
    # Logging verbosity:
    tf.logging.set_verbosity(tf.logging.INFO)

    # GoingDeeper Configuration:
    # main(root_dir='D:\\data\\GoingDeeperData\\images')

    # BOON Configuration:
    main(root_dir='D:\\data\\BOON\\images')
