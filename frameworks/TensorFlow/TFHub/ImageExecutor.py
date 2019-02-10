"""

"""
import pandas as pd
import tensorflow as tf
import os
from collections import OrderedDict
import numpy as np

class ImageExecutor:

    MIN_NUM_SAMPLES_PER_CLASS = 20
    img_root_dir = None
    accepted_extensions = None
    df_images = None
    image_lists = None

    def __init__(self, img_root_dir, accepted_extensions):
        self.img_root_dir = img_root_dir
        self.accepted_extensions = accepted_extensions
        self.df_images = None
        self._clean_images()
        self.cleaned_images = True

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
            if '(' in label:
                label_sans_authorship = label.split('(')[0].strip()
            elif '.' in label:
                label_sans_authorship = ''
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

    def get_image_lists(self):
        if self.cleaned_images:
            return self.image_lists
        else:
            self._clean_images()
            self.cleaned_images = True
            return self.image_lists


def main(root_dir):
    img_executor = ImageExecutor(img_root_dir=root_dir, accepted_extensions=['jpg', 'jpeg'])
    img_executor._clean_images()


if __name__ == '__main__':
    # Logging verbosity:
    tf.logging.set_verbosity(tf.logging.INFO)

    # GoingDeeper Configuration:

    # BOON Configuration:
    main(root_dir='D:\\data\\BOON\\images')
