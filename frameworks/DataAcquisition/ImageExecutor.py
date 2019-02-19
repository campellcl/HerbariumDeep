"""
ImageExecutor.py
Performs aggregation across image directories and the cleaning of class labels. Use this class to obtain image_lists
that can then be fed into the BottleneckExecutor. Image lists returned by this classes' get_image_lists() method are
guaranteed to be cleaned prior to return.
"""
import pandas as pd
import tensorflow as tf
import os
from collections import OrderedDict
import numpy as np
import copy


class ImageExecutor:

    img_root_dir = None
    logging_dir = None
    accepted_extensions = None
    df_images = None
    image_lists = None

    def __init__(self, img_root_dir, logging_dir, accepted_extensions):
        self.img_root_dir = img_root_dir
        self.logging_dir = logging_dir
        self.accepted_extensions = accepted_extensions
        self.df_images = None
        self._clean_images()

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
                tf.logging.info(msg='\tNo files found in \'%s\'. Class label omitted from data sets.' % dir_name)
            label_name = dir_name.lower()
            for file in file_list:
                series = pd.Series({'class': label_name, 'path': file, 'bottleneck': None})
                df_images = df_images.append(series, ignore_index=True)
        df_images['class'] = df_images['class'].astype('category')
        return df_images

    def _get_raw_image_lists(self):
        image_lists = OrderedDict()
        labels_with_no_files = []
        sub_dirs = sorted(x[0] for x in tf.gfile.Walk(self.img_root_dir))
        for i, sub_dir in enumerate(sub_dirs):
            file_list = []
            dir_name = os.path.basename(sub_dir)
            if i == 0:
                # skip root dir
                continue
            tf.logging.info('\tLocating images in: \'%s\'' % dir_name)
            for extension in self.accepted_extensions:
                file_glob = os.path.join(self.img_root_dir, dir_name, '*.' + extension)
                file_list.extend(tf.gfile.Glob(file_glob))
            if not file_list:
                tf.logging.info(msg='\t\tNo files found in \'%s\'. Class label omitted from data sets.' % dir_name)
                labels_with_no_files.append(os.path.join(self.img_root_dir, dir_name))
            label_name = dir_name.lower()
            if label_name not in image_lists:
                image_lists[label_name] = file_list
            else:
                image_lists[label_name].extend(file_list)
        with open(os.path.join(self.logging_dir, 'raw_labels_no_images.txt'), 'w') as fp:
            for label in labels_with_no_files:
                fp.write(label + '\n')
        tf.logging.info(msg='\tExported list of excluded directories with no images (for verification) to \'%s\''
                            % (self.logging_dir + '\\raw_labels_no_images.txt'))
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
                tf.logging.info(msg='Conflicting classes after dropping designation varietas:'
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
                tf.logging.info(msg='Conflicting classes after dropping designation subspecies:'
                                       '\n\tOriginal label: \'%s\' New label: \'%s\'' % (label, label_sans_subsp))
                if label not in new_categorical_mapping:
                    new_categorical_mapping[label] = label_sans_subsp
        # Update the dataframe:
        self.df_images['class'] = self.df_images['class'].rename(columns=new_categorical_mapping)
        self.df_images['class'] = self.df_images['class'].cat.remove_unused_categories()

    def _remove_taxon_ranks_and_merge_keys(self):
        unique_labels = np.unique(list(self.image_lists.keys()))
        tf.logging.info(msg='\t\tDetected %d unique labels/\'scientificName(s)\' prior to cleaning.' % len(unique_labels))
        tf.logging.info(msg='\t\tDetecting labels with variety and subspecies taxonomic designations...')
        species_with_variety_info = []
        species_with_subspecies_info = []
        for label in unique_labels:
            if 'var.' in label:
                species_with_variety_info.append(label)
            elif 'subsp.' in label:
                species_with_subspecies_info.append(label)
        tf.logging.info(msg='\t\t\tDetected %d unique labels with variety (varietas) designation.' % len(species_with_variety_info))
        tf.logging.info(msg='\t\t\tDetected %d unique labels with subspecies designation.' % len(species_with_subspecies_info))

        species_with_variety_old_label_mappings = OrderedDict()
        tf.logging.info(msg='\t\tRemapping class labels with \'varietas\' designation...')
        ''' Remove 'var.' varietas from target label and merge keys: '''
        for label in species_with_variety_info:
            label_sans_var = label.split('var.')[0].strip()
            if label_sans_var in unique_labels:
                # This class should be merged with an existing one:
                tf.logging.info(msg='\t\t\tConflicting classes. Remapped class label \'%s\' to existing label: \'%s\''
                                       % (label, label_sans_var))
                self.image_lists[label_sans_var].extend(self.image_lists[label])
                self.image_lists.pop(label)
                if label not in species_with_variety_old_label_mappings:
                    species_with_variety_old_label_mappings[label_sans_var] = [label]
                else:
                    species_with_variety_old_label_mappings[label_sans_var].append(label)
            else:
                # No need to merge class with existing, but shorten the name:
                self.image_lists[label_sans_var] = self.image_lists[label]
                self.image_lists.pop(label)
                tf.logging.info(msg='\t\t\tNo conflictig classes. Remapped class label \'%s\' to \'%s\''
                                    % (label, label_sans_var))
                if label not in species_with_variety_old_label_mappings:
                    species_with_variety_old_label_mappings[label_sans_var] = [label]
                else:
                    species_with_variety_old_label_mappings[label_sans_var].append(label)
        tf.logging.info('\t\tDone, performed remapping of %d class labels with \'varietas\' designation.' % len(species_with_variety_info))
        with open(self.logging_dir + '\\labels_with_variety_old_to_new.txt', 'w') as fp:
            for new_label, old_labels in species_with_variety_old_label_mappings.items():
                for old_label in old_labels:
                    out = '\'%s\' -> \'%s\'\n' % (old_label, new_label)
                    fp.write(out)
        tf.logging.info('\t\tExported a record of mappings from old-species-label-with-variety-info to new-species-label at location: \'%s\'' % os.path.join(self.logging_dir, '\\lables_with_variety_old_to_new.txt'))

        species_with_subspecies_old_label_mappings = OrderedDict()
        tf.logging.info(msg='\t\tRemapping class labels with \'subspecies\' designation...')
        ''' Remove 'subsp.' subspecies designation from target label and merge keys: '''
        for label in species_with_subspecies_info:
            label_sans_subsp = label.split('subsp.')[0].strip()
            if label_sans_subsp in unique_labels:
                # This class should be merged with an existing one:
                tf.logging.info(msg='\t\t\tConflicting classes. Remapped class label \'%s\' to existing label: \'%s\''
                                       % (label, label_sans_subsp))
                self.image_lists[label_sans_subsp].extend(self.image_lists[label])
                self.image_lists.pop(label)
                if label not in species_with_subspecies_old_label_mappings:
                    species_with_subspecies_old_label_mappings[label_sans_subsp] = [label]
                else:
                    species_with_subspecies_old_label_mappings[label_sans_subsp].append(label)
            else:
                # No need to merge class with existing, but shorten the name:
                self.image_lists[label_sans_subsp] = self.image_lists[label]
                self.image_lists.pop(label)
                tf.logging.info(msg='\t\t\tNo conflictig classes. Remapped class label \'%s\' to \'%s\''
                                    % (label, label_sans_subsp))
                if label not in species_with_subspecies_old_label_mappings:
                    species_with_subspecies_old_label_mappings[label_sans_subsp] = [label]
                else:
                    species_with_subspecies_old_label_mappings[label_sans_subsp].append(label)
        tf.logging.info('\t\tDone, performed remapping of %d class labels with \'subspecies\' designation.'
                        % len(species_with_subspecies_info))
        with open(self.logging_dir + '\\labels_with_subspecies_old_to_new.txt', 'w') as fp:
            for new_label, old_labels in species_with_subspecies_old_label_mappings.items():
                for old_label in old_labels:
                    out = '\'%s\' -> \'%s\'\n' % (old_label, new_label)
                    fp.write(out)
        tf.logging.info('\t\tExported a record of mappings from old-species-label-with-subspecies-info '
                        'to new-species-label at location: \'%s\''
                        % os.path.join(self.logging_dir, '\\lables_with_subspecies_old_to_new.txt'))
        return self.image_lists

    def _remove_genus_level_scientific_names(self):
        genus_level_labels = []
        for label in self.image_lists.keys():
            if len(label.split(' ')) == 1:
                genus_level_labels.append(label)
        tf.logging.info(msg='\t\tDetected %d labels designated at genus level taxonomic rank.' % len(genus_level_labels))
        for genus in genus_level_labels:
            tf.logging.info(msg='\t\tRemoving specie label with genus level designation: %s' % genus)
            self.image_lists.pop(genus)
        tf.logging.info('\t\tRemoved %d labels designated at genus level taxonomic rank.' % len(genus_level_labels))
        with open(os.path.join(self.logging_dir, 'removed_labels_genus_only.txt'), 'w') as fp:
            for label in genus_level_labels:
                fp.write('\'%s\'\n' % label)
        tf.logging.info('\t\tExported a record of removed genus level species labels for verification to: \'%s\'' % (self.logging_dir + '\\removed_labels_genus_only.txt'))
        return self.image_lists

    def _remove_authorship_info_and_merge_keys(self):
        labels_with_authorship_info = []
        for label in self.image_lists.keys():
            if '(' in label or ')' in label:
                labels_with_authorship_info.append(label)
            elif '.' in label and len(label.split(' ')) > 1:
                labels_with_authorship_info.append(label)
        tf.logging.info(msg='\t\tDetected %d labels determined to contain taxonomic authorship information.'
                            % len(labels_with_authorship_info))
        labels_with_authorship_info_mappings = OrderedDict()
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
                # Merge classes
                self.image_lists[label_sans_authorship].extend(self.image_lists[label])
                self.image_lists.pop(label)
                tf.logging.info(msg='\t\t\tConflicting classes. Remapped class label \'%s\' to existing label: \'%s\''
                                       % (label, label_sans_authorship))
            else:
                # Update class name
                self.image_lists[label_sans_authorship] = self.image_lists[label]
                self.image_lists.pop(label)
                tf.logging.info(msg='\t\t\tNo conflictig classes. Remapped class label \'%s\' to \'%s\''
                                    % (label, label_sans_authorship))
            if label_sans_authorship not in labels_with_authorship_info_mappings:
                labels_with_authorship_info_mappings[label_sans_authorship] = [label]
            else:
                labels_with_authorship_info_mappings[label_sans_authorship].append(label)
        with open(os.path.join(self.logging_dir, 'labels_with_authorship_old_to_new.txt'), 'w') as fp:
            for new_label, old_labels in labels_with_authorship_info_mappings.items():
                for old_label in old_labels:
                    fp.write('\'%s\' -> \'%s\'\n' % (old_label, new_label))
        tf.logging.info(msg='\t\tExported a record of mappings from old-species-label-with-authorship-info to new-species-label at: \'%s\'' % (self.logging_dir + '\\labels_with_authorship_old_to_new.txt'))
        return self.image_lists

    def _remove_hybrid_genera_scientific_names(self):
        labels_with_hybrid_genera = []
        for label in self.image_lists.keys():
            for sub_label in label.split(' '):
                if sub_label == 'x':
                    labels_with_hybrid_genera.append(label)
                    break
        tf.logging.info(msg='\t\tDetected %d labels with hybrid genera.' % len(labels_with_hybrid_genera))
        for label in labels_with_hybrid_genera:
            self.image_lists.pop(label)
            tf.logging.info(msg='\t\t\tDropping label with designation hybrid-genus: %s' % label)
        with open(os.path.join(self.logging_dir, 'removed_labels_hybrid_genera.txt'), 'w') as fp:
            for label in labels_with_hybrid_genera:
                fp.write('\'%s\'\n' % label)
        tf.logging.info(msg='\t\tExported a list of excluded class labels with hybrid-genus to: \'%s\'' % (self.logging_dir + '\\removed_labels_hybrid_genera.txt'))
        return self.image_lists

    def _remove_user_identified_abnormalities(self):
        problem_labels = ['asclepias incarnata pulchra']
        problem_labels_mappings = OrderedDict()
        for label in problem_labels:
            if label in self.image_lists.keys():
                if label == 'asclepias incarnata pulchra':
                    if 'asclepias incarnata' in self.image_lists.keys():
                        self.image_lists['asclepias incarnata'].extend(self.image_lists[label])
                        self.image_lists.pop(label)
                    else:
                        self.image_lists['asclepias incarnata'] = self.image_lists[label]
                        self.image_lists.pop(label)
                    problem_labels_mappings[label] = 'asclepias incarnata'
        with open(os.path.join(self.logging_dir, 'labels_with_user_identified_anomalies_old_to_new.txt'), 'w') as fp:
            for old_label, new_label in problem_labels_mappings.items():
                fp.write('\'%s\' -> \'%s\'\n' % (old_label, new_label))
        tf.logging.info(msg='\t\tExported list of anomalous user-identified class labels removed to: \'%s\'' % (self.logging_dir + '\\labels_with_user_identified_anomalies_old_to_new.txt'))
        return self.image_lists

    def _clean_scientific_name(self):
        # DwC scientificName: 'Genus + specificEpithet
        # Rename categoricals with 'genus species var. subspecies' to just 'genus species'. Also,
        #   rename 'genus species subsp. subspecie' to just 'genus species':
        tf.logging.info(msg='\tRemoving taxon ranks: {var., subsp.} and merging class labels...')
        self.image_lists = self._remove_taxon_ranks_and_merge_keys()
        tf.logging.info(msg='\tDone, purged taxonomic rank information.')

        # Remove 'Genus' level scientific names, it is assumed there will be too much variation within one genus:
        tf.logging.info(msg='\tRemoving genus level scientific names...')
        self.image_lists = self._remove_genus_level_scientific_names()
        tf.logging.info(msg='\tDone, purged genus-level scientific names.')

        # Remove taxonomic authorship information (this is not pertinent to this method of classification):
        tf.logging.info(msg='\tRemoving taxonomic authorship info...')
        self.image_lists = self._remove_authorship_info_and_merge_keys()
        tf.logging.info(msg='\tDone, purged taxonomic authorship info.')

        # Remove hybrid genera (genera derived from cross breeding plants of two different genuses) as this violates
        #   the assumption of statistical independence in class labels:
        tf.logging.info(msg='\tRemoving hybrid genera due to phenological violation of fundimental statistical inference hypothesis.')
        self.image_lists = self._remove_hybrid_genera_scientific_names()
        tf.logging.info(msg='\tDone, purged hybrid genera from class labels.')

        # Remove user-identified abnormalities:
        tf.logging.info(msg='\tRemoving user-identified abnormalities...')
        self.image_lists = self._remove_user_identified_abnormalities()
        tf.logging.info(msg='\tDone, purged user-identified anomalous class labels.')
        return self.image_lists

    def _clean_images(self):
        # self.df_images = self._get_raw_image_lists_df()
        tf.logging.info(msg='Populating raw image lists prior to cleaning...')
        self.image_lists = self._get_raw_image_lists()
        tf.logging.info(msg='Done, obtained raw image lists. Now cleaning class label: \'scientificName\' in the obtained raw data...')
        self.image_lists = self._clean_scientific_name()
        self.cleaned_images = True

    def get_image_lists(self, min_num_images_per_class):
        if not self.cleaned_images:
            self._clean_images()
        image_lists = copy.deepcopy(self.image_lists)
        for label in self.image_lists:
            if len(self.image_lists[label]) <= min_num_images_per_class:
                image_lists.pop(label)
        return image_lists


def main(root_dir, logging_dir):
    img_executor = ImageExecutor(img_root_dir=root_dir, logging_dir=logging_dir,accepted_extensions=['jpg', 'jpeg'])
    image_lists = img_executor.get_image_lists(min_num_images_per_class=20)


if __name__ == '__main__':
    # Logging verbosity:
    tf.logging.set_verbosity(tf.logging.INFO)

    # GoingDeeper Configuration:

    # BOON Configuration:
    root_dir = 'D:\\data\\BOON\\images'
    logging_dir = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\DataAcquisition\\CleaningResults\\BOON'
    main(root_dir=root_dir, logging_dir=logging_dir)

    # SERNEC Configuration:
    # main(root_dir='D:\\data\\SERNEC\\images')
