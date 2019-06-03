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
import statistics
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class ImageExecutor:

    img_root_dir = None
    logging_dir = None
    accepted_extensions = None
    df_images = None
    image_lists = None

    def __init__(self, img_root_dir, logging_dir, min_num_images_per_class, accepted_extensions):
        self.img_root_dir = img_root_dir
        self.logging_dir = logging_dir
        self.min_num_images_per_class = min_num_images_per_class
        self.accepted_extensions = accepted_extensions
        self.df_images = None
        self.cleaned_images = False
        self._summary_stats = {}
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
            if 'var.' in label or 'variety' in label:
                species_with_variety_info.append(label)
            elif 'subsp.' in label or 'subspecies' in label:
                species_with_subspecies_info.append(label)
        tf.logging.info(msg='\t\t\tDetected %d unique labels with variety (varietas) designation.' % len(species_with_variety_info))
        self._summary_stats['num_raw_labels_with_var_designation'] = len(species_with_variety_info)
        tf.logging.info(msg='\t\t\tDetected %d unique labels with subspecies designation.' % len(species_with_subspecies_info))
        self._summary_stats['num_raw_labels_with_subspecies_designation'] = len(species_with_subspecies_info)

        species_with_variety_old_label_mappings = OrderedDict()
        tf.logging.info(msg='\t\tRemapping class labels with \'varietas\' designation...')
        ''' Remove 'var.' varietas from target label and merge keys: '''
        for label in species_with_variety_info:
            if 'var.' in label:
                label_sans_var = label.split('var.')[0].strip()
            else:
                # 'variety' in label:
                label_sans_var = label.split('variety')[0].strip()
            if label_sans_var in unique_labels:
                # This class should be merged with an existing one:
                self.image_lists[label_sans_var].extend(self.image_lists[label])
                self.image_lists.pop(label)
                tf.logging.info(msg='\t\t\tConflicting classes. Remapped class label \'%s\' to existing label: \'%s\''
                                       % (label, label_sans_var))
            else:
                # No need to merge class with existing, but shorten the name:
                self.image_lists[label_sans_var] = self.image_lists[label]
                self.image_lists.pop(label)
                tf.logging.info(msg='\t\t\tNo conflicting classes. Remapped class label \'%s\' to \'%s\''
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
            if 'subsp.' in label:
                label_sans_subsp = label.split('subsp.')[0].strip()
            else:
                # 'subspecies' in label:
                label_sans_subsp = label.split('subspecies')[0].strip()
            if label_sans_subsp in unique_labels:
                # This class should be merged with an existing one:
                tf.logging.info(msg='\t\t\tConflicting classes. Remapped class label \'%s\' to existing label: \'%s\''
                                       % (label, label_sans_subsp))
                self.image_lists[label_sans_subsp].extend(self.image_lists[label])
                self.image_lists.pop(label)
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
        self._summary_stats['num_raw_class_labels_with_genus_level_rank_only'] = len(genus_level_labels)
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
        self._summary_stats['num_raw_class_labels_with_taxonomic_authorship_info'] = len(labels_with_authorship_info)
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
        self._summary_stats['num_raw_class_labels_with_hybrid_genera'] = len(labels_with_hybrid_genera)
        for label in labels_with_hybrid_genera:
            self.image_lists.pop(label)
            tf.logging.info(msg='\t\t\tDropping label with designation hybrid-genus: %s' % label)
        with open(os.path.join(self.logging_dir, 'removed_labels_hybrid_genera.txt'), 'w') as fp:
            for label in labels_with_hybrid_genera:
                fp.write('\'%s\'\n' % label)
        tf.logging.info(msg='\t\tExported a list of excluded class labels with hybrid-genus to: \'%s\'' % (self.logging_dir + '\\removed_labels_hybrid_genera.txt'))
        return self.image_lists

    def _remove_user_identified_abnormalities(self):
        explicit_mappings_incorrect_to_correct = OrderedDict({
            'acer pennsylvanicum': 'acer pensylvanicum',
            'asclepias incarnata pulchra': 'asclepias incarnata',
            'agastache scrophulariaefolia': 'agastache scrophulariifolia',
            'agrostis hiemalis': 'agrostis hyemalis',
            'amphicarpa bracteata': 'amphicarpaea bracteata',
            'carex pennsylvanica': 'carex pensylvanica',
            'andropogon gerardi': 'andropogon gerardii',
            'zephyranthes atamasco' : 'zephyranthes atamasca',
            'viola rafinesquei': 'viola rafinesquii',
            'viburnum rafinesqueanum': 'viburnum rafinesquianum',
            'spirodela polyrrhiza': 'spirodela polyrhiza'
        })
        num_class_labels_manually_remapped = 0
        encountered_mappings = {}
        for incorrect_label, correct_label in explicit_mappings_incorrect_to_correct.items():
            if incorrect_label in self.image_lists.keys():
                if incorrect_label not in encountered_mappings:
                    encountered_mappings[incorrect_label] = correct_label
                if correct_label in self.image_lists.keys():
                    self.image_lists[correct_label].extend(self.image_lists[incorrect_label])
                    self.image_lists.pop(incorrect_label)
                else:
                    self.image_lists[correct_label] = self.image_lists[incorrect_label]
                    self.image_lists.pop(incorrect_label)
                num_class_labels_manually_remapped += 1
        self._summary_stats['num_class_labels_manually_remapped'] = num_class_labels_manually_remapped
        with open(os.path.join(self.logging_dir, 'labels_with_user_identified_anomalies_old_to_new.txt'), 'w') as fp:
            for old_label, new_label in encountered_mappings.items():
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

    def _run_sanity_checks_on_cleaned_data(self):
        # for clss_label, image_paths in self.image_lists.items():
        #     assert len(clss_label.split(' ')) == 2, 'Length assertion failed for class label: \'%s\'' % clss_label
        unique_class_labels = np.unique(list(self.image_lists.keys()))
        with open(os.path.join(self.logging_dir, 'cleaned_unique_labels.txt'), 'w') as fp:
            for label in unique_class_labels:
                fp.write('\'%s\'\n' % label)

    def _ensure_min_num_sample_images(self):
        num_cleaned_labels_with_less_than_min_num_samples = 0
        cleaned_labels_with_more_than_min_num_samples = []
        image_lists = copy.deepcopy(self.image_lists)
        for label in self.image_lists:
            if len(self.image_lists[label]) <= self.min_num_images_per_class:
                image_lists.pop(label)
                num_cleaned_labels_with_less_than_min_num_samples += 1
            else:
                cleaned_labels_with_more_than_min_num_samples.append(label)
        self.image_lists = image_lists
        self._summary_stats['num_cleaned_labels_with_less_than_min_num_samples'] = num_cleaned_labels_with_less_than_min_num_samples

        with open(os.path.join(self.logging_dir, 'cleaned_labels_with_sample_count_enforced.txt'), 'w') as fp:
            for label in cleaned_labels_with_more_than_min_num_samples:
                fp.write('\'%s\'\n' % label)
        return self.image_lists

    def _calculate_summary_statistics(self):
        assert self.cleaned_images is True
        num_unique_cleaned_class_labels = len(self.image_lists.keys())
        num_present_sample_images = sum(len(lst) for lst in self.image_lists.values())  # may not be valid jpegs
        enforced_min_num_samples_per_class = self.min_num_images_per_class
        class_labels_and_image_counts = {}
        for class_label, image_list in self.image_lists.items():
            if class_label not in class_labels_and_image_counts:
                class_labels_and_image_counts[class_label] = len(image_list)
            else:
                class_labels_and_image_counts[class_label] += len(image_list)
        actual_min_num_samples_per_class = min(class_labels_and_image_counts.values())
        actual_max_num_samples_per_class = max(class_labels_and_image_counts.values())
        mean_num_samples_per_class = np.mean(list(class_labels_and_image_counts.values()))
        median_num_samples_per_class = statistics.median(list(class_labels_and_image_counts.values()))
        mode_num_samples_per_class = statistics.mode(class_labels_and_image_counts.values())
        mode_count = len([img_count for clss_label, img_count in class_labels_and_image_counts.items() if img_count == mode_num_samples_per_class])
        with open(os.path.join(self.logging_dir, 'image_executor_summary_stats.txt'), 'w') as fp:
            fp.write('Number of Unique Cleaned Class Labels: %d\n' % num_unique_cleaned_class_labels)
            fp.write('Total Number of Present (not necessarily readable) Sample Images: %d\n' % num_present_sample_images)
            fp.write('Enforced Minimum Number of Samples-per-Class: %d\n' % enforced_min_num_samples_per_class)
            fp.write('Actual Minimum Number of Samples-per-Class: %d\n' % actual_min_num_samples_per_class)
            fp.write('Enforced Maximum Number of Samples-per-Class: 1^{27} - 1 ~= 134M\n')
            fp.write('Actual Maximum Number of Samples-per-Class: %d\n' % actual_max_num_samples_per_class)
            fp.write('Mean Number of Samples-per-Class: %.2f\n' % mean_num_samples_per_class)
            fp.write('Median Number of Samples-per-Class: %d\n' % median_num_samples_per_class)
            fp.write('Mode Number of Samples-per-Class: %d (%d counts)\n' % (mode_num_samples_per_class, mode_count))

    def _clean_images(self):
        # self.df_images = self._get_raw_image_lists_df()
        tf.logging.info(msg='Populating raw image lists prior to cleaning...')
        self.image_lists = self._get_raw_image_lists()
        # These are the raw class labels that have at least one downloaded image:
        self._summary_stats['num_raw_class_labels_with_images'] = len(self.image_lists.keys())
        # TODO: check all the acquired images to see if they are actually decode-able in tensorflow.
        tf.logging.info(msg='Done, obtained raw image lists. Now cleaning class label: \'scientificName\' in the obtained raw data...')
        self.image_lists = self._clean_scientific_name()
        tf.logging.info(msg='Done, cleaned class label: \'scientificName\'. Now removing class labels with less than %d images...' % self.min_num_images_per_class)
        self._ensure_min_num_sample_images()
        tf.logging.info(msg='Running sanity checks on cleaned data...')
        self._run_sanity_checks_on_cleaned_data()
        tf.logging.info(msg='Sanity checks complete. ImageExecutor instance flagging images as cleaned.')
        self.cleaned_images = True
        tf.logging.info(msg='Calculating summary statistics...')
        self._calculate_summary_statistics()
        tf.logging.info(msg='Exported relevant summary statistics to: \'%s\'' % os.path.join(self.logging_dir, 'image_executor_summary_stats.txt'))

    def get_image_lists(self):
        if self.cleaned_images:
            return self.image_lists
        else:
            self._clean_images()
            return self.image_lists

    def get_random_sample_images(self, num_random_images):
        if not self.cleaned_images:
            self._clean_images()
        random_class_indices = np.random.randint(0, len(self.image_lists.keys()), size=num_random_images)
        random_class_labels = [list(self.image_lists.keys())[idx] for idx in random_class_indices]
        random_relative_indices = [np.random.randint(0, len(self.image_lists[rand_class_label])) for rand_class_label in random_class_labels]
        return [self.image_lists[rand_class_label][rand_relative_idx] for rand_class_label, rand_relative_idx in zip(random_class_labels, random_relative_indices)]

    def display_sample_images(self, sample_image_paths):
        images = []
        for img_path in sample_image_paths:
            img = mpimg.imread(img_path)
            images.append(img)
        fig, axes = plt.subplots(2, 2)
        print(axes)
        axes[0, 0].imshow(images[0])
        axes[0, 0].set_aspect('equal')
        axes[0, 1].imshow(images[1])
        axes[0, 1].set_aspect('equal')
        axes[1, 0].imshow(images[2])
        axes[1, 0].set_aspect('equal')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()


def main(root_dir, logging_dir):
    img_executor = ImageExecutor(img_root_dir=root_dir, logging_dir=logging_dir, accepted_extensions=['jpg', 'jpeg'], min_num_images_per_class=20)
    image_lists = img_executor.get_image_lists()
    random_images = img_executor.get_random_sample_images(num_random_images=3)
    print(random_images)
    img_executor.display_sample_images(random_images)


if __name__ == '__main__':
    # Logging verbosity:
    tf.logging.set_verbosity(tf.logging.INFO)

    # BOON Configuration:
    root_dir = 'D:\\data\\BOON\\images'
    logging_dir = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\DataAcquisition\\CleaningResults\\BOON'
    main(root_dir=root_dir, logging_dir=logging_dir)

    # GoingDeeper Configuration:
    # root_dir = 'D:\\data\\GoingDeeperData\\images'
    # logging_dir = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\DataAcquisition\\CleaningResults\\GoingDeeper'
    # main(root_dir=root_dir, logging_dir=logging_dir)

    # SERNEC Configuration:
    # root_dir = 'D:\\data\\SERNEC\\images'
    # logging_dir = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\DataAcquisition\\CleaningResults\\SERNEC'
    # main(root_dir=root_dir, logging_dir=logging_dir)
