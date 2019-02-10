"""

"""
import pandas as pd
import tensorflow as tf
import os

class ImageExecutor:

    MIN_NUM_SAMPLES_PER_CLASS = 20
    img_root_dir = None
    accepted_extensions = None
    df_images = None

    def __init__(self, img_root_dir, accepted_extensions):
        self.img_root_dir = img_root_dir
        self.accepted_extensions = accepted_extensions
        self.df_images = None

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
        unique_labels = self.df_images['class'].unique()
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



    def _clean_scientific_name(self):
        # DwC scientificName: 'Genus + specificEpithet
        # Rename categoricals with 'genus species var. subspecies' to just 'genus species'. Also,
        #   rename 'genus species subsp. subspecie' to just 'genus species':
        self._remove_taxon_ranks_and_remap_categoricals()
        # Remove 'Genus' level scientific names, it is assumed there will be too much variation within one species:
        self._remove_species_level

        print('yay')
        print()


    def clean_images(self):
        self.df_images = self._get_raw_image_lists_df()
        self.df_images = self._clean_scientific_name()



def main(root_dir):
    img_executor = ImageExecutor(img_root_dir=root_dir, accepted_extensions=['jpg', 'jpeg'])
    img_executor.clean_images()


if __name__ == '__main__':
    # Logging verbosity:
    tf.logging.set_verbosity(tf.logging.INFO)

    # GoingDeeper Configuration:

    # BOON Configuration:
    main(root_dir='D:\\data\\BOON\\images')
