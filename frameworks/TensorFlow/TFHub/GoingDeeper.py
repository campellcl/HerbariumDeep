"""
GoingDeeper.py
Implementation of Pipelined TensorFlow Transfer Learning.
"""
import os
import sys
import argparse
import tensorflow as tf
import collections
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import time
from frameworks.TensorFlow.TFHub.TFHClassifier import TFHClassifier
import pandas as pd

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

def _prepare_tensor_board_directories():
    """
    _prepare_tensor_board_directories: Ensures that if a TensorBoard storage directory is defined in the command line
        flags, that said directory is purged of old TensorBoard files, and that this program has sufficient permissions
        to write new TensorBoard summaries to the specified path.
    :return None: see above ^
    """
    # Check to see if the file exists:
    if tf.gfile.Exists(CMD_ARG_FLAGS.summaries_dir):
        # Delete everything in the file recursively:
        tf.gfile.DeleteRecursively(CMD_ARG_FLAGS.summaries_dir)
    # Re-create (or create for the first time) the storage directory:
    tf.gfile.MakeDirs(CMD_ARG_FLAGS.summaries_dir)
    # Check to see if intermediate computational graphs are to be stored:
    if CMD_ARG_FLAGS.intermediate_store_frequency > 0:
        if not os.path.exists(CMD_ARG_FLAGS.intermediate_output_graphs_dir):
            os.makedirs(CMD_ARG_FLAGS.intermediate_output_graphs_dir)
    return


def _is_bottleneck_for_every_sample(image_lists, bottlenecks):
    train_image_paths = []
    val_image_paths = []
    test_image_paths = []
    for species, datasets in image_lists.items():
        # Training set images:
        species_train_image_paths = datasets['train']
        for species_train_image_path in species_train_image_paths:
            train_image_paths.append(species_train_image_path)
        # Validation set images:
        species_val_image_paths = datasets['val']
        for species_val_image_path in species_val_image_paths:
            val_image_paths.append(species_val_image_path)
        # Testing set images:
        species_test_image_paths = datasets['test']
        for species_test_image_path in species_test_image_paths:
            test_image_paths.append(species_test_image_path)
    # Ensure every training image has a bottleneck:
    for train_image_path in train_image_paths:
        if train_image_path not in bottlenecks['path'].values:
            return False
    # Ensure every validation image has a bottleneck tensor in bottlenecks:
    for val_image_path in val_image_paths:
        if val_image_path not in bottlenecks['path'].values:
            return False
    # Ensure every test image has a bottleneck tensor in the bottlenecks dataframe:
    for test_image_path in test_image_paths:
        if test_image_path not in bottlenecks['path'].values:
            return False
    return True


def _update_and_retrieve_bottlenecks(image_lists):
    """
    _update_and_retrieve_bottlenecks:
    :return:
    """
    bottleneck_path = CMD_ARG_FLAGS.bottleneck_path
    if os.path.isfile(os.path.basename(CMD_ARG_FLAGS.bottleneck_path)):
        # Bottlenecks file exists, read from disk:
        tf.logging.info(msg='Bottleneck file successfully located at the provided path: \'%s\'.'
                            % CMD_ARG_FLAGS.bottleneck_path)
        bottlenecks = pd.read_pickle(os.path.basename(CMD_ARG_FLAGS.bottleneck_path))
        tf.logging.info(msg='Bottleneck file \'%s\' successfully restored from disk.'
                            % os.path.basename(CMD_ARG_FLAGS.bottleneck_path))
        if _is_bottleneck_for_every_sample(image_lists, bottlenecks):
            # Partition the bottleneck dataframe:
            ON RESUME: Why even have image_list partitions? It seems the only point is for bottleneck creation.
            once the bottlenecks are created then they are loaded in and partitioned regardless. So no partition of
            training images, just load them in and run the existance check, then get the bottlenecks and partition those.
            Reference back to the image to get the label is in the bottlenecks dataframe already.
        # TODO: use the command line 'advanced' flags to override this check:



    pass


def _get_image_lists(image_dir):
    """
    _get_image_lists: Creates a dictionary of file paths to images on the hard drive indexed by class label.
    :param image_dir: <str> The file path pointing to the parent directory housing a series of sample images partitioned
        into their respective subdirectories by class label.
    :return image_lists: <collections.OrderedDict> A dictionary indexed by class label, which provides as its value a
        list of file paths for images belonging to the chosen key/species/class-label.
    """
    '''
    Check to see if the root directory exists. We use tf.gfile which is a C++ FileSystem API wrapper for the Python
        file API that also supports Google Cloud Storage and HDFS. For more information see:
        https://stackoverflow.com/questions/42256938/what-does-tf-gfile-do-in-tensorflow
    '''
    if not tf.gfile.Exists(image_dir):
        tf.logging.error("Root image directory '" + image_dir + "' not found.")
        return None

    accepted_extensions = ['jpg', 'jpeg']   # Note: Includes JPG and JPEG b/c the glob is case insensitive
    image_lists = collections.OrderedDict()

    sub_dirs = sorted(x[0] for x in tf.gfile.Walk(image_dir))
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if is_root_dir:
            is_root_dir = False
            # Skip the root_dir:
            continue
        if dir_name == image_dir:
            # Return control to beginning of for-loop:
            continue
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in accepted_extensions:
            # Get a list of all accepted file extensions and the targeted file_name:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            # Append all items from the file_glob to the list of files (if extension exists):
            file_list.extend(tf.gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning(msg='No files found in \'%s\'. Class label omitted from data sets.' % dir_name)
            # Return control to beginning of for-loop:
            continue
        if len(file_list) < 20:
            tf.logging.warning('WARNING: Folder has less than 20 images, which may cause issues. See: %s for info.'
                               % 'https://stackoverflow.com/questions/38175673/critical-tensorflowcategory-has-no-images-validation')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            tf.logging.warning(
                'WARNING: Folder {} has more than {} images. Some images will '
                'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        # label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        label_name = dir_name.lower()
        if label_name not in image_lists:
            image_lists[label_name] = file_list
    return image_lists


def _partition_image_lists(image_lists, train_percent, val_percent, test_percent, random_state):
    """
    _partition_image_lists: Partitions the provided dict of class labels and file paths into training, validation, and
        testing datasets.
    :param image_lists: <collections.OrderedDict> A dictionary indexed by class label, which provides as its value a
        list of file paths for images belonging to the chosen key/species/class-label.
    :param train_percent: What percentage of the training data is to remain in the training set.
    :param val_percent: What percentage of the remaining training data (after removing test set) is to be allocated
        for a validation set.
    :param test_percent: What percentage of the training data is to be allocated to a testing set.
    :param random_state: A seed for the random number generator controlling the stratified partitioning.
    :return partitioned_image_lists: <collections.OrderedDict> A dictionary indexed by class label which returns another
        dictionary indexed by the dataset type {'train','val','test'} which in turn, returns the list of image file
        paths that correspond to the chosen class label and that reside in the chosen dataset.
    """
    partitioned_image_lists = collections.OrderedDict()
    for class_label, image_paths in image_lists.items():
        class_label_train_images, class_label_test_images = model_selection.train_test_split(
            image_paths, train_size=train_percent,
            test_size=test_percent, shuffle=True,
            random_state=random_state
        )
        class_label_train_images, class_label_val_images = model_selection.train_test_split(
            class_label_train_images, train_size=train_percent,
            test_size=val_percent, shuffle=True,
            random_state=random_state
        )
        partitioned_image_lists[class_label] = {
            'train': class_label_train_images,
            'val': class_label_val_images,
            'test': class_label_test_images
        }
    return partitioned_image_lists


def main(_):
    # Enable visible logging output:
    tf.logging.set_verbosity(tf.logging.INFO)
    # Delete any TensorBoard summaries left over from previous runs:
    _prepare_tensor_board_directories()
    tf.logging.info(msg='Removed left over tensorboard summaries from previous runs.')
    # TODO: could add an advanced cmd-line flag here to bypass image directory walk if user knows bottlenecks are up-to-date.
    ''' Recursively walk the image directory and build up a dict of all images and their associated class names and 
        file paths. 
    '''
    tf.logging.info(msg='Recursively walking the provided image directory and aggregating image paths by class '
                        'label...')
    ts = time.time()
    image_lists = _get_image_lists(image_dir=CMD_ARG_FLAGS.image_dir)
    tf.logging.info(msg='Recursive image directory walk performed in: %s seconds (%.2f minutes).'
                        % ((time.time() - ts), (time.time() - ts)/60))
    # TODO: Do we really need to partition images again with the bottlenecks already generated?
    # Partition images into training, validation, and testing sets:
    # TODO: add support for user specified test set proportions? Right now constrained to numbers given in paper.
    train_percent = .80
    val_percent = .20
    test_percent = .20
    tf.logging.info(msg='Partitioning each classes\' list of images into training (n=%.2f%%), validation (n=%.2f%%), '
                        'and testing (n=%.2f%%) datasets.'
                        % (train_percent * 100, val_percent * 100, test_percent * 100))
    ts = time.time()
    image_lists = _partition_image_lists(
        image_lists=image_lists, train_percent=0.80,
        val_percent=.20, test_percent=.20,
        random_state=0
    )
    tf.logging.info(msg='Performed this partitioning of sample images in: %s seconds (%.2f minutes).'
                        % ((time.time() - ts), (time.time() - ts)/60))
    num_classes = len(list(image_lists.keys()))
    num_images = 0
    num_train_images = 0
    num_test_images = 0
    num_val_images = 0
    for class_label, datasets in image_lists.items():
        if 'train' in datasets:
            num_train_images += len(datasets['train'])
            num_images += len(datasets['train'])
        if 'val' in datasets:
            num_val_images += len(datasets['val'])
            num_images += len(datasets['val'])
        if 'test' in datasets:
            num_test_images += len(datasets['test'])
            num_images += len(datasets['test'])
    tf.logging.info(msg='Found %d total images. Found %d unique classes. Partitioned into %d total training images, '
                        '%d total validation images, and %d total testing images.'
                        % (num_images, num_classes, num_train_images, num_val_images, num_test_images))
    tf.logging.info(msg='This partitioning was performed on a class-by-class basis. The sampled distribution of '
                        'class labels is:\n\ttraining (%d/%d) = %.2f%% of all sample images'
                        '\n\tvalidation (%d/%d) = %.2f%% of all sample images\n\ttesting (%d/%d) = %.2f%% of '
                        'all sample images' % (num_train_images, num_images, ((num_train_images*100)/num_images),
                                               num_val_images, num_images, ((num_val_images*100)/num_images),
                                               num_test_images, num_images, ((num_test_images*100)/num_images)))

    tfh_classifier = TFHClassifier(
        tfhub_module_url=CMD_ARG_FLAGS.tfhub_module,
        init_type=CMD_ARG_FLAGS.init_type,
        learning_rate_type=CMD_ARG_FLAGS.learning_rate_type,
        learning_rate=CMD_ARG_FLAGS.learning_rate,
        num_unique_classes=num_classes)


    bottlenecks = _update_and_retrieve_bottlenecks(image_lists)

    print()
    pass

    # tfh_classifier.fit(X=image_lists['train'], y=image_lists[])



def _parse_known_evaluation_args(parser):
    '''
    Evaluation specific command line arguments go here:
    '''
    print('Error: Support for evaluation mode only has only been partially implemented.')
    raise NotImplementedError


def _parse_train_dynamic_learn_rate_optimizer_known_args(parser):
    """
    _parse_known_train_dynamic_learn_rate_optimizer: This helper method is invoked if the user provides the following
        sequence of flags during invocation:
            --learn_rate_type dynamic --optimizer_type adaptive_optimizer
        This helper method ensures that the arguments necessary for user-specified optimization module are all provided
        during invocation and prior to instantiation.
    :param parser: <argparse.ArgumentParser> A reference to the instance of the global/parent argument parser.
    :return None: Upon completion, the provided argparse object will be updated with the necessary set of required
        command line flags, and argparse will have ensured that the required flags were provided during invocation.
    """
    parser.add_argument(
        '--learning_rate_optimizer',
        dest='lr_optimizer',
        choices=('tf.train.MomentumOptimizer', 'tf.train.AdagradOptimizer', 'tf.train.AdamOptimizer'),
        nargs=1,
        required=True,
        help='You have chosen a dynamic learning rate controlled by an optimization algorithm. Specify the '
             'module housing the optimization algorithm of your choice: '
             '{tf.train.MomentumOptimizer,tf.train.AdagradOptimizer,tf.train.AdamOptimizer}. '
    )
    CMD_ARG_FLAGS, unknown = parser.parse_known_args()
    if CMD_ARG_FLAGS.lr_optimizer[0] == 'tf.train.MomentumOptimizer':
        parser.add_argument(
            '--momentum',
            dest='momentum',
            nargs=1,
            type=float,
            required=True,
            metavar='[0-1]',
            help='Threshold (0-1) indicating '
        )
    elif CMD_ARG_FLAGS.lr_optimizer[0] == 'tf.train.AdagradOptimizer':
        print('Error: The dynamic learning rate optimizer \'tf.train.AdagradOptimizer\' is not yet supported.')
        raise NotImplementedError
    elif CMD_ARG_FLAGS.lr_optimizer[0] == 'tf.train.AdamOptimizer':
        print('Error: The dynamic learning rate optimizer \'tf.train.AdamOptimizer\' is not yet supported.')
        raise NotImplementedError


def _parse_train_tensorflow_related_hyperparameter_known_args(parser):
    """
    _parse_train_tensorflow_related_heyperparameter_known_args: This helper method is executed if the user provides the
        following sequence of flags during script invocation:
            --use_case train
    :param parser: parser: <argparse.ArgumentParser> A reference to the instance of the global/parent argument parser.
    :return: None: Upon completion, the provided argparse object will be updated with the necessary set of required
        command line flags, and argparse will have ensured that the required flags were provided during invocation.
    """
    CMD_ARG_FLAGS, unknown = parser.parse_known_args()
    parser.add_argument(
        '--eval_step_interval',
        dest='eval_step_interval',
        type=int,
        default=CMD_ARG_FLAGS.num_epochs[0] // 10,
        nargs='?',
        help='Specifies how many epochs should pass during training before performing an evaluation step (i.e. computing'
             ' the chosen training metrics on the validation batch).'
    )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='tmp/summaries',
        required=True,
        nargs=1,
        help='The directory to which TensorBoard summaries will be saved.'
    )
    CMD_ARG_FLAGS, unknown = parser.parse_known_args()
    parser.add_argument(
        '--intermediate_store_frequency',
        type=int,
        default=CMD_ARG_FLAGS.eval_step_interval,
        nargs='?',
        help="""Specifies how many epochs should be run (during training) before storing an intermediate checkpoint of 
                the computational graph. If 0 is provided, then no checkpoints of the computational graph will be stored 
                during the training process; but one will be stored once the training process is complete.
                NOTE: The disk write required to store the computational graph is somewhat computationally expensive in
                    comparison to the training process (when using pre-computed bottleneck values). As a result, the 
                    saving (and automatic restore from) the computational graph should be invoked somewhat sparingly. 
                    However, in the event of a critical failure; the training process will only ever to be capable of 
                    resuming from the last-saved checkpoint of the model. If frequent interruptions to the
                    training process is anticipated, then it is recommended to set this frequency at least as high as 
                    the default value (e.g. trigger the saving of a checkpoint every time the evaluation step is run).  
                """
    )
    parser.add_argument(
        '--output_graph',
        type=str,
        default='tmp/saved_model.pb',
        nargs='?',
        help="""Indicates the directory in which to save the computational graph of the final trained model. 
            NOTE: It is not the final checkpoint that is saved (of the form checkpoint, *.index, *.meta) but rather; the
            final trained model in it's entirety (architecture included) as a *.pb file. It is this file that will be
            used, if this program is later invoked with the flag:
                --use_case eval 
        """
    )
    parser.add_argument(
        '--intermediate_output_graphs_dir',
        type=str,
        default='tmp/intermediate_graphs/',
        nargs='?',
        help="""Indicates the directory used to save intermediate copies of the computational graph. The anticipated use
            case is that a model may not have been finished training, but still may have converged. In this case, it may 
            be convenient to compare and contrast this model (viewed externally as a fully trained model) with other 
            already trained models, or other equally partially-trained models at the same epoch.
            NOTE: This does NOT specify the directory to house checkpoints generated during training. Instead, this flag
                specifies the directory to house architectural copies of the entire model (as a *.pb file) during 
                training. It is this file that will be used, if the program is later invoked with the --use_case flag 
                set to 'eval' mode, and the previous model never fully completed training (therebye failing to generate 
                the trained \'saved_model.pb\' file which is usually preferred in the evaluation invocation context).  
        """
    )


def _parse_known_training_args(parser):
    """
    _parse_known_training_args: This helper method is executed if the user provides the following sequence of flags
        during script invocation:
            --use_case train
        This helper method ensures that the arguments necessary for all forms of training are provided during invocation
        and prior to instantiation.
    :param parser: <argparse.ArgumentParser> A reference to the instance of the global/parent argument parser.
    :return None: Upon completion, the provided argparse object will be updated with the necessary set of required
        command line flags, and argparse will have ensured that the required flags were provided during invocation.
    """
    '''
    Training mode specific command line arguments go here:
    '''
    parser.add_argument(
        '--training_state',
        choices=('new_model', 'resume_training'),
        dest='train_state',
        nargs=1,
        required=True,
        help='Whether or not the model\'s weights should be re-initialized {new_model} or restored '
             '{resume_training} from a previous session wherein the model was not able to finish training.'
    )
    CMD_ARG_FLAGS, unknown = parser.parse_known_args()
    if CMD_ARG_FLAGS.train_state[0] == 'new_model':
        # _parse_known_train_new_model_args(parser=parser)
        parser.add_argument(
            '--tfhub_module',
            type=str,
            default='https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1',
            nargs=1,
            required=True,
            help="""\
                    Which TensorFlow Hub module to use.
                    See https://github.com/tensorflow/hub/blob/r0.1/docs/modules/image.md
                    for some publicly available ones.\
                    """
        )
        parser.add_argument(
            '--init_type',
            choices=('he', 'xavier', 'random'),
            type=str,
            nargs=1,
            required=True,
            help='Initialization technique {he, xavier, random}. URL resources to be added soon.'
        )
    else:
        print('Error: Support for resuming training of a previous model has not been implemented yet.')
        raise NotImplementedError
    parser.add_argument(
        '--num_epochs',
        dest='num_epochs',
        type=int,
        nargs=1,
        required=True,
        help='The number of training epochs (the number of passes over the entire training dataset during training).'
    )
    CMD_ARG_FLAGS, unknown = parser.parse_known_args()
    parser.add_argument(
        '-learning_rate_type',
        choices=('static', 'dynamic'),
        type=str,
        nargs=1,
        required=True,
        dest='learning_rate_type',
        help='Specify the type of learning rate as either static {static} (e.g. fixed) or dynamic {dynamic} (e.g. '
             'learning rate schedule, learning rate optimization technique).'
    )
    CMD_ARG_FLAGS, unknown = parser.parse_known_args()
    if CMD_ARG_FLAGS.learning_rate_type[0] == 'static':
        parser.add_argument(
            '--learning_rate',
            dest='learning_rate',
            # default=0.01,
            nargs=1,
            type=float,
            required=True,
            help='Specify the learning rate (eta). The default value is eta=0.01. This will initialize a '
                 'tf.train.GradientDescentOptimizer designed to use a constant learning rate.'
        )
    else:
        parser.add_argument(
            '--optimizer_type',
            choices=['fixed_lr_schedule', 'adaptive_optimizer'],
            dest='optimizer_type',
            nargs=1,
            required=True,
            help='Specify the type of dynamic learning rate as either a user-provided Learning Rate Schedule ' \
                 '{learning_rate_schedule} or a specific op algorithm {optimization_algorithm} such as:'
                 ' Nestrov Accelerated Gradient, ...., etc..'
        )
        CMD_ARG_FLAGS, unknown = parser.parse_known_args()
        if CMD_ARG_FLAGS.optimizer_type[0] == 'learning_rate_schedule':
            # _parse_known_train_optimizer_learn_schedule(parser=parser)
            print('ERROR: Support for custom learning rate schedule not implemented yet. See: '
                  'https://stackoverflow.com/questions/33919948/how-to-set-adaptive-learning-rate-for-gradientdescentoptimizer')
            raise NotImplementedError
        else:
            _parse_train_dynamic_learn_rate_optimizer_known_args(parser=parser)
    parser.add_argument(
        '-train_batch_size',
        dest='train_batch_size',
        type=int,
        default=-1,
        nargs='?',
        help='The number of images per mini-batch during training. If -1 (default) is provided the entire training '
             'dataset will be used.'
    )
    parser.add_argument(
        '-val_batch_size',
        dest='val_batch_size',
        type=int,
        default=-1,
        nargs='?',
        help='The number of images to use during an evaluation step on the validation dataset while training. By default'
             ' this value is -1, the size of the entire validation datset.'
    )
    _parse_train_tensorflow_related_hyperparameter_known_args(parser=parser)


def _parse_known_advanced_user_args(parser):
    """
    _parse_known_advanced_user_args: This helper method is executed in the global scope during command line invocation.
        This helper method parses out advanced command line arguments and warns the user the implications of an improper
        invocation using an argument of this type.
    :param parser: <argparse.ArgumentParser> A reference to the instance of the global/parent argument parser.
    :return None: Upon completion, the provided argparse object will be updated with the necessary set of required
        command line flags, and argparse will have ensured that the optional flags were provided properly during
        invocation.
    """

    parser.add_argument(
        '--force_bypass_bottleneck_updates',
        dest='force_bypass_bottleneck_updates',
        default=False,
        action='store_true',
        help="""WARNING: This command line argument belongs to a special class of \'force\' flags used for override 
        functionality, and only intended for users who really know what they are doing.
        For this command in particular: An invocation of the script with the \'--force_bypass_bottleneck_updates\' flag
            enabled will result in a bypass of the entire image directory walk while confirming the existence of
            generated bottlenecks for all samples. That is to say, using this argument tells the program that there were
            NO new samples added since the last time the bottlenecks have been generated. This allows for a faster
            initialization step, since the entire parent image directory no longer needs to be fully traversed. 
        """
    )


def _force_type_coercion_of_argparse_cmd_arg_flags():
    """
    _force_type_coercion_of_argparse_cmd_arg_flags: Ensures that the argparse read arguments are the types that
        argparse was instructed to ensure. Variable length input arguments are stored in arrays regardless if the
        input type was specified as a scalar. This method converts all arrays with one element into the respective
        element, modifying the type in the globally scoped CMD_ARG_FLAGS. Care must be taken to run this method after
        all parsing has been completed, so that the cmd argument types match what the rest of the program is expecting.
    :return None: see above^
    """
    if CMD_ARG_FLAGS.summaries_dir:
        CMD_ARG_FLAGS.summaries_dir = CMD_ARG_FLAGS.summaries_dir[0]
    if CMD_ARG_FLAGS.image_dir:
        CMD_ARG_FLAGS.image_dir = CMD_ARG_FLAGS.image_dir[0]
    if CMD_ARG_FLAGS.bottleneck_path:
        CMD_ARG_FLAGS.bottleneck_path = CMD_ARG_FLAGS.bottleneck_path[0]
    if CMD_ARG_FLAGS.init_type:
        CMD_ARG_FLAGS.init_type = CMD_ARG_FLAGS.init_type[0]
    if CMD_ARG_FLAGS.learning_rate_type:
        CMD_ARG_FLAGS.learning_rate_type = CMD_ARG_FLAGS.learning_rate_type[0]
    if CMD_ARG_FLAGS.learning_rate:
        CMD_ARG_FLAGS.learning_rate = CMD_ARG_FLAGS.learning_rate[0]
    if CMD_ARG_FLAGS.num_epochs:
        if type(CMD_ARG_FLAGS.num_epochs) is list:
            CMD_ARG_FLAGS.num_epochs = CMD_ARG_FLAGS.num_epochs[0]
    if CMD_ARG_FLAGS.tfhub_module:
        CMD_ARG_FLAGS.tfhub_module = CMD_ARG_FLAGS.tfhub_module[0]
    print('Application Executed with the following parsed command line arguments:\n\t%s' % CMD_ARG_FLAGS)


if __name__ == '__main__':
    """
    __main__: Global scope initialization. Performs argument parsing prior to invoking the main method. If this script
        is functioning properly there is NO chance that a method called in main that relies on user input will fail to 
        have access to the required user-provided arguments during initialization. This method is intended to ensure 
        that all required TensorFlow constructor's parameters have associated user-provided values during invocation, 
        and prior to instantiation.
    """
    # Create top-level parser:
    parser = argparse.ArgumentParser(description='TensorFlow Transfer Learning on Going Deeper Datasets')
    '''
    Initialization-Level global command line arguments go here:
    '''
    parser.add_argument(
        '--use_case',
        choices=('train', 'eval'),
        dest='use_case',
        nargs=1,
        required=True,
        help='Indicates the intended use case of the program by the user. This program can be run in either training '
             '{train} or evaluation {eval} mode.'
    )
    CMD_ARG_FLAGS, unknown = parser.parse_known_args()
    if CMD_ARG_FLAGS.use_case == 'eval':
        # Parse command line arguments only relevant to the use case of evaluating existing and already trained models:
        _parse_known_evaluation_args(parser=parser)
    else:
        # Parse command line arguments only relevant to the use case of training a model:
        _parse_known_training_args(parser=parser)
    '''
    Post-Initialization-Level global command line arguments go here:
        For global non-positional arguments use a single - prefix instead of a double -- prefix. 
    '''
    parser.add_argument(
        '--image_dir',
        dest='image_dir',
        type=str,
        required=True,
        nargs=1,
        help='Path to folders of labeled images. If this script is being invoked in training mode this directory will '
             'later be partitioned into training, validation, and testing datasets. However, if this script is being '
             'invoked in evaluation mode, this entire directory will be inferred solely as a testing dataset.'
    )
    parser.add_argument(
        '--bottleneck_path',
        dest='bottleneck_path',
        type=str,
        required=True,
        nargs=1,
        help="""The path to the dataframe storing the cached bottleneck layer values. It is standard to name this file:
        \'bottlenecks.pkl\'. Note that bottleneck generation via forward propagation is computationally expensive. It 
        is therefore recommended that great care should be taken to prevent altering this path repeatedly, unless it is
        absolutely necessary (as in the case of a new architecture or newly applied data augmentation technique). 
            * If this script is being invoked in training mode, and a new model is being trained, the bottlenecks file
                may not yet exist. In this case, the provided argument will be interpreted as the location in which the
                bottlenecks file is to reside upon it\'s eventual creation. 
            * If data augmentation is being performed, then this method will need to be adapted to support a parent 
                directory that houses multiple bottleneck files (one for each form of data augmentation and network
                 architecture that is applied). 
        """
    )
    # Parse 'advanced' usage flags enabling faster execution at the cost of knowledge complexity:
    _parse_known_advanced_user_args(parser=parser)
    # CMD_ARG_FLAGS, unparsed = parser.parse_known_args()
    # global non-positional arguments go here (use single - prefix instead of double dash --).

    # Tests:
    # Valid:
    # CMD_ARG_FLAGS, unparsed = parser.parse_known_args(['--train', '--init', 'random', '--train_batch_size 0'])
    # CMD_ARG_FLAGS, unknown = parser.parse_known_args(['--train', '--image_dir', 'C:\\sample_images\\root_dir'])
    CMD_ARG_FLAGS, unparsed = parser.parse_known_args()
    ''' Force conversions to the correct type (argparse supports chained arguments so need to convert singleton 
        lists to actual singletons, etc...). 
        NOTE: This method MUST be executed after the very last call to parser.parse_known_args(), or otherwise great
              care must be taken to prevent an override of the global command argument flags array while parsing. After
              this method enforces a type coercion of the argparse flags, this will not be performed again without
              a repeated manual invocation. 
    '''
    _force_type_coercion_of_argparse_cmd_arg_flags()
    '''
    Execute this script under a shell instead of importing as a module. Ensures that the main function is called with
    the proper command line arguments (builds on default argparse). For more information see:
    https://stackoverflow.com/questions/33703624/how-does-tf-app-run-work
    '''
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
