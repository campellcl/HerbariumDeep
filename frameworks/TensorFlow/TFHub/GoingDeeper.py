"""
GoingDeeper.py
Implementation of Pipelined TensorFlow Transfer Learning.
"""
import os
import sys
import argparse
import tensorflow as tf


def prepare_tensor_board_directories():
    """
    prepare_tensor_board_directories: Ensures that if a TensorBoard storage directory is defined in the command line
        flags, that said directory is purged of old TensorBoard files, and that this program has sufficient permissions
        to write new TensorBoard summaries to the specified path.
    :return:
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


def main(_):
    # Enable visible logging output:
    tf.logging.set_verbosity(tf.logging.INFO)
    # Delete any TensorBoard summaries left over from previous runs:
    prepare_tensor_board_directories()
    tf.logging.info(msg='Removed left over tensorboard summaries from previous runs.')
    # Ensure that the declared bottleneck file actually exists, or create it (if it does not):
    # ON RESUME: add a cmd line flag for overriding image_list generation if user is positive all samples have bottleneck vectors
    # Otherwise, it is computationally expensive to do a full search of the image directory to compare every classes
    # sample count to the uncompressed bottleneck dataframes length.


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
        default=CMD_ARG_FLAGS.eval_step_interval[0],
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
            '-tfhub_module',
            type=str,
            default='https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1',
            nargs='?',
            help="""\
                    Which TensorFlow Hub module to use.
                    See https://github.com/tensorflow/hub/blob/r0.1/docs/modules/image.md
                    for some publicly available ones.\
                    """
        )
        parser.add_argument(
            '--init_type',
            choices=('he', 'xavier', 'random'),
            nargs=1,
            required=True,
            help='Initialization technique {he, xavier, random}. URL resources to be added soon.'
        )
    else:
        print('Error: Support for resuming training of a previous model has not been implemented yet.')
        raise NotImplementedError
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
    parser.add_argument(
        '--num_epochs',
        dest='num_epochs',
        type=int,
        default=10000,
        nargs=1,
        required=True,
        help='The number of training epochs (the number of passes over the entire training dataset during training).'
    )
    _parse_train_tensorflow_related_hyperparameter_known_args(parser=parser)
    parser.add_argument(
        '--learning_rate_type',
        choices=('static', 'dynamic'),
        default='static',
        nargs=1,
        dest='learning_rate_type',
        help='Specify the type of learning rate as either static {static} (e.g. fixed) or dynamic {dynamic} (e.g. '
             'learning rate schedule, learning rate optimization technique).'
    )
    CMD_ARG_FLAGS, unknown = parser.parse_known_args()
    if CMD_ARG_FLAGS.learning_rate_type[0] == 'static':
        parser.add_argument(
            '--learning_rate',
            dest='learning_rate',
            default=0.01,
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
        help="""WARNING: This command line argument belongs to a special class of \'force\' flags used for override 
        functionality, and only intended for users who really know what they are doing.
        For this command in particular: An invocation of the script with the \'--force_bypass_bottleneck_updates\' flag
            enabled will result in 
        """
    )


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
    # CMD_ARG_FLAGS, unparsed = parser.parse_known_args()
    # global non-positional arguments go here (use single - prefix instead of double dash --).

    # Tests:
    # Valid:
    # CMD_ARG_FLAGS, unparsed = parser.parse_known_args(['--train', '--init', 'random', '--train_batch_size 0'])
    # CMD_ARG_FLAGS, unknown = parser.parse_known_args(['--train', '--image_dir', 'C:\\sample_images\\root_dir'])
    CMD_ARG_FLAGS, unparsed = parser.parse_known_args()
    print(CMD_ARG_FLAGS)
    '''
    Execute this script under a shell instead of importing as a module. Ensures that the main function is called with
    the proper command line arguments (builds on default argparse). For more information see:
    https://stackoverflow.com/questions/33703624/how-does-tf-app-run-work
    '''
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
