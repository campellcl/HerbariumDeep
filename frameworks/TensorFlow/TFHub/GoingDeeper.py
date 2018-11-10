"""
GoingDeeper.py
Implementation of Pipelined TensorFlow Transfer Learning.
"""
import os
import sys
import argparse
import tensorflow as tf


def main(_):
    # Ensure required parameters have been provided.
    if not CMD_ARG_FLAGS.image_dir:
        tf.logging.error(msg='...')


if __name__ == '__main__':
    # Create top-level parser:
    parser = argparse.ArgumentParser(description='TensorFlow Transfer Learning on Going Deeper Datasets')
    # Create namespace to hold partially parsed statements for action tree:
    CMD_ARG_FLAGS = argparse.Namespace()
    # Use case training or testing?
    # parser.add_argument(
    #     '--eval',
    #     dest='global_use_case_is_evaluation',
    #     action='store_true',
    #     help='Indicates that evaluation is to be performed (as opposed to training).'
    # )
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
        # Evaluation specific command line arguments go here.
        print('Support for evaluation only has not been implemented yet.')
        raise NotImplementedError
    else:
        ''' Training mode specific command line arguments go here: '''
        parser.add_argument(
            '--init_type',
            choices=('he', 'xavier', 'random'),
            default='random',
            nargs=1,
            help='Initialization technique {he, xavier, random}. URL resources to be added soon.'
        )
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
            '--learning_rate_type',
            choices=('static', 'dynamic'),
            default='static',
            nargs=1,
            dest='learning_rate_type',
            help='Specify the type of learning rate as either static {static} (e.g. fixed) or dynamic {dynamic} (e.g. '
                 'learning rate schedule, learning rate optimization technique).'
        )
        CMD_ARG_FLAGS, unknown = parser.parse_known_args()
        # CMD_ARG_FLAGS, unparsed = parser.parse_known_args()
        if CMD_ARG_FLAGS.learning_rate_type[0] == 'static':
            parser.add_argument(
                '-learning_rate',
                dest='learning_rate',
                default=0.01,
                nargs=1,
                type=float,
                required=True,
                help='Specify the learning rate (eta). The default value is eta=0.01.'
            )
        else:
            parser.add_argument(
                '--dynamic_learning_rate_type',
                choices=['learning_rate_schedule', 'optimization_algorithm'],
                dest='dynamic_learning_rate_type',
                nargs=1,
                required=True,
                help='Specify the type of dynamic learning rate as either a user-provided Learning Rate Schedule ' \
                     '{learning_rate_schedule} or a specific optimization algorithm {optimization_algorithm} such as:'
                     ' Nestrov Accelerated Gradient, ...., etc..'
            )
            CMD_ARG_FLAGS, unknown = parser.parse_known_args()
            if CMD_ARG_FLAGS.dynamic_learning_rate_type[0] == 'learning_rate_schedule':
                print('ERROR: Learning rate scheduling not implemented yet.')
                raise NotImplementedError
            else:
                parser.add_argument(
                    '-learning_rate_optimizer',
                    dest='lr_optimizer',
                    choices=('tf.optim.MomentumOptimizer', 'tf.optim.ClassName'),
                    nargs=1,
                    required=True,
                    help='You have chosen a dynamic learning rate controlled by an optimization algorithm. Specify the '
                         'module housing the optimization algorithm of your choice: '
                         '{tf.optim.MomentumOptimizer,tf.optim.ClassName}. '
                )

    # CMD_ARG_FLAGS, unparsed = parser.parse_known_args()
    # global non-positional arguments go here (use single - prefix instead of double dash --).





# if __name__ == '__main__':
#     # Create top-level parser:
#     parser = argparse.ArgumentParser(description='TensorFlow Transfer Learning on Going Deeper Datasets')
#     # Either training or evaluating:
#     global_use_case = None
#     mutual_exclusion_train_eval_group = parser.add_mutually_exclusive_group(required=True)
#     # Create the parser for the 'train' command
#     train_parser = mutual_exclusion_train_eval_group.add_argument(
#         '--train',
#         type=str,
#         default='',
#         nargs='?',
#         help='Indicates that training is to be performed (as opposed to evaluation).'
#     )
#     test_parser = mutual_exclusion_train_eval_group.add_argument(
#         '--test',
#         type=str,
#         default='',
#         help='Indicates that testing is to be performed (as opposed to training).'
#     )

# if __name__ == '__main__':
#     # Create top-level parser:
#     parser = argparse.ArgumentParser(description='TensorFlow Transfer Learning on Going Deeper Datasets')
#     # Either training or evaluating:
#     global_use_case = None
#     parser.add_argument(
#         dest='global_use_case',
#         choices=['--train', '--eval'],
#         # action='store_const',
#         # const='--train',
#         nargs='?',
#         help='The intended use case of the program, either training {--train} or evaluating {--eval}.'
#     )
#     parser.add_argument(
#         '--image_dir',
#         dest='image_dir',
#         help='The path to the directory housing sample images partitioned into subdirectories named by class label.',
#     )
#
#
#     parser.add_argument(
#         '--log_dir',
#         type=str,
#         default=os.path.join(os.getenv('TEST_TMPDIR', 'tmp'),
#                              'tensorflow/mnist/logs/mnist_with_summaries'),
#         help='Summaries log directory')


    # # Hyper Parameters
    # model_names = sorted(name for name in models.__dict__
    #                      if name.islower() and not name.startswith("__")
    #                      and callable(models.__dict__[name]))
    #
    # parser = argparse.ArgumentParser(description='PyTorch Transfer Learning Demo on Going Deeper Herbaria 1K Dataset')
#     # parser.add_argument('STORE', metavar='DIR', help='Data storage directory.')
#     # parser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true',
#     #                     help='Enable verbose print statements (yes, no)?')
#     # parser.add_argument('--arch', '-a', metavar='ARCH', default='inception_v3', choices=model_names,
#     #                     help='model architecture: ' + ' | '.join(model_names) + ' (default: inception_v3)')
#     # parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
#     #                     metavar='LR', help='initial learning rate')
#     # # parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#     # #                     help='momentum')
#     # # parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
#     # #                     metavar='W', help='weight decay (default: 1e-4)')
#     # parser.add_argument('--resume', default='', type=str, metavar='PATH',
#     #                     help='path to latest checkpoint (default: none)')
#
#     # CMD_ARG_FLAGS, unknown = parser.parse_known_args()
#     # if CMD_ARG_FLAGS.global_use_case == 'train':
#     #     parser.add_argument(
#     #         dest='--image_dir',
#     #         metavar=str,
#     #         dest='image_dir'
#     #     )
#
#     # mutual_exclusion_train_eval_group = parser.add_mutually_exclusive_group(required=True)
#     # # Create the parser for the 'train' command
#     # train_parser = mutual_exclusion_train_eval_group.add_argument(
#     #     '--train',
#     #     type=str,
#     #     default='',
#     #     help='Indicates that training is to be performed (as opposed to evaluation).'
#     # )
#     # test_parser = mutual_exclusion_train_eval_group.add_argument(
#     #     '--test',
#     #     type=str,
#     #     default='',
#     #     help='Indicates that testing is to be performed (as opposed to training).'
#     # )
#
    # Tests:
    # Valid:
    # CMD_ARG_FLAGS, unparsed = parser.parse_known_args(['--train', '--init', 'random', '--train_batch_size 0'])
    # CMD_ARG_FLAGS, unknown = parser.parse_known_args(['--train', '--image_dir', 'C:\\sample_images\\root_dir'])
    CMD_ARG_FLAGS, unparsed = parser.parse_known_args()
    print(CMD_ARG_FLAGS)
    # CMD_ARG_FLAGS, unknown = parser.parse_known_args(['--eval'])
    # CMD_ARG_FLAGS, unknown = parser.parse_known_args(['--train', '--eval'])

    # Invalid:
    # CMD_ARG_FLAGS, unknown = parser.parse_known_args([])    # TODO: Add case for do nothing (executed as valid).
    # CMD_ARG_FLAGS, unknown = parser.parse_known_args(['--train --eval'])
    # CMD_ARG_FLAGS, unknown = parser.parse_known_args(['--train', '--eval'])     # TODO: Add case for both mutually exclusive group ignore.
#
#
#
#
#     # # create the top-level parser
#     # parser = argparse.ArgumentParser(prog='PROG')
#     # parser.add_argument('--foo', action='store_true', help='foo help')
#     # subparsers = parser.add_subparsers(help='sub-command help', dest='subparser_name')
#     #
#     # # create the parser for the "command_a" command
#     # parser_a = subparsers.add_parser('command_a', help='command_a help')
#     # parser_a.add_argument('bar', type=int, help='bar help')
#     #
#     # # create the parser for the "command_b" command
#     # parser_b = subparsers.add_parser('command_b', help='command_b help')
#     # parser_b.add_argument('--baz', choices='XYZ', help='baz help')
#     #
#     # # parse some argument lists
#     # argv = ['--foo', 'command_a', '12', 'command_b', '--baz', 'Z']
#     # while argv:
#     #     print(argv)
#     #     options, argv = parser.parse_known_args(argv)
#     #     print(options)
#     #     if not options.subparser_name:
#     #         break
#
#     # # Repeatedly call parse on chained commands:
#     # ### this function takes the 'extra' attribute from global namespace
#     # parser = argparse.ArgumentParser()
#     # # Add all subparsers:
#     # subparsers = parser.add_subparsers(help='sub-command help', dest='subparser_name')
#     # # Add setup options for each subparser:
#     # train_use_case_parser = subparsers.add_parser('--scratch', help='Train from scratch')
#     # namespace = parser.parse_known_args()
#     # _parse_remaining_chained_commands(parser, namespace)
