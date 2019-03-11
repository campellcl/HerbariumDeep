# import tensorflow as tf
#
# tb_log_dir = 'C:\\Users\\ccamp\Documents\\GitHub\\HerbariumDeep\\frameworks\\TensorFlow\\TFHub\\tmp\\summaries\\train\\INIT_HE_NORMAL,OPTIM_ADAM,ACTIVATION_LEAKY_RELU,TRAIN_BATCH_SIZE__20\\events.out.tfevents.1551910450.DESKTOP-CHRIS'
# for i, e in enumerate(tf.train.summary_iterator(tb_log_dir)):
#     if e.summary.value:
#         print('step: %d:' % i)
#         # print(e.summary.value)
#         for v in e.summary.value:
#             if v.tag == 'loss_1':
#                 print('\tloss_1: %s' % v.simple_value)
#             elif v.tag == 'accuracy_1':
#                 print('\taccuracy_1: %s' % v.simple_value)
#             elif v.tag == 'top5_accuracy':
#                 print('\ttop5_accuracy: %s' % v.simple_value)
#             else:
#                 pass
#     print(e)

"""
TensorBoardDataExporter.py
source: https://stackoverflow.com/a/52095336/3429090
author: Spen: https://stackoverflow.com/users/2230045/spen
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class TensorBoardDataExporter:
    root_summaries_dir = None
    target_dirs = None

    def __init__(self, root_summaries_dir):
        self.root_summaries_dir = root_summaries_dir
        excluded_parent_dirs = [
            os.path.join(self.root_summaries_dir, 'gs'),
            os.path.join(self.root_summaries_dir, 'train'),
            os.path.join(self.root_summaries_dir, 'val'),
            os.path.join(self.root_summaries_dir, 'gs\\train'),
            os.path.join(self.root_summaries_dir, 'gs\\val')
        ]
        subdirs = sorted(x[0] for x in tf.gfile.Walk(self.root_summaries_dir))
        subdirs = subdirs[1:]
        # Remove parent directories with no event files for logging:
        self.target_dirs = [subdir for subdir in subdirs if subdir not in excluded_parent_dirs]


    @staticmethod
    def _tabulate_tensorboard_events(event_files):
        """
        tabulate_tensorboard_events: Receives a series of TensorBoard Event files representing a single run of the model,
            extracts relevant metrics from the Event files and aggregates the results into a dictionary.
        :param event_files: <list> A list of file pointers to the event files pertaining to a SINGLE model run.
        :returns event_steps, accumulated_output:
            :return event_steps: The series of discrete steps taken during the event, for which scalar data was accumulated.
            :return accumulated_output: A dictionary holding the tags to extract (specified in this method) and their
                corresponding tf.Summary values at the corresponding event_step.
        """
        # Accumulate all TensorBoard events into a single list of TensorBoard.Summary iterators:
        summary_iterators = [EventAccumulator(event_path).Reload() for event_path in event_files]

        # Specify the tags to be extracted from these events:
        tags_to_extract = summary_iterators[0].Tags()['scalars']

        # Setup a default dictionary so we can directly append without membership testing:
        accumulated_output = defaultdict(list)
        event_steps = []

        # Iterate through the tags to extract, each time updating the accumulated dictionary:
        for tag in tags_to_extract:
            # Pull the scalar events associated with this tag from all event writers:
            tag_associated_scalar_events = [scalar_event.Scalars(tag) for scalar_event in summary_iterators][0]

            # Record the number of steps taken during all events (as indices):
            event_steps = [event.step for event in tag_associated_scalar_events]

            # Update the accumulated dictionary with the recorded values logged during the event:
            accumulated_output[tag].append([event.value for event in tag_associated_scalar_events])
            # Drop the extra array dimensions inserted by the default-dict (since data already in list):
            accumulated_output[tag] = np.squeeze(accumulated_output[tag])
        return event_steps, accumulated_output

    @staticmethod
    def _tf_event_files_to_csv(event_files, output_path):
        events_df = None
        log_dir = os.path.abspath(os.path.join(event_files[0], os.pardir))
        steps, accumulated_output = TensorBoardDataExporter._tabulate_tensorboard_events(event_files=event_files)
        mean_loss = np.mean(accumulated_output['train_graph/final_retrain_ops/loss_1'])
        mean_acc = np.mean(accumulated_output['train_graph/final_retrain_ops/accuracy_1'])
        mean_top_five_acc = np.mean(accumulated_output['train_graph/final_retrain_ops/top_five_accuracy'])

        tags, values = zip(*accumulated_output.items())
        as_np_array = np.array(values)
        for i, tag in enumerate(tags):
            # df = pd.DataFrame(as_np_array[i], index=np.array(steps), columns=list(tags[i]))
            # df = pd.DataFrame(as_np_array[i], index=np.array(steps), columns=np.array(tags[i]))
            df = pd.DataFrame(as_np_array[i])
            df = df.rename(columns={0: os.path.basename(tag)})
            if i == 0:
                events_df = df.copy()
            else:
                events_df = events_df.join(df)
            # output_fname = tag.replace('/', '_') + '.csv'
            # df.to_csv(os.path.join(log_dir, output_fname))
        events_df.to_csv(output_path)
        return events_df

    def export_all_summaries_as_csv(self):
        for i, dir in enumerate(self.target_dirs):
            file_list = []
            dir_name = os.path.basename(dir)
            tf.logging.info('Converting TensorBoard events to csv files in: \'%s\'' % dir)
            file_glob = os.path.join(dir, 'events.out.tfevents.*')
            file_list.extend(tf.gfile.Glob(file_glob))
            output_dir = os.path.join(dir, 'events.csv')
            events_df = TensorBoardDataExporter._tf_event_files_to_csv(file_list, output_path=output_dir)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    # path = "C:\\Users\\ccamp\Documents\\GitHub\\HerbariumDeep\\frameworks\\TensorFlow\\TFHub\\tmp\\summaries\\val\\"
    __path = 'C:\\Users\\ccamp\Documents\\GitHub\\HerbariumDeep\\frameworks\\TensorFlow\\TFHub\\tmp\\summaries\\'
    tb_exporter = TensorBoardDataExporter(root_summaries_dir=__path)
    tb_exporter.export_all_summaries_as_csv()

