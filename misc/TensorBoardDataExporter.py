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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import cufflinks as cf

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from collections import defaultdict, OrderedDict
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
        # init_notebook_mode(connected=True)
        cf.go_offline()


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
        for i, tag in enumerate(tags_to_extract):
            # Pull the scalar events associated with this tag from all event writers:
            tag_associated_scalar_events = [scalar_event.Scalars(tag) for scalar_event in summary_iterators][0]

            # Record the number of steps taken during all events (as indices):
            event_steps = [event.step for event in tag_associated_scalar_events]

            # Record the time each step was taken (in seconds):
            event_wall_times_in_seconds = [event.wall_time for event in tag_associated_scalar_events]

            # Update the accumulated dictionary with the recorded values logged during the event:
            accumulated_output[tag].append([event.value for event in tag_associated_scalar_events])

            # Drop the extra array dimensions inserted by the default-dict (since data already in list):
            accumulated_output[tag] = np.squeeze(accumulated_output[tag])

            if i == 0:
                # Update the accumulated dictionary with the wall time in seconds of each step (same across metrics):
                accumulated_output['wall_times'] = event_wall_times_in_seconds
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
        hyper_strings = []
        event_dataframes = []
        for i, dir in enumerate(self.target_dirs):
            file_list = []
            dir_name = os.path.basename(dir)
            tf.logging.info('Converting TensorBoard events to csv files in: \'%s\'' % dir)
            file_glob = os.path.join(dir, 'events.out.tfevents.*')
            file_list.extend(tf.gfile.Glob(file_glob))
            output_dir = os.path.join(dir, 'events.csv')
            events_df = TensorBoardDataExporter._tf_event_files_to_csv(file_list, output_path=output_dir)
            hyper_strings.append(dir_name)
            event_dataframes.append(events_df)
        return hyper_strings, event_dataframes

    @staticmethod
    def generate_hyperparameter_heatmap(hyperparameter_strings, hyperparameter_event_dataframes):
        hyperstring_stats = OrderedDict()
        mean_losses = []
        mean_accuracies = []
        mean_top_five_accuracies = []
        for (hyper_string, event_dataframe) in zip(hyperparameter_strings, hyperparameter_event_dataframes):
            mean_loss = np.mean(event_dataframe['loss_1'])
            mean_losses.append(mean_loss)

            mean_acc = np.mean(event_dataframe['accuracy_1'])
            mean_accuracies.append(mean_acc)

            mean_top_five_acc = np.mean(event_dataframe['top_five_accuracy'])
            mean_top_five_accuracies.append(mean_top_five_acc)

            hyperstring_statistics = {'mean_loss': mean_loss, 'mean_acc': mean_acc, 'mean_top_five_acc': mean_top_five_acc}
            hyperstring_stats[hyper_string] = hyperstring_statistics
        hyper_string = list(hyperstring_stats.keys())[0]

        hyper_params_df = pd.DataFrame(columns=['mean_acc', 'mean'])
        raise NotImplementedError

def convert_to_hyperparameter_dataframe(hyper_strings, event_dataframes):
    hyperparams_df = pd.DataFrame(columns=['initializer', 'optimizer', 'activation', 'train_batch_size', 'fit_time_sec', 'mean_acc', 'mean_loss', 'mean_top_five_acc'])
    for hyper_string, event_dataframe in zip(hyper_strings, event_dataframes):
        initializer = hyper_string.split(',')[0]
        optimizer = hyper_string.split(',')[1]
        activation = hyper_string.split(',')[2]
        train_batch_size = int((hyper_string.split(',')[3]).split('__')[-1])
        wall_times = event_dataframe['wall_times'].values
        relative_wall_times = [0]
        elapsed_wall_times = [next_wall_time - current_wall_time for current_wall_time, next_wall_time in zip(wall_times, wall_times[1:])]
        relative_wall_times.extend(elapsed_wall_times)
        fit_time_sec = np.sum(relative_wall_times)
        mean_acc = np.mean(event_dataframe['accuracy_1'])
        mean_loss = np.mean(event_dataframe['loss_1'])
        mean_top_five_acc = np.mean(event_dataframe['top_five_accuracy'])
        df_series = pd.Series({
            'initializer': initializer, 'optimizer': optimizer, 'activation': activation,
            'train_batch_size': train_batch_size, 'fit_time_sec': fit_time_sec, 'mean_acc': mean_acc,
            'mean_loss': mean_loss, 'mean_top_five_acc': mean_top_five_acc
        })
        hyperparams_df = hyperparams_df.append(df_series, ignore_index=True)
    hyperparams_df['train_batch_size'] = hyperparams_df.train_batch_size.astype(int)
    hyperparams_df['fit_time_sec'] = hyperparams_df.fit_time_sec.astype(float)
    hyperparams_df['mean_acc'] = hyperparams_df.mean_acc.astype(float)
    hyperparams_df['mean_loss'] = hyperparams_df.mean_loss.astype(float)
    hyperparams_df['mean_top_five_acc'] = hyperparams_df.mean_top_five_acc.astype(float)
    return hyperparams_df

def plot_scatter_plot(hyperparams_df):
    """
    plot_scatter_plot: Uses plotly to create a 3d scatter plot of the hyperparameter combinations and their accuracies.
    :source url: https://github.com/xoelop/Medium-posts/blob/master/3d%20cross%20validation/ML%206%20-%20Gridsearch%20visulizations%20.ipynb
    :author: Xoel Lopez Barata (modified by Chris Campell)
    :param hyperparams_df:
    :return:
    """
    mean_fit_time = np.mean(hyperparams_df['fit_time_sec'])
    grid_search_text_list = list(
        zip(
            'init: ' + hyperparams_df.initializer.apply(str),
            'optim: ' + hyperparams_df.optimizer.apply(str),
            'activ: ' + hyperparams_df.activation.apply(str),
            'fit time (sec): ' + hyperparams_df.fit_time_sec.apply(str),
            'mean acc: ' + hyperparams_df.mean_acc.apply(str),
            'mean loss: ' + hyperparams_df.mean_loss.apply(str),
            'mean top-k (k=5) acc: ' + hyperparams_df.mean_top_five_acc.apply(str)
        )
    )
    text = ['<br>'.join(t) for t in grid_search_text_list]
    hyperparams_df['Text'] = text
    trace = go.Scatter3d(
        x=hyperparams_df['initializer'],
        y=hyperparams_df['optimizer'],
        z=hyperparams_df['activation'],
        mode='markers',
        marker=dict(
            # size=mean_fit_time ** (1 / 3),
            color=hyperparams_df.mean_acc,
            opacity=0.99,
            colorscale='Viridis',
            colorbar=dict(title='Mean Acc', tickmode='auto', nticks=10),
            line=dict(color='rgb(140, 140, 170)')
        ),
        text=hyperparams_df.Text,
        hoverinfo='text'
    )
    data = [trace]
    layout = go.Layout(
        title='3D Scatter Plot of Grid Search Results',
        margin=dict(
            l=30,
            r=30,
            b=30,
            t=30
        ),
        # height=600,
        # width=960,
        scene=dict(
            xaxis=dict(
                title='initializer',
                nticks=10
            ),
            yaxis=dict(
                title='optimizer'
            ),
            zaxis=dict(
                title='activation'
            ),
        )
    )
    fig = go.Figure(data=data, layout=layout)
    plot(fig)

def plot_scatter_plot_with_sliders(hyperparams_df):
    """
    plot_scatter_plot: Uses plotly to create a 3d scatter plot of the hyperparameter combinations and their accuracies.
    :source url: https://github.com/xoelop/Medium-posts/blob/master/3d%20cross%20validation/ML%206%20-%20Gridsearch%20visulizations%20.ipynb
    :author: Xoel Lopez Barata (modified by Chris Campell)
    :param hyperparams_df:
    :return:
    """
    mean_fit_time = np.mean(hyperparams_df['fit_time_sec'])
    grid_search_text_list = list(
        zip(
            'init: ' + hyperparams_df.initializer.apply(str),
            'optim: ' + hyperparams_df.optimizer.apply(str),
            'activ: ' + hyperparams_df.activation.apply(str),
            'fit time (sec): ' + hyperparams_df.fit_time_sec.apply(str),
            'mean acc: ' + hyperparams_df.mean_acc.apply(str),
            'mean loss: ' + hyperparams_df.mean_loss.apply(str),
            'mean top-k (k=5) acc: ' + hyperparams_df.mean_top_five_acc.apply(str)
        )
    )
    text = ['<br>'.join(t) for t in grid_search_text_list]
    hyperparams_df['Text'] = text
    trace = go.Scatter3d(
        x=hyperparams_df['initializer'],
        y=hyperparams_df['optimizer'],
        z=hyperparams_df['activation'],
        mode='markers',
        marker=dict(
            # size=mean_fit_time ** (1 / 3),
            color=hyperparams_df.mean_acc,
            opacity=0.99,
            colorscale='Viridis',
            colorbar=dict(title='Mean Acc', tickmode='auto', nticks=10),
            line=dict(color='rgb(140, 140, 170)')
        ),
        text=hyperparams_df.Text,
        hoverinfo='text'
    )
    data = [trace]

    steps = []
    for i in range(len(hyperparams_df['train_batch_size'].unique())):
        step = dict(
            method='update',
            args=['visible', [False] * len(hyperparams_df['train_batch_size'].unique())],
        )
        step['args'][1][i] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={'prefix': 'Frequency'},
        pad={'t': 50},
        steps=steps
    )]

    layout = go.Layout(
        title='3D Scatter Plot of Grid Search Results',
        margin=dict(
            l=30,
            r=30,
            b=30,
            t=30
        ),
        # height=600,
        # width=960,
        scene=dict(
            xaxis=dict(
                title='initializer',
                nticks=10
            ),
            yaxis=dict(
                title='optimizer'
            ),
            zaxis=dict(
                title='activation'
            ),
        ),
        sliders=sliders
    )

    fig = go.Figure(data=data, layout=layout)
    plot(fig)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    # path = "C:\\Users\\ccamp\Documents\\GitHub\\HerbariumDeep\\frameworks\\TensorFlow\\TFHub\\tmp\\summaries\\val\\"
    __path = 'C:\\Users\\ccamp\Documents\\GitHub\\HerbariumDeep\\frameworks\\TensorFlow\\TFHub\\tmp\\summaries\\'
    tb_exporter = TensorBoardDataExporter(root_summaries_dir=__path)
    __hyper_strings, __event_dataframes = tb_exporter.export_all_summaries_as_csv()
    hyperparams_df = convert_to_hyperparameter_dataframe(__hyper_strings, __event_dataframes)
    plot_scatter_plot(hyperparams_df)
    plot_scatter_plot_with_sliders(hyperparams_df)
    # plot_simple_scatter_plot(hyperparams_df)
    # hyperparameter_heatmap = TensorBoardDataExporter.generate_hyperparameter_heatmap(hyperparameter_strings=__hyper_strings, hyperparameter_event_dataframes=__event_dataframes)
