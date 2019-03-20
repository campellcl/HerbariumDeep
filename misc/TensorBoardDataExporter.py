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
from pandas.api.types import CategoricalDtype
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
        # Houses the gs/train and gs/val hyperparameter strings:
        gs_hyper_strings = {'train': [], 'val': []}
        # Houses the grid search winner  train/ and val/ hyperparameter strings:
        winner_hyper_strings = {'train': None, 'val': None}
        # Houses the gs/train and gs/val event dataframes:
        gs_event_dataframes = {'train': [], 'val': []}
        winner_event_dataframes = {'train': None, 'val': None}
        # hyper_strings = []
        # event_dataframes = []
        for i, dir in enumerate(self.target_dirs):
            file_list = []
            dir_name = os.path.basename(dir)
            tf.logging.info('Converting TensorBoard events to csv files in: \'%s\'' % dir)
            file_glob = os.path.join(dir, 'events.out.tfevents.*')
            file_list.extend(tf.gfile.Glob(file_glob))
            output_dir = os.path.join(dir, 'events.csv')
            events_df = TensorBoardDataExporter._tf_event_files_to_csv(file_list, output_path=output_dir)
            # hyper_strings.append(dir_name)
            # event_dataframes.append(events_df)
            # Recurse two paths backward to get the root of the logging directory:
            # root_tb_log_dir = os.path.abspath(os.path.join(os.path.join(dir, os.pardir), os.pardir))
            relative_parent_dir = dir[len(self.root_summaries_dir):]
            relative_parent_dir_split = relative_parent_dir.split('\\')
            if relative_parent_dir_split[0] == 'gs':
                # The root folder under self.log_dir is part fo the grid search:
                if relative_parent_dir_split[1] == 'train':
                    # Training grid search run.
                    gs_hyper_strings['train'].append(dir_name)
                    gs_event_dataframes['train'].append(events_df)
                elif relative_parent_dir_split[1] == 'val':
                    # Validation grid search run.
                    gs_hyper_strings['val'].append(dir_name)
                    gs_event_dataframes['val'].append(events_df)
            elif relative_parent_dir_split[0] == 'train':
                # Training run of the winning grid search model:
                winner_hyper_strings['train'] = dir_name
                winner_event_dataframes['train'] = events_df
            elif relative_parent_dir_split[0] == 'val':
                # Validation run of the winning grid search model:
                winner_hyper_strings['val'] = dir_name
                winner_event_dataframes['val'] = events_df
            elif relative_parent_dir_split[0] == 'summaries':
                # root dir
                pass
            else:
                raise NotImplementedError
        return gs_hyper_strings, gs_event_dataframes, winner_hyper_strings, winner_event_dataframes

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

    @staticmethod
    def _convert_grid_search_events_to_hyperparameter_dataframe(gs_hyper_strings, gs_event_dataframes):
        gs_hyperparameter_df = pd.DataFrame(columns=[
            'initializer', 'optimizer', 'activation', 'train_batch_size', 'fit_time_sec', 'best_epoch_idx_by_loss_min',
            'best_epoch_acc', 'best_epoch_loss', 'best_epoch_top_five_acc'
        ])
        # Go through grid search data:
        gs_best_epoch_loss = 1
        if isinstance(gs_hyper_strings, str):
            # Only single entry:
            gs_hyper_strings = [gs_hyper_strings]
            gs_event_dataframes = [gs_event_dataframes]
        for gs_hyper_string, gs_event_dataframe in zip(gs_hyper_strings, gs_event_dataframes):
            initializer = gs_hyper_string.split(',')[0]
            optimizer = gs_hyper_string.split(',')[1]
            activation = gs_hyper_string.split(',')[2]
            train_batch_size = int((gs_hyper_string.split(',')[3]).split('__')[-1])
            wall_times = gs_event_dataframe['wall_times'].values
            relative_wall_times = [0]
            elapsed_wall_times = [next_wall_time - current_wall_time for current_wall_time, next_wall_time in zip(wall_times, wall_times[1:])]
            relative_wall_times.extend(elapsed_wall_times)
            fit_time_sec = np.sum(relative_wall_times)

            '''
            This is for the GridSearch training hyperparameter model runs:
                1. Identify the best performing epoch (according to the minimization of loss) for this particular hyperparameter combination.
                    a) Record the loss associated with this epoch. 
                    b) Record the accuracy associated with this epoch. 
                    c) Record the top five accuracy associated with this epoch.
                2. Check to see if the recorded statistics are the best (according to minimization of loss) across all GridSearch training hyperparameter model runs:
                    a) If this new hyperparameter's best epoch-loss is the smallest/best seen so far, mark this hyperparameter combination as the best.
                    b) Update the associated variables that keep track of the GridSearch's training performance across all hyperparameter combinations.  
            '''
            # 1. Record the index associated with the best performing epoch (smallest loss) in this particular hyperparameter set:
            event_dataframe_best_epoch_index = gs_event_dataframe.loss_1.idxmin(axis=0, skipna=True)

            # Maintain a reference to the best performing epoch in this particular hyperparameter set:
            event_dataframe_best_epoch_by_loss_minimization = gs_event_dataframe.iloc[event_dataframe_best_epoch_index]

            # 1. a) Record the loss associated with the best performing epoch in this particular hyperparameter set:
            event_dataframe_best_epoch_loss = event_dataframe_best_epoch_by_loss_minimization.loss_1

            # 1. b) Record the accuracy associated with the best performing epoch in this particular hyperparameter set:
            event_dataframe_best_epoch_acc = event_dataframe_best_epoch_by_loss_minimization.accuracy_1

            # 2. c) Record the top five accuracy associated with the best performing epoch in this particular hyperparameter set:
            event_dataframe_best_epoch_top_five_acc = event_dataframe_best_epoch_by_loss_minimization.top_five_accuracy

            # 2. Check to see if this hyperparameter set produced the best performing epoch, out of all GridSearch hyperparameter combinations evaluated on the training data:
            if event_dataframe_best_epoch_loss < gs_best_epoch_loss:
                gs_best_epoch_index_by_loss_minimization = event_dataframe_best_epoch_index
                gs_best_epoch_loss = event_dataframe_best_epoch_loss
                gs_best_epoch_acc = event_dataframe_best_epoch_acc
                gs_best_epoch_top_five_acc = event_dataframe_best_epoch_top_five_acc
            df_series = pd.Series({
                'initializer': initializer, 'optimizer': optimizer, 'activation': activation,
                'train_batch_size': train_batch_size, 'fit_time_sec': fit_time_sec,
                'best_epoch_idx_by_loss_min': event_dataframe_best_epoch_index,
                'best_epoch_acc': event_dataframe_best_epoch_acc, 'best_epoch_loss': event_dataframe_best_epoch_loss,
                'best_epoch_top_five_acc': event_dataframe_best_epoch_top_five_acc
            })
            gs_hyperparameter_df = gs_hyperparameter_df.append(df_series, ignore_index=True)
        gs_hyperparameter_df['initializer'] = gs_hyperparameter_df.initializer.astype(dtype='category')
        gs_hyperparameter_df['optimizer'] = gs_hyperparameter_df.optimizer.astype(dtype='category')
        gs_hyperparameter_df['activation'] = gs_hyperparameter_df.activation.astype(dtype='category')
        gs_hyperparameter_df['train_batch_size'] = gs_hyperparameter_df.train_batch_size.astype(int)
        gs_hyperparameter_df['fit_time_sec'] = gs_hyperparameter_df.fit_time_sec.astype(float)
        gs_hyperparameter_df['best_epoch_idx_by_loss_min'] = gs_hyperparameter_df.best_epoch_idx_by_loss_min.astype(int)
        gs_hyperparameter_df['best_epoch_acc'] = gs_hyperparameter_df.best_epoch_acc.astype(float)
        gs_hyperparameter_df['best_epoch_loss'] = gs_hyperparameter_df.best_epoch_loss.astype(float)
        gs_hyperparameter_df['best_epoch_top_five_acc'] = gs_hyperparameter_df.best_epoch_top_five_acc.astype(float)
        return gs_hyperparameter_df

    @staticmethod
    def _convert_to_hyperparameter_dataframes(gs_hyper_strings, gs_event_dataframes, winner_hyper_strings, winner_event_dataframes):
        gs_train_hyperparams_df = TensorBoardDataExporter._convert_grid_search_events_to_hyperparameter_dataframe(gs_hyper_strings=gs_hyper_strings['train'], gs_event_dataframes=gs_event_dataframes['train'])
        gs_val_hyperparams_df = TensorBoardDataExporter._convert_grid_search_events_to_hyperparameter_dataframe(gs_hyper_strings=gs_hyper_strings['val'], gs_event_dataframes=gs_event_dataframes['val'])
        winner_train_hyperparams_df = TensorBoardDataExporter._convert_grid_search_events_to_hyperparameter_dataframe(gs_hyper_strings=winner_hyper_strings['train'], gs_event_dataframes=winner_event_dataframes['train'])
        winner_val_hyperparams_df = TensorBoardDataExporter._convert_grid_search_events_to_hyperparameter_dataframe(gs_hyper_strings=winner_hyper_strings['val'], gs_event_dataframes=winner_event_dataframes['val'])
        return gs_train_hyperparams_df, gs_val_hyperparams_df, winner_train_hyperparams_df, winner_val_hyperparams_df

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
    __gs_hyper_strings, __gs_event_dataframes, __winner_hyper_strings, __winner_event_dataframes = tb_exporter.export_all_summaries_as_csv()
    __gs_train_hyperparams_df, __gs_val_hyperparams_df, __winner_train_hyperparams_df, __winner_val_hyperparams_df = \
        tb_exporter._convert_to_hyperparameter_dataframes(
            __gs_hyper_strings, __gs_event_dataframes,
            __winner_hyper_strings, __winner_event_dataframes
        )
    __export_path = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\tests'
    __gs_train_hyperparams_df.to_pickle(os.path.join(__export_path, 'gs_train_hyperparams.pkl'))
    __gs_val_hyperparams_df.to_pickle(os.path.join(__export_path, 'gs_val_hyperparams.pkl'))
    __winner_train_hyperparams_df.to_pickle(os.path.join(__export_path, 'winner_train_hyperparams.pkl'))
    __winner_val_hyperparams_df.to_pickle(os.path.join(__export_path, 'winner_val_hyperparams.pkl'))

    # ON RESUME: Tie in the rest of this logic with partitioned dataframes into the visuals (new visual for each dataframe)
    # then pickle the output. Decide if I want one dataframe with categorical flags for gs or train or val.


    # hyperparams_df.to_pickle(os.path.join(__path, 'hyperparams.pkl'))
    # plot_scatter_plot(hyperparams_df)
    # plot_scatter_plot_with_sliders(hyperparams_df)
    # plot_simple_scatter_plot(hyperparams_df)
    # hyperparameter_heatmap = TensorBoardDataExporter.generate_hyperparameter_heatmap(hyperparameter_strings=__hyper_strings, hyperparameter_event_dataframes=__event_dataframes)
