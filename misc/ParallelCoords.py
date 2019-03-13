import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from sklearn.datasets import load_iris
# from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
# import inflect




# inflect_engine = inflect.engine()

# plt.figure()
# parallel_coordinates(df[['initializer', 'optimizer', 'activation']], 'initializer')
# parallel_coordinates(df[['initializer', 'optimizer', 'activation', 'mean_acc']], 'initializer')
# parallel_coordinates(df[['initializer', 'optimizer', 'mean_acc']], 'mean_acc')
# plt.show()
# parallel_coordinates(df, 'initializer')
#
# cols = ['initializer', 'optimizer', 'activation']
# x = [i for i in range(len(cols))]
# colors = ['blue', 'red', 'green']

# def parallel_coordinates(data_sets, style=None):
#
#     dims = len(data_sets[0])
#     x    = range(dims)
#     fig, axes = plt.subplots(1, dims-1, sharey='none')
#
#     if style is None:
#         style = ['r-']*len(data_sets)
#
#     # Calculate the limits on the data
#     min_max_range = list()
#     for m in zip(*data_sets):
#         mn = min(m)
#         mx = max(m)
#         if mn == mx:
#             mn -= 0.5
#             mx = mn + 1.
#         r  = float(mx - mn)
#         min_max_range.append((mn, mx, r))
#
#     # Normalize the data sets
#     norm_data_sets = list()
#     for ds in data_sets:
#         nds = [(value - min_max_range[dimension][0]) /
#                 min_max_range[dimension][2]
#                 for dimension,value in enumerate(ds)]
#         norm_data_sets.append(nds)
#     data_sets = norm_data_sets
#
#     # Plot the datasets on all the subplots
#     for i, ax in enumerate(axes):
#         for dsi, d in enumerate(data_sets):
#             ax.plot(x, d, style[dsi])
#         ax.set_xlim([x[i], x[i+1]])
#
#     # Set the x axis ticks
#     for dimension, (axx,xx) in enumerate(zip(axes, x[:-1])):
#         axx.xaxis.set_major_locator(ticker.FixedLocator([xx]))
#         ticks = len(axx.get_yticklabels())
#         labels = list()
#         step = min_max_range[dimension][2] / (ticks - 1)
#         mn   = min_max_range[dimension][0]
#         for i in range(ticks):
#             v = mn + i*step
#             labels.append('%4.2f' % v)
#         axx.set_yticklabels(labels)
#
#
#     # Move the final axis' ticks to the right-hand side
#     axx = plt.twinx(axes[-1])
#     dimension += 1
#     axx.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
#     ticks = len(axx.get_yticklabels())
#     step = min_max_range[dimension][2] / (ticks - 1)
#     mn   = min_max_range[dimension][0]
#     labels = ['%4.2f' % (mn + i*step) for i in range(ticks)]
#     axx.set_yticklabels(labels)
#
#     # Stack the subplots
#     plt.subplots_adjust(wspace=0)
#
#     return plt


# if __name__ == '__main__':
    # import random
    # base  = [0,   0,  5,   5,  0]
    # scale = [1.5, 2., 1.0, 2., 2.]
    # data = [[base[x] + random.uniform(0., 1.)*scale[x]
    #         for x in range(5)] for y in range(30)]
    # colors = ['r'] * 30
    #
    # base  = [3,   6,  0,   1,  3]
    # scale = [1.5, 2., 2.5, 2., 2.]
    # data.extend([[base[x] + random.uniform(0., 1.)*scale[x]
    #              for x in range(5)] for y in range(30)])
    # colors.extend(['b'] * 30)
    #
    # parallel_coordinates(data, style=colors).show()


def parallel_coords(df):
    df['train_batch_size'] = df.train_batch_size.astype('category')
    df.train_batch_size = df.train_batch_size.apply(str)
    # df['train_batch_size_encoded'] = df.train_batch_size.cat.codes
    cols = ['optimizer', 'activation', 'train_batch_size', 'mean_acc']
    x = [i for i in range(len(cols) - 1)]   # -1 for colorbar var. 'mean_acc' which is excluded, len(cols) not len(cols)-1 because shared y-axis.
    mean_acc_colors = ['red', 'orange', 'yellow', 'green', 'blue']
    mean_acc_cut = pd.cut(df.mean_acc, [0.0, 0.25, 0.5, 0.75, 1.0])
    mean_acc_color_mappings = {mean_acc_cut.cat.categories[i]: mean_acc_colors[i] for i, _ in enumerate(mean_acc_cut.cat.categories)}

    fig = plt.figure()
    # First axis is for optimizer:
    optimizer_axis = plt.subplot(1, len(x), 1)
    fig.add_subplot(optimizer_axis, sharex=None, sharey=None)
    # plt.setp(optimizer_axis.get_xticklabels(), fontsize=6)

    # Second axis is for activation:
    activation_axis = plt.subplot(1, len(x), 2)
    fig.add_subplot(activation_axis, sharex=None, sharey=None)

    # Third axis is for train_batch_size and does sharex:
    # train_batch_axis = plt.subplot(1, len(x), 3)
    # fig.add_subplot(train_batch_axis, sharex=activation_axis, sharey=None)
    # fig.add_subplot()

    # Third axis is for colorbar:
    cax = plt.subplot(1, len(x), 3)
    fig.add_subplot(cax, sharex=None, sharey=None)

    # axes = [optimizer_axis, activation_axis, train_batch_axis, cax]
    axes = [optimizer_axis, activation_axis, cax]

    # min, max, and range for each column:
    min_max_range = {}
    for col in cols:
        if col == 'optimizer' or col == 'activation' or col == 'train_batch_size':
            # Range for categorical's is dependent upon number of unique categories:
            min_max_range[col] = [df[col].cat.codes.min(), df[col].cat.codes.max(), np.ptp(df[col].cat.codes)]
        else:
            min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
            # Normalize the column:
            df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))

    # Plot each row
    for i, ax in enumerate(axes):
        if i == len(axes) - 1:
            continue
        else:
            for idx in df.index:
                mean_acc_interval = mean_acc_cut.loc[idx]
                ax.plot(x, df.loc[idx, ['optimizer', 'activation', 'train_batch_size']], mean_acc_color_mappings[mean_acc_interval])
            ax.set_xlim([x[i], x[i+1]])

    # Save the original tick labels for the last axis:
    df_y_tick_labels = [tick.get_text() for tick in axes[0].get_yticklabels(minor=False)]

    # set tick positions and labels on y axis for each plot
    def set_ticks_for_axis(dim, ax, categorical, ticks, ytick_labels=None):
        min_val, max_val, val_range = min_max_range[cols[dim]]
        step = val_range / float(ticks-1)

        # For final column:
        if categorical:
            norm_min = df[cols[dim]].cat.codes.min()
            norm_range = np.ptp(df[cols[dim]].cat.codes)
        else:
            norm_min = df[cols[dim]].min()
            norm_range = np.ptp(df[cols[dim]])
        norm_step = norm_range / float(ticks-1)

        if not ytick_labels:
            df_tick_labels = ax.get_yticklabels(minor=False)
            tick_labels = [tick_label.get_text().split('_')[-1] for tick_label in df_tick_labels]
        else:
            tick_labels = ytick_labels
        ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
        if dim == 0:
            # Optimizer
            relevant_tick_labels = [0, len(tick_labels)-1]
        elif dim == 1:
            # Activation
            relevant_tick_labels = [1, len(tick_labels)-2]
        elif dim == 2:
            # Train batch size
            relevant_tick_labels = [2, 3]
        else:
            relevant_tick_labels = None
        tick_labels = [tick_labels[i] if i in relevant_tick_labels else '' for i in range(len(tick_labels))]
        ax.set_yticklabels(tick_labels)
        # ax.set_you
        # ax.set_ylim([0, 1], auto=True)
        # ax.autoscale(enable=True, axis=ax.yaxis)

    for dim, ax in enumerate(axes):
        if dim == len(axes) - 1:
            ax.xaxis.set_major_locator(ticker.FixedLocator([0]))
            set_ticks_for_axis(dim, ax, ytick_labels=df_y_tick_labels, categorical=True, ticks=2)
            ax.set_xticklabels([cols[dim]])
        else:
            ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
            set_ticks_for_axis(dim, ax, ytick_labels=None, categorical=True, ticks=2)
            ax.set_xticklabels([cols[dim]])


    # Move final axis' ticks to right-hand side
    # ax = plt.twinx(axes[1])
    # dim = 1
    # ax.xaxis.set_major_locator(ticker.FixedLocator([x[0], x[1]]))
    # set_ticks_for_axis(dim=dim, ax=ax, ticks=2)

    # ax.set_xticklabels
    # dim = len(axes)
    # ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    # set_ticks_for_axis(dim, ax, ticks=2)
    # ax.set_xticklabels([cols[-2], cols[-1]])

    # Remove space between subplots:
    plt.subplots_adjust(wspace=0)

    # Remove unused parts of x-axis
    axes[-1].spines['right'].set_visible(False)
    axes[-1].spines['top'].set_visible(False)
    axes[-1].spines['bottom'].set_visible(False)

    # Add colorbar:
    # cax = plt.twinx(axes[-1])
    # fig.axes[-1].imshow(df['mean_acc'].values, interpolation='nearest', cmap=cm.coolwarm)

    # custom colormap:
    # cdic

    # cbar = fig.colorbar(fig.axes[-1], ticks=[0, 1, 2, 3], orientation='vertical')

    # add legend:
    plt.legend(
        [plt.Line2D((0, 1), (0, 0), color=mean_acc_color_mappings[cat]) for cat in mean_acc_cut.cat.categories],
        mean_acc_cut.cat.categories, bbox_to_anchor=(1.2, 1), loc=0, borderaxespad=0.0
    )

    # cbar.ax.set_yticklabels(['< -1', '0', '> 1'])

    plt.title('Accuracy with varying hyperparameters')

    plt.show()


if __name__ == '__main__':
    # _path = 'C:\\Users\\ccamp\Documents\\GitHub\\HerbariumDeep\\frameworks\\TensorFlow\\TFHub\\tmp\\summaries\\hyperparams.pkl'
    __path = 'C:\\Users\\ccamp\Documents\\GitHub\\HerbariumDeep\\tests\\hyperparams.pkl'
    df = pd.read_pickle(__path)
    parallel_coords(df)
    print('')
    # scale = 10
    # df['initializer_encoded'] = df.initializer.cat.codes * scale
    # df['optimizer_encoded'] = df.optimizer.cat.codes * scale
    # df['activation_encoded'] = df.activation.cat.codes * scale
    # # df['train_batch_size'] = df.train_batch_size.astype('category')
    # df['mean_acc'] = pd.cut(df['mean_acc'], [0, 0.25, 0.5, 0.75, 1.0])
    # parallel_coordinates(df[['initializer_encoded', 'activation_encoded', 'optimizer_encoded', 'train_batch_size', 'mean_acc']], class_column='mean_acc', colormap='viridis')
    # ax = plt.gca()
    # for i, (label, val) in df.loc[:, ['initializer', 'initializer_encoded']].drop_duplicates().iterrows():
    #     ax.annotate(label, xy=(0, val), ha='left', va='center')
    # plt.show()

    # data = load_iris()
    # df = pd.DataFrame(data.data, columns=data.feature_names)
    # mappings = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    # df['target'] = data.target
    # print(df.head())
    # plt.figure()
    # parallel_coordinates(df, 'target')
    # plt.show()

    # parallel_coordinates(df[['initializer', 'optimizer', 'activation', 'mean_acc']], 'initializer')
    # parallel_coordinates(df, cols=['sepal length (cm)', 'sepal width (cm)'], class_column=['sepal length (cm)'])
    # parallel_coordinates(df, style=colors)
