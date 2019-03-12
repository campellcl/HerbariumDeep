import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
# from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import inflect



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
    cols = ['optimizer', 'activation', 'mean_acc']
    x = [i for i in range(len(cols) - 1)]
    optimizer_colors = ['blue', 'black', 'red', 'green']
    mean_acc_colors = ['#2e8ad8', '#cd3785', 'yellow', 'green', 'blue']
    mean_acc_cut = pd.cut(df.mean_acc, [0.0, 0.25, 0.5, 0.75, 1.0])
    mean_acc_color_mappings = {mean_acc_cut.cat.categories[i]: mean_acc_colors[i] for i, _ in enumerate(mean_acc_cut.cat.categories)}

    # color_mapping = {pd.cut(df['mean_acc']}
    fig, axes = plt.subplots(1, len(x), sharey='none', figsize=(15,2))

    # min, max, and range for each column:
    min_max_range = {}
    for col in cols:
        if col == 'optimizer' or col == 'activation':
            min_max_range[col] = [df[col].cat.codes.min(), df[col].cat.codes.max(), np.ptp(df[col].cat.codes)]
        else:
            min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
            # Normalize the column:
            df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))

    # Plot each row
    for i, ax in enumerate(axes):
        for idx in df.index:
            mean_acc_interval = mean_acc_cut.loc[idx]
            ax.plot(x, df.loc[idx, ['optimizer', 'activation']], mean_acc_color_mappings[mean_acc_interval])
        if len(axes) == 2:
            if i == 0:
                ax.set_xlim([x[i], x[i+1]])
            elif i == 1:
                ax.set_xlim(x[i], x[i]+1)
        else:
            ax.set_xlim([x[i], x[i+1]])

    # set tick positions and labels on y axis for each plot
    def set_ticks_for_axis(dim, ax, ticks):
        min_val, max_val, val_range = min_max_range[cols[dim]]
        step = val_range / float(ticks-1)

        if dim == 0 or dim == 1:
            tick_labels = [df[cols[dim]].cat.categories[code] for code in df[cols[dim]].cat.codes.unique()]
        else:
            tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
        try:
            norm_min = df[cols[dim]].min()
        except TypeError:
            norm_min = df[cols[dim]].cat.codes.min()
        try:
            norm_range = np.ptp(df[cols[dim]])
        except TypeError:
            norm_range = np.ptp(df[cols[dim]].cat.codes)
        norm_step = norm_range / float(ticks-1)
        ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
        ax.yaxis.set_ticks(ticks)
        ax.set_yticklabels(tick_labels)

    for dim, ax in enumerate(axes):
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        set_ticks_for_axis(dim, ax, ticks=2)
        ax.set_xticklabels([cols[dim]])

    # Move final axis' ticks to right-hand side
    ax = plt.twinx(axes[-1])
    dim = len(axes)
    ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    set_ticks_for_axis(dim, ax, ticks=2)
    ax.set_xticklabels([cols[-2], cols[-1]])

    # Remove space between subplots:
    plt.subplots_adjust(wspace=0)

    plt.show()

if __name__ == '__main__':
    _path = 'C:\\Users\\ccamp\Documents\\GitHub\\HerbariumDeep\\frameworks\\TensorFlow\\TFHub\\tmp\\summaries\\hyperparams.pkl'
    df = pd.read_pickle(_path)
    parallel_coords(df)

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
