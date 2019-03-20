import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

def main():
    __path = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\tests\\gs_val_hyperparams.pkl'
    df = pd.read_pickle(__path)
    optimizers = df.optimizer.unique()
    num_optimizers = len(optimizers)
    print('Optimizers: %s' % optimizers)

    activations = df.activation.unique()
    num_activations = len(activations)
    print('Activations: %s' % activations)

    train_batch_sizes = df.train_batch_size.unique()
    num_train_batch_sizes = len(train_batch_sizes)
    print('Train Batch Sizes: %s' % train_batch_sizes)

    initializers = df.initializer.unique()
    num_initializers = len(initializers)
    print('Initializers: %s' % initializers)

    heatmap_dims = ((num_activations * num_optimizers), (num_initializers * num_train_batch_sizes))
    data = np.zeros(heatmap_dims)
    print(data.shape)
    x_tick_labels_bot = []
    x_ticks_bot = np.arange(0, heatmap_dims[1], 1)
    x_tick_labels_top = []
    x_ticks_top_major = np.arange(0, heatmap_dims[1], 1)
    x_ticks_top_minor = np.arange(0, heatmap_dims[1], 0.5)
    y_tick_labels_left_major = []
    y_tick_labels_left_minor = []

    for i in range(data.shape[0]):
        optim_index = i // 2
        optimizer = optimizers[optim_index]
        activ_index = i % 2
        activation = activations[activ_index]
        y_tick_labels_left_major.append(optimizer)
        y_tick_labels_left_minor.append(activation)
        for j in range(data.shape[1]):
            tb_index = j % 2
            train_batch_size = train_batch_sizes[tb_index]
            initializer_index = j // 2
            initializer = initializers[initializer_index]

            df_subset = df[df.optimizer == optimizer]
            df_subset = df_subset[df_subset.activation == activation]
            df_subset = df_subset[df_subset.train_batch_size == train_batch_size]
            df_subset = df_subset[df_subset.initializer == initializer]
            assert df_subset.shape[0] == 1
            data[i][j] = df_subset.iloc[0].best_epoch_loss
            # x_tick_labels.append(initializer.split('_')[1:])
            if i == 0:
                x_tick_labels_bot.append(''.join(initializer.split('_')[1:]))
                x_tick_labels_top.append('TB=%s' % str(train_batch_size))
    # fig = plt.figure(1, figsize=(4, 6))
    fig, axes = plt.subplots(1, 2)
    ax_bot = axes[0]
    ax_bot.imshow(data)
    # plt.imshow(data)
    print('xticks_bot: %s' % x_ticks_bot)
    print('xtick_labels_bot: %s' % x_tick_labels_bot)
    print('xtick_labels_top: %s' % x_tick_labels_top)
    # print('ytick_labels_left_major: %s' % y_tick_labels_left_major)
    # print('ytick_labels_left_minor: %s' % y_tick_labels_left_minor)
    ax_bot.set_xticks(x_ticks_bot, minor=False)
    ax_bot.set_xticklabels(x_tick_labels_bot, minor=False)
    ax_bot.set_yticklabels(y_tick_labels_left_major, minor=False)
    # ax_bot.set_yticklabels(y_tick_labels_left_minor, minor=True)
    # ax_top = ax_bot.twiny()
    ax_top = ax_bot.twiny()
    ax_top.set_xticks(ax_bot.get_xticks(), minor=False)
    ax_top.set_xticklabels(x_tick_labels_top, minor=False)

    # ax_top.xaxis.tick_top()
    # print(type(ax_top))
    # ax_top = fig.axes.append(ax_bot)
    # ax_top.set_xticks(ax_bot.get_xticks(), minor=False)
    # ax_top.set_xticks(x_ticks_top_minor, minor=True)
    # ax_top.set_xticklabels(x_tick_labels_top, minor=True)

    # print('xticks_top: %s' % x_ticks_top)
    # ax_top.set_xticks(x_ticks_top, minor=False)
    # ax_top.set_xticklabels(x_tick_labels_top)
    # ax_top.tick_params('x', length=6, width=2)
    # ax_top.tick_params('x', )
    scalar_mappable = cm.ScalarMappable(cmap=plt.cm.get_cmap('viridis'), norm=plt.Normalize(vmin=0, vmax=1))
    scalar_mappable._A = []
    plt.colorbar(scalar_mappable, cax=axes[1])
    plt.show()

    # TODO: plt.text don't clone axis.
    # TODO: plot log loss.



if __name__ == '__main__':
    main()

