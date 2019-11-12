import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    __path = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\Boone\\gs_val_hyperparams.pkl'
    df = pd.read_pickle(__path)
    optimizers = df.optimizer.unique()
    num_optimizers = len(optimizers)
    print('Optimizers: %s' % optimizers.categories)

    activations = df.activation.unique()
    num_activations = len(activations)
    print('Activations: %s' % activations.categories)

    train_batch_sizes = df.train_batch_size.unique()
    num_train_batch_sizes = len(train_batch_sizes)
    print('Train Batch Sizes: %s' % train_batch_sizes)

    initializers = df.initializer.unique()
    num_initializers = len(initializers)
    print('Initializers: %s' % initializers.categories)

    heatmap_dims = ((num_activations * num_optimizers), (num_initializers * num_train_batch_sizes))
    data = np.zeros(heatmap_dims)
    print('HeatMap Dimensions: %s' %(data.shape,))

    print('Columns: %s\n' % df.columns.values)

    fig = plt.figure()
    ax_bot = plt.gca()

    for i in range(data.shape[0]):
        optim_index = i % num_optimizers
        optimizer = optimizers[optim_index]
        activ_index = i % num_activations
        activation = activations[activ_index]
        # optimizer_repr = convert_optimizer_to_axes_label(optimizer)
        # y_tick_labels_left_major.append(optimizer_repr)
        # y_tick_labels_left_minor.append(activation)
        for j in range(data.shape[1]):
            tb_index = j % num_train_batch_sizes
            train_batch_size = train_batch_sizes[tb_index]
            initializer_index = j % num_initializers
            initializer = initializers[initializer_index]

            df_subset = df[df.optimizer == optimizer]
            df_subset = df_subset[df_subset.activation == activation]
            df_subset = df_subset[df_subset.train_batch_size == train_batch_size]
            df_subset = df_subset[df_subset.initializer == initializer]
            assert df_subset.shape[0] == 1
            data[i][j] = df_subset.iloc[0].best_epoch_loss
            # x_tick_labels.append(initializer.split('_')[1:])
            if i == 0:
                # init_repr = convert_initializer_to_axes_label(initializer=initializer)
                # x_tick_labels_bot_major.append(init_repr)
                # x_tick_labels_top_major.append('TB=%s' % str(train_batch_size))
                pass

    # ax_bot.imshow(data)
    # ax_bot.imshow(df['best_epoch_loss'].values.reshape(heatmap_dims[0], heatmap_dims[1]))

    unique_activation_types = df['activation'].cat.categories.unique().values
    # ax_bot.set_xticklabels(unique_activation_types)
    ax_bot.set_xticks(np.arange(0, heatmap_dims[1], 1))
    ax_bot.set_xticklabels(df['activation'].apply(lambda act: act.split('_')[-1]))

    ax_top = ax_bot.twiny()
    ax_top.set_xticks(np.arange(0, heatmap_dims[1], 1))
    ax_top.set_xticklabels(df['train_batch_size'])
    # ax_top.autoscale(enable=False, tight=True)
    ax_top.set_xlim(ax_bot.get_xlim())
    # ax_bot_pos = ax_bot.get_position()
    # ax_top.set_major_locator(ax_bot.axes_locator)

    ax_bot.set_yticks(np.arange(0, heatmap_dims[0], 1))
    ax_bot.set_yticklabels(df['optimizer'].apply(lambda opt: opt.split('_')[-1]))

    plt.show()


if __name__ == '__main__':
    main()
