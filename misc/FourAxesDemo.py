import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# heatmap_dims = (4, 6)


def convert_initializer_to_axes_label(initializer):
    if initializer == 'INIT_HE_NORMAL':
        return 'HE-NORM'
    elif initializer == 'INIT_HE_UNIFORM':
        return 'HE-UNIF'
    elif initializer == 'INIT_NORMAL_TRUNCATED':
        return 'NORM-TRUNC'
    else:
        print('ERROR: Could not identify initializer: %s' % initializer)
        return None


def convert_optimizer_to_axes_label(optimizer):
    if optimizer == 'OPTIM_ADAM':
        return 'ADAM'
    elif optimizer == 'OPTIM_NESTEROV':
        return 'NESTEROV'
    else:
        print('ERROR: Could not identify optimizer: %s' % optimizer)
        return None


def main():
    __path = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\tests\\gs_val_hyperparams.pkl'
    df = pd.read_pickle(__path)
    df['activation'] = df['activation'].str.replace('ACTIVATION_ELU', 'ELU').str.replace('ACTIVATION_LEAKY_RELU', 'RELU')
    df.activation = df.activation.astype('category')
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
    print('HeatMap Dimensions: %s\n' % (data.shape,))

    x_tick_labels_bot_major = []
    x_tick_labels_bot_minor = []
    x_ticks_bot_major = np.arange(0, heatmap_dims[1], 1)
    x_ticks_bot_minor = np.arange(0, heatmap_dims[1], 0.5)

    x_tick_labels_top_major = []
    x_tick_labels_top_minor = []
    x_ticks_top_major = np.arange(0, heatmap_dims[1], 1)
    x_ticks_top_minor = np.arange(0, heatmap_dims[1], 0.5)

    y_tick_labels_left_major = []
    y_tick_labels_left_minor = []
    y_ticks_left_major = np.arange(0, heatmap_dims[0], 1)
    y_ticks_left_minor = np.arange(0, heatmap_dims[0], 0.5)

    y_tick_labels_right_major = []
    y_tick_labels_right_minor = []
    y_ticks_right_major = np.arange(0, heatmap_dims[0], 1)
    y_ticks_right_minor = np.arange(0, heatmap_dims[0], 0.5)

    for i in range(data.shape[0]):
        optim_index = i // 2
        optimizer = optimizers[optim_index]
        activ_index = i % 2
        activation = activations[activ_index]
        optimizer_repr = convert_optimizer_to_axes_label(optimizer)
        y_tick_labels_left_major.append(optimizer_repr)
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
                init_repr = convert_initializer_to_axes_label(initializer=initializer)
                x_tick_labels_bot_major.append(init_repr)
                x_tick_labels_top_major.append('TB=%s' % str(train_batch_size))

    fig, ax = plt.subplots(1, 1, sharex='row', sharey='col')
    print(type(ax))
    ax_bot = ax
    print(type(ax_bot))
    print(dir(ax_bot))

    ax_bot.set_adjustable('box')
    # ax_bot.set_adjustable('box-forced')
    ax_bot.set_aspect('equal')

    ax_bot.set_xticks(x_ticks_bot_major, minor=False)
    ax_bot.set_xticks(x_ticks_bot_minor, minor=True)
    ax_bot.set_yticks(y_ticks_left_major, minor=False)
    ax_bot.set_yticks(y_ticks_left_minor, minor=True)

    ax_bot.set_xticklabels(x_tick_labels_bot_major, minor=False)
    ax_bot.set_yticklabels(y_tick_labels_left_major, minor=False)

    print('x_ticks_bot_major: %s' % x_ticks_bot_major)
    print('x_ticks_bot_minor: %s' % x_ticks_bot_minor)
    print('x_tick_labels_bot_major: %s' % x_tick_labels_bot_major)
    for i, x_tick_label in enumerate(x_tick_labels_bot_major):
        x_tick_labels_bot_minor.append('')
        x_tick_labels_bot_minor.append(x_tick_label)
    print('x_tick_labels_bot_minor: %s' % x_tick_labels_bot_minor)
    print('')

    ax_top = ax_bot.twiny()
    # ax_top.set_adjustable('box')
    # ax_top.set_aspect('equal')
    ax_top.set_xticks(x_ticks_top_major, minor=False)
    ax_top.set_xticks(x_ticks_top_minor, minor=True)
    ax_top.set_yticks(y_ticks_right_major, minor=False)

    ax_top.set_xticklabels(x_tick_labels_top_major, minor=False)

    print('x_ticks_top_major: %s' % x_ticks_top_major)
    print('x_ticks_top_minor: %s' % x_ticks_top_minor)
    print('x_tick_labels_top_major: %s' % x_tick_labels_top_major)
    print('')
    # plt.setp(ax_bot, aspect=1.0, adjustable='box-forced')
    # This one:
    # plt.setp(ax_bot, adjustable='datalim')

    ax_right = ax_top.twinx()
    ax_right.set_yticks(y_ticks_right_major, minor=False)
    ax_right.set_yticks(y_ticks_right_minor, minor=True)

    ax_right.set_yticklabels(y_tick_labels_left_minor, minor=False)
    # ax_right.set_yticklabels(y_tick_labels_right_minor, minor=False)

    print('y_ticks_left_major: %s' % y_ticks_left_major)
    print('y_ticks_left_minor: %s' % y_ticks_left_minor)
    print('y_tick_labels_left_major: %s' % y_tick_labels_left_major)
    # for i, y_tick_label in enumerate(y_tick_labels_left_minor):
    #     y_tick_labels_right_minor
    print('y_tick_labels_left_minor: %s' % y_tick_labels_left_minor)
    # plt.setp(ax_bot, aspect='equal', datalim='')

    # ax_bot = ax_bot.twinx()
    # ax_right = ax_bot.twinx()

    ax_bot.imshow(data)
    # ax_top.imshow(data)

    plt.show()


if __name__ == '__main__':
    main()
