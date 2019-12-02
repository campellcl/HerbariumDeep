import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


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
    visualization_types = {'cross_entropy_loss': 'cross_entropy_loss', 'top_one_acc': 'top_one_acc', 'top_five_acc': 'top_five_acc'}
    colorization_types = {'relative_to_data': 'relative_to_data', 'relative_to_acc': 'relative_to_acc'}
    datasets = {'BOONE': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\Boone\\gs_val_hyperparams.pkl', 'GoingDeeper': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\GoingDeeper\\gs_val_hyperparams.pkl'}

    # Choose Visualization Settings:
    # __path = datasets['GoingDeeper']
    __path = datasets['BOONE']

    # visual_type = visualization_types['cross_entropy_loss']
    # colorization_type = colorization_types['relative_to_data']

    visual_type = visualization_types['top_five_acc']
    # visual_type = visualization_types['top_one_acc']
    colorization_type = colorization_types['relative_to_acc']


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

    for i in range(data.shape[0]):
        optim_index = i % num_optimizers
        optimizer = optimizers[optim_index]
        activ_index = i % num_activations
        activation = activations[activ_index]
        optimizer_repr = convert_optimizer_to_axes_label(optimizer)
        y_tick_labels_left_major.append(optimizer_repr)
        y_tick_labels_left_minor.append(activation)
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
            if visual_type == 'cross_entropy_loss':
                # For best epoch cross entropy loss:
                data[i][j] = df_subset.iloc[0].best_epoch_loss
            elif visual_type == 'top_one_acc':
                # For best epoch top-1 accuracy:
                data[i][j] = df_subset.iloc[0].best_epoch_acc
            elif visual_type == 'top_five_acc':
                # For best epoch top-5 accuracy:
                data[i][j] = df_subset.iloc[0].best_epoch_top_five_acc
                # x_tick_labels.append(initializer.split('_')[1:])
            else:
                raise NotImplementedError('Failed to recognize type of visual: \'%s\'' % visual_type)
            if i == 0:
                init_repr = convert_initializer_to_axes_label(initializer=initializer)
                x_tick_labels_bot_major.append(init_repr)
                x_tick_labels_top_major.append('TB=%s' % str(train_batch_size))

    print('x_ticks_bot_major: %s' % x_ticks_bot_major)
    print('x_ticks_bot_minor: %s' % x_ticks_bot_minor)
    print('x_tick_labels_bot_major: %s' % x_tick_labels_bot_major)
    for i, x_tick_label in enumerate(x_tick_labels_bot_major):
        x_tick_labels_bot_minor.append('')
        x_tick_labels_bot_minor.append(x_tick_label)
    print('x_tick_labels_bot_minor: %s' % x_tick_labels_bot_minor)
    print('')
    print('x_ticks_top_major: %s' % x_ticks_top_major)
    print('x_ticks_top_minor: %s' % x_ticks_top_minor)
    print('x_tick_labels_top_major: %s' % x_tick_labels_top_major)
    for i, x_tick_label in enumerate(x_tick_labels_top_major):
        x_tick_labels_top_minor.append('')
        x_tick_labels_top_minor.append(x_tick_label)
    print('x_tick_labels_top_minor: %s' % x_tick_labels_top_minor)
    print('')
    print('y_ticks_left_major: %s' % y_ticks_left_major)
    print('y_ticks_left_minor: %s' % y_ticks_left_minor)
    print('y_tick_labels_left_major: %s' % y_tick_labels_left_major)
    print('y_tick_labels_left_minor: %s' % y_tick_labels_left_minor)
    y_tick_labels_left_combined = []
    for i, (y_tick_label_major, y_tick_label_minor) in enumerate(zip(y_tick_labels_left_major, y_tick_labels_left_minor)):
        y_tick_labels_left_combined.append(y_tick_label_major)
        y_tick_labels_left_combined.append(y_tick_label_minor)
    print('y_tick_labels_left_combined: %s' % y_tick_labels_left_combined)

    # fig = plt.figure('DEBUG DATASET')
    fig, ax = plt.subplots(1, 1, sharey='col')
    # ax_bot = fig.gca()
    ax_bot = ax

    ax_bot.set_xticks(x_ticks_bot_major, minor=False)
    ax_bot.set_xticks(x_ticks_bot_minor, minor=True)
    ax_bot.set_xticklabels(x_tick_labels_bot_major, minor=False)
    # ax_bot.set_xticklabels('', major=True)

    ax_bot.set_yticks(y_ticks_left_major, minor=False)
    ax_bot.set_yticks(y_ticks_left_minor, minor=True)
    ax_bot.set_yticklabels(y_tick_labels_left_major, minor=False, fontsize=16)
    ax_bot.set_xlabel('Initializer', fontsize=20)
    ax_bot.set_ylabel('Optimizer', fontsize=20)

    # ax_right = ax_bot.twinx()
    # ax_right.set_aspect('equal')
    # ax_right.set_yticks(y_ticks_left_major, minor=False)
    # ax_right = plt.Axes(fig=fig, rect=fig.patch)

    ax_top = ax_bot.twiny()
    ax_top.set_xticks(x_ticks_top_major, minor=False)
    ax_top.set_xticks(x_ticks_top_minor, minor=True)
    ax_top.set_xticklabels(x_tick_labels_top_major, minor=False)
    ax_top.set_xlabel('Training Batch Size', fontsize=20, labelpad=15)
    # ax_top.set_xticklabels('', major=True)

    # ax_right = ax_top.get_shared_y_axes().join(ax_bot, ax_top)
    # ax_right.set_yticks(y_ticks_left_major, minor=False)

    # fig.text()

    # Modify font sizes of x-axes:
    ax_bot.tick_params(axis='x', labelsize=16)
    ax_top.tick_params(axis='x', labelsize=16)


    if colorization_type == 'relative_to_data':
        # Uncomment this for plotting colors relative to data:
        ax_bot.imshow(data)
        ax_top.imshow(data)
        cmap = plt.cm.get_cmap('viridis')
        scalar_mappable = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, data.max()))
        scalar_mappable.set_clim(vmin=0, vmax=data.max())
        cbar = plt.colorbar(scalar_mappable)
    elif colorization_type == 'relative_to_acc':
        # Uncomment this for plotting colors relative to 100% accuracy:
        cmap = plt.cm.get_cmap('viridis')
        colors = cmap(data)
        ax_bot.imshow(colors)
        ax_top.imshow(colors)
        scalar_mappable = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=100))
        scalar_mappable.set_clim(vmin=0, vmax=100)
        cbar = plt.colorbar(scalar_mappable, ticks=np.arange(0, 110, 10))

    # Legacy code:
    # scalar_mappable = cm.ScalarMappable(cmap=plt.get_cmap(name='viridis'), norm=plt.Normalize(vmin=0, vmax=data.max()))
    # scalar_mappable = cm.ScalarMappable(cmap=plt.get_cmap(name='viridis'), norm=plt.Normalize(vmin=0, vmax=100))
    # scalar_mappable._A = []
    # scalar_mappable.set_clim(vmin=0, vmax=100)
    # scalar_mappable.set_clim(vmin=data.min(), vmax=data.max())
    # cbar = plt.colorbar(mappable=scalar_mappable, ticks=np.arange(0, 110, 10.0))
    # cbar = plt.colorbar(mappable=scalar_mappable, ticks=np.arange(data.min(), data.max() + 10.0, 10.0))

    if visual_type == 'cross_entropy_loss':
        # For x-entropy loss:
        # cbar = plt.colorbar(mappable=scalar_mappable)
        cbar.set_label('Cross Entropy Loss', rotation=270, labelpad=25, fontsize=24)
        plt.title('Grid Search Hyperparameter Settings and Validation Set X-Entropy Loss', fontsize=24, pad=15)
    elif visual_type == 'top_one_acc':
        # For top-1 accuracy:
        if colorization_type == 'relative_to_data':
            cbar.set_label('Top-1 Accuracy', rotation=270, labelpad=25)
        else:
            cbar.set_label('Top-1 Accuracy (Percentage)', rotation=270, labelpad=25, fontsize=24)
        plt.title('Grid Search Hyperparameter Settings and Validation Set Top-1 Accuracy', fontsize=24, pad=15)
    elif visual_type == 'top_five_acc':
        # For top-5 accuracy:
        if colorization_type == 'relative_to_data':
            cbar.set_label('Top-5 Accuracy', rotation=270, labelpad=25)
        else:
            cbar.set_label('Top-5 Accuracy (Percentage)', rotation=270, labelpad=25, fontsize=24)
        plt.title('Grid Search Hyperparameter Settings and Validation Set Top-5 Accuracy', fontsize=24, pad=15)
    cbar.ax.tick_params(labelsize=16)
    plt.show()


if __name__ == '__main__':
    main()
