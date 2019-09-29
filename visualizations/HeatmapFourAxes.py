import pandas as pd
import numpy as np


def main():
    __path = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\gs_val_hyperparams.pkl'
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
    print('HeatMap Dimensions: %s\n' %(data.shape,))

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
    y_ticks_right_major = np.arange(0, heatmap_dims[0], 0.5)





