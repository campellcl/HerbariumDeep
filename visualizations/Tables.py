import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    __path = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\tests\\gs_val_hyperparams.pkl'
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
    print('HeatMap Dimensions: %s' % (data.shape,))

    print('Columns: %s\n' % df.columns.values)

    cols_top = optimizers

    num_rows = data.shape[0]
    num_cols = data.shape[1]

    index = np.arange(len(cols_top))

    # Plot bars and create text labels for the table
    bar_width = 0.4
    cell_text = []
    for row in range(num_rows):
        plt.bar(index, df.iloc[row]['best_epoch_acc'], bar_width)
        cell_text.append('?')

    # Reverse colors and text labels to display the last value at the top.
    cell_text.reverse()

    # Add a table at the bottom of the axes:
    accuracy_table = plt.table(cellText=cell_text, colLables=cols_top)

    plt.show()

    columns = ('')


if __name__ == '__main__':
    main()
