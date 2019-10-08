import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def plot_eval_metrics(df):
    fig, ax = plt.subplots(1, 1, facecolor='w')
    # ax.set_xticks(np.linspace(0, 10, 1))
    plt.title('Validation Evaluation Metrics')
    # val_acc_top_1_scatter = plt.scatter(df['best_epoch_loss'], df['best_epoch_acc'])
    # x_axis_ticks = np.linspace(0, 10, 1)
    cm = plt.cm.get_cmap('viridis')
    ax.scatter(np.arange(0, len(df['best_epoch_loss']), 1), df['best_epoch_loss'])
    plt.show()


def plot_bar_chart_train_batch_size_vs_train_time(df):
    # df[['fit_time_sec', 'initializer']].plot.hist(stacked=True, cumulative=True)
    # fig, ax = plt.subplots()
    # plt.title('Training Batch Size vs. Fit Time')
    # plt.xlabel('Training Batch Size')
    # plt.ylabel('Fit Time (minutes)')
    # fit_time_in_min = df['fit_time_sec'].apply(lambda x: x / 60)
    # train_batch_size = df['train_batch_size']
    # labels = ['20', '60', '100', '1000']
    # # Label locations:
    # x = np.arange(len(labels))
    # bar_width = 0.35
    # rects = ax.bar(x + bar_width, fit_time_in_min, bar_width, label='Men')
    # plt.show()
    pass


def plot_boxplot_train_batch_size_vs_train_time(df, data_set='BOONE', process='Validation'):
    df['fit_time_min'] = df['fit_time_sec'].apply(lambda x: x / 60)
    plot = df.boxplot(column='fit_time_min', by='train_batch_size')
    plt.xlabel('Training Batch Size')
    plt.ylabel('Fit Time (Minutes)')
    plt.title('Training Time vs. Training Batch Size')
    plt.suptitle("%s %s Set" % (data_set, process))
    plt.show()


def plot_barplot_initializer_vs_best_epoch_acc(df):
    mean_best_epoch_acc = df['best_epoch_acc'].mean()


def mean_of_each_categorical_initializer(df):
    df_cat = df[['initializer', 'best_epoch_acc']]
    df_cat['initializer'] = df_cat['initializer'].astype('category')
    print(df_cat.groupby(['initializer'], as_index=False).mean())
    means = df_cat.groupby(['initializer'], as_index=False).mean()
    means.plot(kind='bar', x=['initializer'], y=['best_epoch_acc'])
    means_values = means['best_epoch_acc'].values


def plot_2d_hist_training_batch_size_vs_best_performing_epoch_acc(df, data_set='BOONE', process='Validation'):
    # Optimization function and optimizer vs accuracy:
    fig = plt.figure()
    # plt.hist2d(df['best_epoch_acc'], )
    # plt.hist2d(df['best_epoch_acc'].values, df['initializer'].cat.codes, bins=10)
    # plt.hist(df['best_epoch_acc'])
    plt.suptitle('%s %s Set' % (data_set, process))
    plt.title('Training Batch Size vs. Best Performing Epoch Accuracy')
    plt.scatter(df['train_batch_size'], df['best_epoch_acc'])
    plt.ylabel('Best Epoch Accuracy')
    plt.xlabel('Training Batch Size')
    # plt.yticks(ticks=df['train_batch_size'])
    # plt.yticks(ticks=df['initializer'].cat.codes.unique(), labels=df['initializer'])
    plt.show()
    # for i in range(data.shape[0]):
    #     optim_index = i


def plot_2d_hist_with_colorbar_train_batch_size_vs_fit_time(df, data_set, process):
    # Reference: https://stackoverflow.com/a/32186074/3429090
    fig = plt.figure()
    plt.suptitle('%s %s Set' % (data_set, process))
    plt.title('Training Batch Size vs. Fit Time')
    plt.xlabel('Training Batch Size')
    plt.ylabel('Fit Time (minutes)')
    fit_time_in_min = df['fit_time_sec'].apply(lambda x: x / 60)
    cm = plt.cm.get_cmap('viridis')
    sc = plt.scatter(df['train_batch_size'], fit_time_in_min, c=df['best_epoch_acc'], vmin=0.0, vmax=max(df['best_epoch_acc']), cmap=cm, s=100, alpha=0.5)
    plt.xticks(ticks=[20, 60, 100, 1000], labels=['20', '60', '100', '1000'])
    clb = plt.colorbar(sc)
    # clb.ax.set_ylim(0, 100)

    clb_title_font_dict = {
        'fontsize':'small', 'fontweight' : matplotlib.rcParams['axes.titleweight'],
        'verticalalignment': 'baseline', 'horizontalalignment': 'center'
    }
    clb.ax.set_title('Best Epoch Accuracy (Validation Set)', fontdict=clb_title_font_dict)
    clb.ax.set_yticklabels(np.arange(0, 60, 10.0))
    plt.show()
    return


def plot_2d_hist_with_colorbar_and_splines_train_batch_size_vs_fit_time(df, data_set, process):
    # Same figure with spines in x-axis:
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, sharey=True, facecolor='w')
    plt.title('Training Batch Size vs. Fit Time (GoingDeeper Dataset)')
    cm = plt.cm.get_cmap('viridis')
    fit_time_in_min = df['fit_time_sec'].apply(lambda x: x / 60)
    # Plot same data on both axes:
    ax.scatter(df['train_batch_size'], fit_time_in_min, c=df['best_epoch_acc'], vmin=0.0, vmax=1.0, cmap=cm)
    ax2.scatter(df['train_batch_size'], fit_time_in_min, c=df['best_epoch_acc'], vmin=0.0, vmax=1.0, cmap=cm)
    ax.set_xticks([20, 60, 100])
    ax.set_xlim(0, 110)
    # ax.set_xticklabels([20, 60, 100])
    ax2.set_xlim(900, 1100)
    # Hide spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.yaxis.tick_left()
    ax2.yaxis.tick_right()

    d = .015
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)

    ax.set_xlabel('Training Batch Size')
    ax.set_ylabel('Fit Time (minutes)')

    # ax3.set_visible(False)
    # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
    # plt.colorbar(fig, ax=ax3)

    # scalar_mappable = matplotlib.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=0, vmax=1))
    # scalar_mappable._A = df['best_epoch_acc']
    # fig.colorbar(scalar_mappable, cax=ax3, orientation='vertical')
    plt.show()


def main():
    datasets = ['BOONE', 'GoingDeeper']
    processes = ['Training', 'Validation']

    # Change these variables for different dataset visualizations:
    __path = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\Boone\\gs_val_hyperparams.pkl'
    dataset = datasets[0]
    process = processes[1]

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

    # Training Batch Size vs. Best Performing Epoch Acc (2D Histogram)
    plot_2d_hist_training_batch_size_vs_best_performing_epoch_acc(df=df, data_set=dataset, process=process)

    # Training Batch Size vs. Fit Time (2D Histogram with Colorbar):
    plot_2d_hist_with_colorbar_train_batch_size_vs_fit_time(df=df, data_set=dataset, process=process)

    # Training Batch Size vs. Fit Time With Spline (2D Histogram with Colorbar and Spline):
    plot_2d_hist_with_colorbar_and_splines_train_batch_size_vs_fit_time(df=df, data_set=dataset, process=process)

    # Training Batch Size vs. Fit Time (Bar Chart)
    # plot_bar_chart_train_batch_size_vs_train_time(df=df)

    # Training Batch Size vs. Fit Time (Box Plot)
    plot_boxplot_train_batch_size_vs_train_time(df=df, data_set=dataset, process=process)

    # Accuracy Metrics in General:
    # plot_eval_metrics(df=df)


if __name__ == '__main__':
    main()
