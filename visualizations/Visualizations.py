import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from frameworks.DataAcquisition.BottleneckExecutor import BottleneckExecutor
from matplotlib.cm import ScalarMappable


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
    sc.set_clim([min(df['best_epoch_acc']), max(df['best_epoch_acc'])])
    plt.xticks(ticks=[20, 60, 100, 1000], labels=['20', '60', '100', '1000'])

    # Custom colorbar axes with correct tick locations:
    # cax = fig.add_axes([0.92, 0.125, 0.04, 0.72])   # Dimensions of the nex axis [left, bottom, width, height]
    # clb_ticks = np.arange(0, 110, 10)
    # clb = matplotlib.colorbar.ColorbarBase(cax, cmap=cm, norm=plt.Normalize(clb_ticks[0], clb_ticks[-1]))
    # clb.set_ticks(ticks=clb_ticks)
    # clb.ax.set_yticklabels(clb_ticks)

    # Better looking colorbar without correct ticks:
    clb = plt.colorbar(sc, ticks=np.arange(0.0, 1.1, .1))
    clb.ax.set_yticklabels(np.arange(0, 110, 10.0))
    clb_title_font_dict = {
        'fontsize':'small', 'fontweight' : matplotlib.rcParams['axes.titleweight'],
        'verticalalignment': 'baseline', 'horizontalalignment': 'center'
    }
    clb.ax.set_title('Best Epoch Accuracy', fontdict=clb_title_font_dict)

    # clb = plt.colorbar(sc, ticks=np.arange(0.1, 1.1, .1))
    # clb_ticks = np.arange(0.1, 1.1, .1)
    # clb = plt.colorbar(sc, ticks=np.arange(min(df['best_epoch_acc']), max(df['best_epoch_acc']), 10.0))
    # clb = plt.colorbar(sc, shrink=1, aspect=12, ticks=np.arange(0, 110, 10))
    # clb = plt.colorbar(sc)
    # clb = plt.colorbar(sc, shrink=1, aspect=12)
    # clb.ax.set_ylim(0, 100)
    # clb.set_clim(0.0, 100)
    # clb.set_ticks(np.arange(0, 110, 10.0))
    # clb.ax.set_yticklabels(np.arange(0, 110, 10.0))
    # clb.set_clim(0, 100)
    # sc.set_clim(0, 100)
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


def plot_boxplot_per_class_top_one_acc(top_1_acc_by_class_df, dataset='BOONE', process='Validation'):
    print(top_1_acc_by_class_df.columns)
    sorted_df = top_1_acc_by_class_df.sort_values('top_1_acc', ascending=True)
    plot = sorted_df.plot(x='class', y='top_1_acc', kind='barh', grid=False, fontsize=6)
    # Remove legend:
    plot.get_legend().remove()
    plt.ylabel('Species/Scientific Name')
    plt.xticks(np.arange(0, 110, 10))
    plt.xlabel('Top-1 Accuracy (Percent)')
    plt.title('Winning Model: Top-1 Accuracy by Class')
    plt.suptitle('%s %s Set' % (dataset, process))
    if dataset == 'GoingDeeper':
        plt.yticks([])
    plt.show()


def plot_boxplot_per_class_top_five_acc(top_5_acc_by_class_df, dataset='BOONE', process='Validation'):
    print(top_5_acc_by_class_df.columns)
    sorted_df = top_5_acc_by_class_df.sort_values('top_5_acc', ascending=True)
    plot = sorted_df.plot(x='class', y='top_5_acc', kind='barh', grid=False, fontsize=6)
    # Remove legend:
    plot.get_legend().remove()
    plt.ylabel('Species/Scientific Name')
    plt.xticks(np.arange(0, 110, 10))
    plt.xlabel('Top-5 Accuracy (Percent)')
    plt.title('Winning Model: Top-5 Accuracy by Class')
    plt.suptitle('%s %s Set' % (dataset, process))
    if dataset == 'GoingDeeper':
        plt.yticks([])
    plt.show()


def plot_boxplot_per_class_top_one_acc_aggregated(top_1_acc_by_class_df, dataset='BOONE', process='Validation'):
    top_1_acc_by_class_df_local = top_1_acc_by_class_df[top_1_acc_by_class_df['top_1_acc'] != 100]
    num_classes_with_perfect_acc = top_1_acc_by_class_df[top_1_acc_by_class_df['top_1_acc'] == 100].shape[0]
    print('Omitting %d classes with perfect top-1 accuracy (100%%)' % num_classes_with_perfect_acc)
    print('There are %d classes without perfect top-1 accuracy remaining.' % top_1_acc_by_class_df_local.shape[0])
    sorted_df = top_1_acc_by_class_df_local.sort_values('top_1_acc', ascending=True)
    plot = sorted_df.plot(x='class', y='top_1_acc', kind='barh', grid=False, fontsize=10)
    # Remove legend:
    plot.get_legend().remove()
    plt.ylabel('Species/Scientific Name')
    plt.xlabel('Top-1 Accuracy (Percent)')
    plt.xticks(np.arange(0, 110, 10.0))
    plt.title('Winning Model: Top-1 Accuracy by Class (Excluding 100% Accurate)')
    plt.suptitle('%s %s Set' % (dataset, process))
    if dataset == 'GoingDeeper':
        plt.yticks([])
    plt.show()


def plot_boxplot_per_class_top_five_acc_aggregated(top_5_acc_by_class_df, dataset='BOONE', process='Validation'):
    top_5_acc_by_class_df_local = top_5_acc_by_class_df[top_5_acc_by_class_df['top_5_acc'] != 100]
    sorted_df = top_5_acc_by_class_df_local.sort_values('top_5_acc', ascending=True)
    plot = sorted_df.plot(x='class', y='top_5_acc', kind='barh', grid=False, fontsize=10)
    # Remove legend:
    plot.get_legend().remove()
    plt.ylabel('Species/Scientific Name')
    plt.xlabel('Top-5 Accuracy (Percent)')
    plt.xticks(np.arange(0, 110, 10.0))
    plt.title('Winning Model: Top-5 Accuracy by Class (Excluding 100% Accurate)')
    plt.suptitle('%s %s Set' % (dataset, process))
    if dataset == 'GoingDeeper':
        plt.yticks([])
    plt.show()


def plot_2d_histogram_per_class_top_one_acc(top_1_acc_by_class_df, dataset='BOONE', process='Validation'):
    top_1_acc_by_class_df.plot(x='class', y='top_1_acc', kind='hist', bins=np.arange(0, 110, 10.0))
    plt.show()


def plot_per_class_top_one_acc_vs_number_of_samples_aggregated(process_top_1_acc_by_class_df, process_bottlenecks_df,
                                                               training_top_1_acc_by_class_df=None,
                                                               training_bottlenecks_df=None, dataset='BOONE',
                                                               process='Validation', preceding_process='Training'):
    # https://stackoverflow.com/questions/51204505/python-barplot-with-colorbar
    fig, ax = plt.subplots()
    print('Subsetting top-1 acc dataframe by removing entries with perfect accuracy...')
    process_top_1_acc_by_class_df = process_top_1_acc_by_class_df[process_top_1_acc_by_class_df['top_1_acc'] != 100]
    print('Merging source data frames...')
    joined_df = pd.merge(process_top_1_acc_by_class_df, process_bottlenecks_df, how='inner', sort=False)
    # Sort ascending:
    # joined_df = joined_df.sort_values('top_1_acc', ascending=True)
    # Remove 100 percent accuracy:

    print('Appending class sample counts...')
    joined_df['num_class_samples'] = joined_df.groupby(['class'])['top_1_acc'].transform('count')
    # joined_df = joined_df.sort_values(by='num_class_samples', ascending=False)
    # Remove the now extraneous rows:
    print('Removing extraneous columns...')
    joined_df = joined_df.drop(['bottleneck', 'path'], axis=1)
    print('Dropping duplicate rows...')
    joined_df = joined_df.drop_duplicates(subset=['class', 'top_1_acc', 'num_class_samples'])
    # Normalize:
    print('Normalizing class sample counts as color data...')
    joined_df['data_color'] = joined_df['num_class_samples'].apply(lambda x: x / max(joined_df['num_class_samples'].values))
    # Re-sort:
    print('Sorting result by top-1-accuracy and number of class samples in descending order...')
    joined_df = joined_df.sort_values(['top_1_acc', 'num_class_samples'], ascending=False)
    print('num_samples_per_class: %s' % joined_df['num_class_samples'].values)
    print('data colors: %s' % joined_df['data_color'].values)

    cmap = plt.cm.get_cmap('GnBu')
    colors = cmap(joined_df['data_color'].values)
    rects = ax.barh(joined_df['class'], joined_df['top_1_acc'], color=colors)

    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max(joined_df['num_class_samples'])))
    sm.set_clim(vmin=0, vmax=max(joined_df['num_class_samples']))

    cbar = plt.colorbar(sm, drawedges=False)
    cbar.set_label('Number of Class Samples (%s Set)' % process, rotation=270, labelpad=25)

    plt.ylabel('Species/Scientific Name')
    plt.xlabel('Top-1 Accuracy')
    plt.title('Winning Model: Top-1 Accuracy by Class (Excluding 100% Accurate)')
    plt.suptitle('%s %s Set' % (dataset, process))
    # Invert the y axis:
    ax.invert_yaxis()

    if dataset == 'GoingDeeper':
        ax.set_yticks([])

    plt.show()

    if training_top_1_acc_by_class_df is not None:
        # Plot again with the colors done by number of training samples instead of validation samples:
        # plt.clf()
        fig, ax = plt.subplots()
        joined_training_df = pd.merge(training_top_1_acc_by_class_df, training_bottlenecks_df, how='inner', sort=False)
        # Remove classes which obtained 100% accuracy on the validation set:
        # this pulls out only the classes that remain in the validation dataframe after dropping those with 100% accuracy:
        joined_training_df_same_class_subset_as_preceding_process = joined_training_df[joined_training_df['class'].isin(joined_df['class'])].dropna()
        # Now we find out the number of training instances belonging to each of those classes:
        joined_training_df_same_class_subset_as_preceding_process['num_class_samples'] = joined_training_df_same_class_subset_as_preceding_process.groupby(['class'])['top_1_acc'].transform('count')
        joined_training_df_same_class_subset_as_preceding_process = joined_training_df_same_class_subset_as_preceding_process.sort_values(by='num_class_samples', ascending=False)
         # Remove extraneous rows:
        joined_training_df_same_class_subset_as_preceding_process = joined_training_df_same_class_subset_as_preceding_process.drop(['bottleneck', 'path'], axis=1)
        joined_training_df_same_class_subset_as_preceding_process = joined_training_df_same_class_subset_as_preceding_process.drop_duplicates(subset=['class', 'top_1_acc', 'num_class_samples'])
        # Normalize:
        joined_training_df_same_class_subset_as_preceding_process['data_color'] = joined_training_df_same_class_subset_as_preceding_process['num_class_samples'].apply(lambda x: x / max(joined_training_df_same_class_subset_as_preceding_process['num_class_samples']))
        # Re-sort:
        joined_training_df_same_class_subset_as_preceding_process = joined_training_df_same_class_subset_as_preceding_process.sort_values(['top_1_acc', 'num_class_samples'], ascending=False)
        print('num_training_samples_per_class: %s' % joined_training_df_same_class_subset_as_preceding_process['num_class_samples'].values)
        print('training data colors: %s' % joined_training_df_same_class_subset_as_preceding_process['data_color'].values)

        training_cmap = plt.get_cmap('GnBu')
        training_colors = cmap(joined_training_df_same_class_subset_as_preceding_process['data_color'].values)
        training_rects = ax.barh(joined_df['class'], joined_df['top_1_acc'], color=training_colors)
        sm = ScalarMappable(cmap=training_cmap, norm=plt.Normalize(0, max(joined_training_df_same_class_subset_as_preceding_process['num_class_samples'])))
        sm.set_clim(vmin=0, vmax=max(joined_training_df_same_class_subset_as_preceding_process['num_class_samples']))

        cbar = plt.colorbar(sm, drawedges=False)
        cbar.set_label('Number of Class Samples (%s Set)' % preceding_process, rotation=270, labelpad=25)
        plt.ylabel('Species/Scientific Name')
        plt.xlabel('Top-1 Accuracy')
        plt.title('Winning Model: Top-1 Accuracy by Class (Excluding 100% Accurate)')
        plt.suptitle('%s %s Set' % (dataset, process))
        # Invert the y-axis
        ax.invert_yaxis()

        if dataset == 'GoingDeeper':
            ax.set_yticks([])
        plt.show()


def plot_per_class_top_five_acc_vs_number_of_samples_aggregated(process_top_5_acc_by_class_df, process_bottlenecks_df, training_top_5_acc_by_class_df=None, training_bottlenecks_df=None, dataset='BOONE', process='Validation', preceding_process='Training'):
    # https://stackoverflow.com/questions/51204505/python-barplot-with-colorbar
    fig, ax = plt.subplots()
    # Remove 100 percent accuracy:
    process_top_5_acc_by_class_df = process_top_5_acc_by_class_df[process_top_5_acc_by_class_df['top_5_acc'] != 100]
    joined_df = pd.merge(process_top_5_acc_by_class_df, process_bottlenecks_df, how='inner', sort=False)
    joined_df['num_class_samples'] = joined_df.groupby(['class'])['top_5_acc'].transform('count')
    # Remove extra rows:
    joined_df = joined_df.drop(['bottleneck', 'path'], axis=1)
    joined_df = joined_df.drop_duplicates(subset=['top_5_acc', 'num_class_samples'])
    # Normalize for colors:
    joined_df['data_color'] = joined_df['num_class_samples'].apply(lambda x: x / max(joined_df['num_class_samples']))
    # Re-sort
    joined_df = joined_df.sort_values(['top_5_acc', 'num_class_samples'], ascending=False)
    print('num_samples_per_class: %s' % joined_df['num_class_samples'].values)
    print('data colors: %s' % joined_df['data_color'].values)

    cmap = plt.cm.get_cmap('GnBu')
    colors = cmap(joined_df['data_color'].values)
    rects = ax.barh(joined_df['class'], joined_df['top_5_acc'], color=colors)

    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max(joined_df['num_class_samples'])))
    sm.set_clim(vmin=min(joined_df['num_class_samples']), vmax=max(joined_df['num_class_samples']))

    cbar = plt.colorbar(sm, drawedges=False)
    cbar.set_label('Number of Class Samples (%s Set)' % process, rotation=270, labelpad=25)

    plt.ylabel('Species/Scientific Name')
    plt.xlabel('Top-5 Accuracy')
    plt.title('Winning Model: Top-5 Accuracy by Class (Excluding 100% Accurate)')
    plt.suptitle('%s %s Set' % (dataset, process))

    # Invert y-axis
    ax.invert_yaxis()

    if dataset == 'GoingDeeper':
        ax.set_yticks([])

    plt.show()

    if training_top_5_acc_by_class_df is not None:
        fig, ax = plt.subplots()
        joined_training_df = pd.merge(training_top_5_acc_by_class_df, training_bottlenecks_df, how='inner', sort=False)
        # Remove classes which obtained 100% accuracy on the validation set:
        joined_training_df_subset = joined_training_df[joined_training_df['class'].isin(joined_df['class'])].dropna()
        # Calculate the number of training instances belonging to each of the classes:
        joined_training_df_subset['num_class_samples'] = joined_training_df_subset.groupby(['class'])['top_5_acc'].transform('count')
        # Drop extraneous rows:
        joined_training_df_subset = joined_training_df_subset.drop(['bottleneck', 'path'], axis=1)
        joined_training_df_subset = joined_training_df_subset.drop_duplicates(subset=['class', 'top_5_acc', 'num_class_samples'])
        # Normalize:
        joined_training_df_subset['data_color'] = joined_training_df_subset['num_class_samples'].apply(lambda x: x / max(joined_training_df_subset['num_class_samples']))
        # Re sort:
        joined_training_df_subset = joined_training_df_subset.sort_values(['top_5_acc', 'num_class_samples'], ascending=False)
        print('num_training_samples_per_class: %s' % joined_training_df_subset['num_class_samples'].values)
        print('training data colors: %s' % joined_training_df_subset['data_color'].values)

        training_cmap = plt.get_cmap('GnBu')
        training_colors = cmap(joined_training_df_subset['data_color'].values)
        rects = ax.barh(joined_df['class'], joined_df['top_5_acc'], color=training_colors)

        sm = ScalarMappable(cmap=training_cmap, norm=plt.Normalize(0, max(joined_training_df_subset['num_class_samples'])))
        sm.set_clim(vmin=0, vmax=max(joined_training_df_subset['num_class_samples']))

        cbar = plt.colorbar(sm, drawedges=False)
        cbar.set_label('Number of Class Samples (%s Set)' % preceding_process, rotation=270, labelpad=25)
        plt.ylabel('Species/Scientific Name')
        plt.xlabel('Top-5 Accuracy')
        plt.title('Winning Model: Top-5 Accuracy by Class (Excluding 100% Accurate)')
        plt.suptitle('%s %s Set' % (dataset, process))
        # Invert the y-axis so it maches the order of the source dataframe:
        ax.invert_yaxis()

        if dataset == 'GoingDeeper':
            ax.set_yticks([])
        plt.show()


def plot_boxplot_hyperparameters_vs_training_time(gs_hyperparams_df, dataset='BOONE', process='Validation'):
    gs_hyperparams_df['fit_time_min'] = gs_hyperparams_df['fit_time_sec'].apply(lambda x: x / 60)

    plot = gs_hyperparams_df.boxplot(column='fit_time_min', by='train_batch_size')
    plt.xlabel('Training Batch Size')
    plt.ylabel('Fit Time (Minutes)')
    plt.title('Training Time vs. Training Batch Size')
    plt.suptitle("%s %s Set" % (dataset, process))
    plt.show()

    plot = gs_hyperparams_df.boxplot(column='fit_time_min', by='initializer')
    plt.xlabel('Initializer')
    plt.ylabel('Fit Time (Minutes)')
    plt.title('Initializer vs. Training Time')
    plt.suptitle("%s %s Set" % (dataset, process))
    plt.show()

    plot = gs_hyperparams_df.boxplot(column='fit_time_min', by='optimizer')
    plt.xlabel('Optimizer')
    plt.ylabel('Fit Time (Minutes)')
    plt.title('Optimizer vs. Training Time')
    plt.suptitle("%s %s Set" % (dataset, process))
    plt.show()

    plot = gs_hyperparams_df.boxplot(column='fit_time_min', by='activation')
    plt.xlabel('Activation')
    plt.ylabel('Fit Time (Minutes)')
    plt.title('Activation vs. Training Time')
    plt.suptitle("%s %s Set" % (dataset, process))
    plt.show()


def plot_bar_chart_class_count_by_top_one_acc(top_1_acc_by_class_df, bottlenecks_df, dataset, process):
    joined_df = pd.merge(top_1_acc_by_class_df, bottlenecks_df, how='outer', sort=False)
    joined_df['class'].value_counts().plot(kind='bar')
    raise NotImplementedError


def plot_scatter_per_class_top_one_acc(top_1_acc_by_class_df, top_5_acc_by_class_df, bottlenecks_df, dataset, process):
    # fig, ax = plt.subplots()
    # ax.plot('class')
    joined_df = pd.merge(top_1_acc_by_class_df, bottlenecks_df, how='outer', sort=False)
    joined_df = pd.merge(joined_df, top_5_acc_by_class_df, how='outer', sort=False)
    # top_1_acc_by_class.plot(kind='scatter', x='class', y='top_1_acc')
    threshold = 95.00
    df_subset = joined_df[joined_df['top_1_acc'] >= threshold]
    num_initial_classes = len(joined_df['class'].unique())
    num_initial_samples = joined_df.shape[0]
    threshold_classes = len(df_subset['class'].unique())
    num_threshold_samples = df_subset.shape[0]
    percent_samples_classified_with_threshold = (num_threshold_samples * 100) / num_initial_samples
    print('Only issuing predictions where the top-1 accuracy is at or above the threshold of %.2f%%, then %.2f%% of samples can be classified automatically.' % (threshold, percent_samples_classified_with_threshold))

    # https://www.idigbio.org/content/simultaneous-transcription-blitzes-success
    # On-site participants averaged 3.48 minutes per Notes from Nature specimen
    # Workforce Efficient Consensus in Crowd Sourced Transcription of Biocollections:
    # Average of 9.18 minutes for transcriptions scoring above 0.7

    print('It is estimated that this will take %s seconds of time. Thereby saving %s seconds of transcription efforts.' % ('Unknown', 'Unknown'))
    print('The remaining percentage must be delegated to top-5 accuracy.')

    df_subset = joined_df[joined_df['top_1_acc'] < threshold]
    df_top_5_acc_subset = df_subset[df_subset['top_5_acc'] >= threshold]
    num_initial_classes = len(df_subset['class'].unique())
    num_initial_samples = df_subset.shape[0]
    threshold_classes = len(df_top_5_acc_subset['class'].unique())
    num_threshold_samples = df_top_5_acc_subset.shape[0]
    percent_samples_classified_with_threshold = (num_threshold_samples * 100) / num_initial_samples
    print('Of the remaining samples...')
    print('Only issuing predictions where the top-5 accuracy is at or above the threshold of %.2f%%, then %.2f%% of samples can be classified semi-automatically.' % (threshold, percent_samples_classified_with_threshold))
    plt.show()


def main(run_config):

    with open(run_config['top_1_per_class_acc_json_path'], 'r') as fp:
        top_1_acc_by_class_df = pd.read_json(fp, orient='index')

    with open(run_config['top_1_per_class_acc_json_path'].replace('1', '5'), 'r') as fp:
        top_5_acc_by_class_df = pd.read_json(fp, orient='index')

    gs_hyperparams_df = pd.read_pickle(run_config['hyperparam_df_path'])
    optimizers = gs_hyperparams_df.optimizer.unique()
    num_optimizers = len(optimizers)
    print('Optimizers: %s' % optimizers.categories)

    activations = gs_hyperparams_df.activation.unique()
    num_activations = len(activations)
    print('Activations: %s' % activations.categories)

    train_batch_sizes = gs_hyperparams_df.train_batch_size.unique()
    num_train_batch_sizes = len(train_batch_sizes)
    print('Train Batch Sizes: %s' % train_batch_sizes)

    initializers = gs_hyperparams_df.initializer.unique()
    num_initializers = len(initializers)
    print('Initializers: %s' % initializers.categories)

    heatmap_dims = ((num_activations * num_optimizers), (num_initializers * num_train_batch_sizes))
    data = np.zeros(heatmap_dims)
    print('HeatMap Dimensions: %s' %(data.shape,))

    print('Columns: %s\n' % gs_hyperparams_df.columns.values)

    bottleneck_executor = BottleneckExecutor(
        image_dir=run_config['image_dir'],
        logging_dir=run_config['logging_dir'],
        tfhub_module_url='https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1',
        compressed_bottleneck_file_path=run_config['bottleneck_path']
    )
    all_bottlenecks = bottleneck_executor.get_bottlenecks()
    class_labels = list(all_bottlenecks['class'].unique())
    train_bottlenecks, val_bottlenecks, test_bottlenecks = bottleneck_executor.get_partitioned_bottlenecks()
    if run_config['process'].lower() == 'training':
        training_bottlenecks_df = None
        training_top_1_acc_by_class_df = None
        training_top_5_acc_by_class_df = None
        bottlenecks_df = train_bottlenecks
    elif run_config['process'].lower() == 'validation':
        training_bottlenecks_df = train_bottlenecks
        training_top_1_acc_by_class_df = None
        training_top_5_acc_by_class_df = None
        proceeding_run_config = run_configs[run_config['dataset']]['train']
        with open(proceeding_run_config['top_1_per_class_acc_json_path'], 'r') as fp:
            training_top_1_acc_by_class_df = pd.read_json(fp, orient='index')
        with open(proceeding_run_config['top_5_per_class_acc_json_path'], 'r') as fp:
            training_top_5_acc_by_class_df = pd.read_json(fp, orient='index')
        bottlenecks_df = val_bottlenecks
    elif run_config['process'].lower() == 'testing':
        raise NotImplementedError("training_bottlenecks_df = train_bottlenecks.join(val_bottlenecks)")
        bottlenecks_df = test_bottlenecks
    else:
        raise NotImplementedError
    # bottlenecks_df = None

    # Training Batch Size vs. Best Performing Epoch Acc (2D Histogram)
    plot_2d_hist_training_batch_size_vs_best_performing_epoch_acc(df=gs_hyperparams_df, data_set=run_config['dataset'], process=run_config['process'])

    # Training Batch Size vs. Fit Time (2D Histogram with Colorbar):
    plot_2d_hist_with_colorbar_train_batch_size_vs_fit_time(df=gs_hyperparams_df, data_set=run_config['dataset'], process=run_config['process'])

    # Training Batch Size vs. Fit Time (Bar Chart)
    plot_bar_chart_train_batch_size_vs_train_time(df=gs_hyperparams_df)

    # per-class top-1 accuracy (Box Plot):
    plot_boxplot_per_class_top_one_acc(top_1_acc_by_class_df=top_1_acc_by_class_df, dataset=run_config['dataset'], process=run_config['process'])

    # per-class top-5 accuracy (Horizontal Bar Plot):
    plot_boxplot_per_class_top_five_acc(top_5_acc_by_class_df, dataset=run_config['dataset'], process=run_config['process'])

    # per-class top-1 accuracy (Box Plot with Aggregation):
    plot_boxplot_per_class_top_one_acc_aggregated(top_1_acc_by_class_df, dataset=run_config['dataset'], process=run_config['process'])

    # per-class top-5 accuracy (Box Plot with Aggregation):
    plot_boxplot_per_class_top_five_acc_aggregated(top_5_acc_by_class_df, dataset=run_config['dataset'], process=run_config['process'])

    # Plot number of samples per-class (colorbar on existing) vs class's top-1 acc
    if run_config['process'].lower() == 'training':
        plot_per_class_top_one_acc_vs_number_of_samples_aggregated(top_1_acc_by_class_df, bottlenecks_df, training_top_1_acc_by_class_df=None, training_bottlenecks_df=None, dataset=run_config['dataset'], process=run_config['process'])
    elif run_config['process'].lower() == 'validation':
        plot_per_class_top_one_acc_vs_number_of_samples_aggregated(top_1_acc_by_class_df, bottlenecks_df, training_top_1_acc_by_class_df=training_top_1_acc_by_class_df, training_bottlenecks_df=training_bottlenecks_df, dataset=run_config['dataset'], process=run_config['process'])
    else:
        raise NotImplementedError("Need to distinguish testing process.")

    # Plot number of samples per-class (colorbar on existing) vs class's top-5 acc
    if run_config['process'].lower() == 'training':
        plot_per_class_top_five_acc_vs_number_of_samples_aggregated(top_5_acc_by_class_df, bottlenecks_df, training_top_5_acc_by_class_df=None, training_bottlenecks_df=None, dataset=run_config['dataset'], process=run_config['process'])
    elif run_config['process'].lower() == 'validation':
        plot_per_class_top_five_acc_vs_number_of_samples_aggregated(top_5_acc_by_class_df, bottlenecks_df, training_top_5_acc_by_class_df=training_top_5_acc_by_class_df, training_bottlenecks_df=training_bottlenecks_df, dataset=run_config['dataset'], process=run_config['process'])
    else:
        raise NotImplementedError("Need to distinguish testing process.")

    # Plot each hyperparameter on y-axis and then training time on the left-axis.
    plot_boxplot_hyperparameters_vs_training_time(gs_hyperparams_df, dataset=run_config['dataset'], process=run_config['process'])

    # per-class top-1 accuracy (scatter):
    plot_scatter_per_class_top_one_acc(top_1_acc_by_class_df, top_5_acc_by_class_df, bottlenecks_df, dataset=run_config['dataset'], process=run_config['process'])


if __name__ == '__main__':
    run_configs = {
        'DEBUG': {
            'dataset': 'DEBUG',
            'image_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\GoingDeeper\\images',
            'bottleneck_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\GoingDeeper\\images\\bottlenecks.pkl',
            'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeepKeras\\frameworks\\DataAcquisition\\CleaningResults\\DEBUG'
        },
        'BOONE': {
            'val': {
                'dataset': 'BOONE',
                'process': 'Validation',
                'image_dir': 'D:\\data\\BOON\\images',
                'bottleneck_path': 'D:\\data\\BOON\\bottlenecks.pkl',
                'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeepKeras\\frameworks\\DataAcquisition\\CleaningResults\\BOON',
                'hyperparam_df_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\Boone\\gs_val_hyperparams.pkl',
                'top_1_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\Boone\\top_1_accuracies_by_class_val_set.json',
                'top_5_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\Boone\\top_5_accuracies_by_class_val_set.json',
                'saved_model_path': 'D:\\data\\BOON\\training summaries\\8-16-2019\\gs_winner\\train'
            },
            'train': {
                'dataset': 'BOONE',
                'process': 'Training',
                'image_dir': 'D:\\data\\BOON\\images',
                'bottleneck_path': 'D:\\data\\BOON\\bottlenecks.pkl',
                'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeepKeras\\frameworks\\DataAcquisition\\CleaningResults\\BOON',
                'hyperparam_df_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\Boone\\gs_train_hyperparams.pkl',
                'top_1_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\Boone\\top_1_accuracies_by_class_train_set.json',
                'top_5_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\Boone\\top_5_accuracies_by_class_train_set.json',
                'saved_model_path': 'D:\\data\\BOON\\training summaries\\8-16-2019\\gs_winner\\train'
            },
            'test':
                {
                    'dataset': 'BOONE',
                    'process': 'Testing',
                    'image_dir': 'D:\\data\\BOON\\images',
                    'bottleneck_path': 'D:\\data\\BOON\\bottlenecks.pkl',
                    'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeepKeras\\frameworks\\DataAcquisition\\CleaningResults\\BOON',
                    'hyperparam_df_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\Boone\\gs_test_hyperparams.pkl',
                    'top_1_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\Boone\\top_1_accuracies_by_class_test_set.json',
                    'top_5_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\Boone\\top_5_accuracies_by_class_test_set.json',
                    'saved_model_path': 'D:\\data\\BOON\\training summaries\\8-16-2019\\gs_winner\\train'
                }
        },
        'GoingDeeper': {
            'train': {
                'dataset': 'GoingDeeper',
                'process': 'Training',
                'image_dir': 'D:\\data\\GoingDeeperData\\images',
                'bottleneck_path': 'D:\\data\\GoingDeeperData\\bottlenecks.pkl',
                'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeepKeras\\frameworks\\DataAcquisition\\CleaningResults\\GoingDeeper',
                'saved_model_path': 'D:\\data\\GoingDeeperData\\training summaries\\10-28-2019\\gs_winner\\train',
                'hyperparam_df_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\GoingDeeper\\gs_train_hyperparams.pkl',
                'top_1_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\GoingDeeper\\top_1_accuracies_by_class_train_set.json',
                'top_5_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\GoingDeeper\\top_5_accuracies_by_class_train_set.json'
            },
            'val': {
                'dataset': 'GoingDeeper',
                'process': 'Validation',
                'image_dir': 'D:\\data\\GoingDeeperData\\images',
                'bottleneck_path': 'D:\\data\\GoingDeeperData\\bottlenecks.pkl',
                'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeepKeras\\frameworks\\DataAcquisition\\CleaningResults\\GoingDeeper',
                'saved_model_path': 'D:\\data\\GoingDeeperData\\training summaries\\10-28-2019\\gs_winner\\train',
                'hyperparam_df_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\GoingDeeper\\gs_val_hyperparams.pkl',
                'top_1_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\GoingDeeper\\top_1_accuracies_by_class_val_set.json',
                'top_5_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\GoingDeeper\\top_5_accuracies_by_class_val_set.json'
            },
            'test': {
                'dataset': 'GoingDeeper',
                'process': 'Testing',
                'image_dir': 'D:\\data\\GoingDeeperData\\images',
                'bottleneck_path': 'D:\\data\\GoingDeeperData\\bottlenecks.pkl',
                'logging_dir': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeepKeras\\frameworks\\DataAcquisition\\CleaningResults\\GoingDeeper',
                'saved_model_path': 'D:\\data\\GoingDeeperData\\training summaries\\10-28-2019\\gs_winner\\train',
                'hyperparam_df_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\GoingDeeper\\gs_test_hyperparams.pkl',
                'top_1_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\GoingDeeper\\top_1_accuracies_by_class_test_set.json',
                'top_5_per_class_acc_json_path': 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\visualizations\\GoingDeeper\\top_5_accuracies_by_class_test_set.json'
            }
        },
        'SERNEC': {}
    }
    main(run_config=run_configs['BOONE']['val'])
