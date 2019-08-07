"""
GridSearchCVSaveRestore.py
Similar to the Sklearn native grid search, but is capable of being interrupted and restarted with save and restore
    functionality. This class is heavily based off of:
    * https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    * https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/model_selection/_search.py#L829
    * https://scikit-learn.org/stable/modules/grid_search.html#grid-search
"""

from sklearn.base import BaseEstimator, is_classifier, clone
# from sklearn.model_selection._search import MetaEstimatorMixin, ABCMeta
from sklearn.model_selection import GridSearchCV, ShuffleSplit


class CrossValidationSplitter(ShuffleSplit):
    """
    CrossValidationSplitter: Custom splitter to yield the same training and testing indices (no K-folds).
        * See the following for more information on custom cross validation via iterator:
            https://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics

    """
    def __init__(self, n_splits, test_size=None, train_size=None, random_state=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
        split: yields the indices corresponding to the training data and testing data.
        :param X:
        :param y:
        :param groups:
        :return:
        """
        yield([i for i in range(self.train_size)], [j for j in range(self.train_size, self.train_size + self.test_size)])


class GridSearchCVSaveRestore:

    def __init__(self, param_grid, scoring, cv, refit, verbose, error_score, return_train_score=False):
        """

        :param param_grid: Dictionary with parameters names (string) as keys and lists of parameter settings to try
            as values.
        :param scoring: string, callable, list/tuple, dict or None, default: None
            A single string (see :ref:`scoring_parameter`) or a callable (see :ref:`scoring`) to evaluate the
            predictions on the test set. For evaluating multiple metrics, either give a list of (unique) strings or a
            dict with names as keys and callables as values.

            NOTE that when using custom scorers, each scorer should return a single value. Metric functions returning a
            list/array of values can be wrapped into multiple scorers that return one value each.

            See :ref:`multimetric_grid_search` for an example.

            If None, the estimator's score method is used.

        :param cv: int, cross-validation generator or an iterable. Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
                - An iterable yielding (train, test) splits as arrays of indices.
        :param refit:  Refit an estimator using the best found parameters on the whole dataset. When there are
            considerations other than maximum score in choosing a best estimator, ``refit`` can be set to a function
            which returns the selected ``best_index_`` given ``cv_results_``. The refitted estimator is made available
            at the ``best_estimator_`` attribute and permits using ``predict`` directly on this ``GridSearchCV`` instance.
        :param verbose: integer
            Controls the verbosity: the higher, the more messages.
        :param error_score: 'raise' or numeric
            Value to assign to the score if an error occurs in estimator fitting. If set to 'raise', the error is raised.
            If a numeric value is given, FitFailedWarning is raised. This parameter does not affect the refit
            step, which will always raise the error. Default is 'raise' but from version 0.22 it will change to np.nan.
        :param return_train_score: boolean, default=False
            If ``False``, the ``cv_results_`` attribute will not include training scores. Computing training scores is
            used to get insights on how different parameter settings impact the overfitting/underfitting trade-off.
            However computing the scores on the training set can be computationally expensive and is not strictly
            required to select the parameters that yield the best generalization performance.
        """
        raise NotImplementedError

    def _check_is_fitted(self, method):
        """
        _check_is_fitted:
        :param method:
        :return:
        """
        raise NotImplementedError

    def score(self, X, y=None):
        raise NotImplementedError

    def fit(self, X, y=None, groups=None, **fit_params):
        """
        Run fit with all sets of parameters.
        :param X: array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and n_features is the number of features.
        :param y: array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression; None for unsupervised learning.
        :param groups: array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into train/test set. Only used in conjunction
            with a "Group" `cv` instance (e.g., `GroupKFold`).
        :param fit_params: dict of string -> object
            Parameters passed to the ```fit``` method of the estimator.
        :return:
        """
        raise NotImplementedError

    def save_grid_search(self, freq):
        """
        save_grid_search: Saves the GridSearchCV by serializing and writing the cv_results to disk. The cv_results will
            be saved to the hard drive every `freq` times a model has been trained. In this manner, the
            ```restore_grid_search``` method is able to restore the search from the last TensorFlow model to generate
            a readable final serialized model.
        :param freq: How many models should be sequentially fit and evaluated via the ```score``` method (during the
            grid search) before performing a save operation.
        :return:
        """
        raise NotImplementedError

    def restore_grid_search(self):
        """
        restore_grid_search: Restores the GridSearchCV using the data serialized by the ```save_grid_search``` method.
            This method will restore the cv_results of a prior interrupted grid search. Training of the last model
            will be resumed via TensorFlow from the checkpoint file.
        :return:
        """
        raise NotImplementedError
