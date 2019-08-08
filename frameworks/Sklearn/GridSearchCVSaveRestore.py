"""
GridSearchCVSaveRestore.py
Similar to the Sklearn native grid search, but is capable of being interrupted and restarted with save and restore
    functionality. This class is heavily based off of:
    * https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    * https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/model_selection/_search.py#L829
    * https://scikit-learn.org/stable/modules/grid_search.html#grid-search
"""
import logging
import numpy as np
from collections.abc import Mapping, Sequence, Iterable
from functools import partial, reduce
from itertools import product
import operator
from sklearn.model_selection._search import BaseSearchCV
from sklearn.base import BaseEstimator, is_classifier, clone
# from sklearn.model_selection._search import MetaEstimatorMixin, ABCMeta
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _fit_and_score
from sklearn.utils.validation import indexable
from sklearn.metrics.scorer import _check_multimetric_scoring, check_scoring
from sklearn.utils._joblib import Parallel, delayed


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


class ParameterGrid:
    """
    ParameterGrid: Grid of parameters with a discrete number of values for each. Can be used to iterate over parameter
        value combinations with the Python built-in function iter. Read more in the :ref:`User Guide <grid_search>`.

    Source: This code was copied verbatim (for educational usage) from the sklearn source code located at:
        * https://github.com/scikit-learn/scikit-learn/blob/1495f69242646d239d89a5713982946b8ffcf9d9/sklearn/model_selection/_search.py#L45
    Accordingly, the author of the rest of this software (Christopher Campell) has no claim to the code written herein.

    Parameters
    ----------
    param_grid : dict of string to sequence, or sequence of such
        The parameter grid to explore, as a dictionary mapping estimator parameters to sequences of allowed values. An
        empty dict signifies default parameters. A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense or have no effect. See the examples below.
    """

    def __init__(self, param_grid):
        if not isinstance(param_grid, (Mapping, Iterable)):
            raise TypeError('Parameter grid is not a dict or '
                            'a list ({!r})'.format(param_grid))

        if isinstance(param_grid, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            param_grid = [param_grid]

        # check if all entries are dictionaries of lists
        for grid in param_grid:
            if not isinstance(grid, dict):
                raise TypeError('Parameter grid is not a '
                                'dict ({!r})'.format(grid))
            for key in grid:
                if not isinstance(grid[key], Iterable):
                    raise TypeError('Parameter grid value is not iterable '
                                    '(key={!r}, value={!r})'
                                    .format(key, grid[key]))

        self.param_grid = param_grid

    def __iter__(self):
        """Iterate over the points in the grid.
        Returns
        -------
        params : iterator over dict of string to any
            Yields dictionaries mapping each estimator parameter to one of its
            allowed values.
        """
        for p in self.param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        """Number of points on the grid."""
        # Product function that can handle iterables (np.product can't).
        product = partial(reduce, operator.mul)
        return sum(product(len(v) for v in p.values()) if p else 1
                   for p in self.param_grid)

    def __getitem__(self, ind):
        """Get the parameters that would be ``ind``th in iteration
        Parameters
        ----------
        ind : int
            The iteration index
        Returns
        -------
        params : dict of string to any
            Equal to list(self)[ind]
        """
        # This is used to make discrete sampling without replacement memory
        # efficient.
        for sub_grid in self.param_grid:
            # XXX: could memoize information used here
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue

            # Reverse so most frequent cycling parameter comes first
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.product(sizes)

            if ind >= total:
                # Try the next grid
                ind -= total
            else:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v_list[offset]
                return out

        raise IndexError('ParameterGrid index out of range')


class GridSearchCVSaveRestore(BaseSearchCV):
    """
    GridSearchCVSaveRestore: Similar to sklearn's GridSearchCV class, except with an additional save and restore
        functionality utilized to resume a previously interrupted GridSearch. This functionality is contingent on
        TensorFlow's model serialization as performed in the base estimator (TFHClassifier.py).

    SOURCE: This code is copied almost verbatim (for educational usage only) from the sklearn source code available here:
        * https://github.com/scikit-learn/scikit-learn/blob/1495f69242646d239d89a5713982946b8ffcf9d9/sklearn/model_selection/_search.py#L829
    """

    def __init__(self, estimator, param_grid, cv_results_save_freq, scoring=None, n_jobs=None, iid='warn', refit=True, cv='warn',
                 verbose=0, pre_dispatch='2*n_jobs', error_score='raise-deprecating', return_train_score=False):
        """
        __init__: Initialization method for objects of type GridSearchCVSaveRestore.
        :param estimator:
        :param param_grid:
        :param cv_results_save_freq: How frequently (in terms of trained models) the GridSearch's c
        :param scoring:
        :param n_jobs:
        :param iid:
        :param refit:
        :param cv:
        :param verbose:
        :param pre_dispatch:
        :param error_score:
        :param return_train_score:
        """
        # Call __init__ method of sklearn's BaseSearchCV class:
        super().__init__(estimator=estimator, scoring=scoring, n_jobs=n_jobs, iid=iid,
                         refit=refit, cv=cv, verbose=verbose, pre_dispatch=pre_dispatch, error_score=error_score,
                         return_train_score=return_train_score)
        self.param_grid = param_grid
        _check_param_grid(param_grid=param_grid)

        ''' My Modifications to Class Attributes '''
        self.cv_results_save_freq = cv_results_save_freq

    def fit(self, X, y=None, groups=None, **fit_params):
        """
        fit: Run fit with all sets of parameters. Periodically serialize the ``cv_results`` dictionary after fitting
            every ``self.cv_results_save_freq`` number of models.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" `cv`
            instance (e.g., `GroupKFold`).
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator=estimator))

        scorers, self.multimetric_ = _check_multimetric_scoring(self.estimator, scoring=self.scoring)

        if self.multimetric_:
            raise NotImplementedError('Multimetric scoring is not yet implemented for overridden sequential-based fit method.')
        else:
            refit_metric = 'score'

        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(scorer=scorers, fit_params=fit_params, return_train_score=self.return_train_score,
                                    return_n_test_samples=True, return_times=True, return_parameters=False,
                                    error_score=self.error_score, verbose=self.verbose)

        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []

            def evaluate_candidate(candidate_param):
                """
                evaluate_candidate: Similar to sklearn's evaluate_candidates method (see below) but only evaluates a single
                    candidate. This is done in order to control the saving of the Grid Search's cv_results dictionary.
                :param candidate_param:
                :return:
                """
                n_candidates = 1

                if self.verbose > 0:
                    print("Fitting {0} folds for each of {1} candidates, totalling {2} fits".format(
                                      n_splits, n_candidates, n_candidates * n_splits))

                out = parallel(delayed(_fit_and_score)(clone(base_estimator),
                                                       X, y, train=train, test=test, parameters=parameters, **fit_and_score_kwargs)
                               for parameters, (train, test) in product(candidate_param, cv.split(X, y, groups)))

                if len(out) < 1:
                    raise ValueError('No fits were performed. '
                                     'Was the CV iterator empty? '
                                     'Were there no candidates?')
                elif len(out) != n_candidates * n_splits:
                    raise ValueError('cv.split and cv.get_n_splits returned '
                                     'inconsistent results. Expected {} '
                                     'splits, got {}'
                                     .format(n_splits,
                                             len(out) // n_candidates))

                all_candidate_params.extend(candidate_param)
                all_out.extend(out)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, scorers, n_splits, all_out)
                logging.warning('evaluate_candidate not finished being implemented.')
                return results

                # raise NotImplementedError('evaluate_candidate not finished being implemented.')
            self._run_search(evaluate_candidate)

    def _run_search(self, evaluate_candidate):
        """
        Repeatedly calls `evaluate_candidates` to conduct a search. This method, implemented in sub-classes (i.e. this
        class), makes it possible to customize the the scheduling of evaluations: GridSearchCV and RandomizedSearchCV
        schedule evaluations for their whole parameter search space at once but other more sequential approaches are also
        possible: for instance is possible to iteratively schedule evaluations for new regions of the parameter search
        space based on previously collected evaluation results. This makes it possible to implement Bayesian
        optimization or more generally sequential model-based optimization by deriving from the BaseSearchCV abstract
        base class.

        Parameters
        ----------
        evaluate_candidates : callable
            This callback accepts a list of candidates, where each candidate is a dict of parameter settings. It
            returns a dict of all results so far, formatted like ``cv_results_``.

        Examples
        --------
        ::
            def _run_search(self, evaluate_candidates):
                'Try C=0.1 only if C=1 is better than C=10'
                all_results = evaluate_candidates([{'C': 1}, {'C': 10}])
                score = all_results['mean_test_score']
                if score[0] < score[1]:
                    evaluate_candidates([{'C': 0.1}])
        """
        # TODO: Looks like this is intended to operate in parallel as evaluate_candidates returns all evaluations. Need to
        #   create a custom evaluate candidate function.

        #  Search all candidates in param_grid:
        #         evaluate_candidates(ParameterGrid(self.param_grid))
        param_grid = ParameterGrid(self.param_grid)

        # NOTE: this method is currently being called by fit with only a single candidate, hence the looping structure
        #   is not only irrelevant, but potentially incorrect/dangerous in the presence of parallelism.

        # for param_set in param_grid:
        #     param_set_results = evaluate_candidate(param_set)
        #     param_set_score = param_set_results['mean_test_score']
        all_results = evaluate_candidate(param_grid)

        raise NotImplementedError('_run_search not implemented in subclass. Awaiting de-parallelization of evaluate_candidates.')

# class GridSearchCVSaveRestore:
#     """
#     GridSearchCVSaveRestore: Similar to sklearn's GridSearchCV class, except with an additional save and restore
#         functionality utilized to resume a previously interrupted GridSearch. This functionality is contingent on
#         TensorFlow's model serialization as performed in the base estimator (TFHClassifier.py).
#     """
#     estimator = None
#     param_grid = None
#     scoring = None
#     cv = None
#     refit = None
#     verbose = None
#     error_score = None
#     return_train_score = None
#     _cv_results = None
#
#     def __init__(self, estimator, param_grid, cv, refit, verbose, error_score, scoring=None, return_train_score=False):
#         """
#         __init__: Initialization method for objects of type GridSearchCVSaveRestore.
#         :param estimator: estimator object (traditionally a TFHClassifier instance for this project)
#             This is assumed to implement the scikit-learn estimator interface. Either estimator needs to provide a
#             ```score``` function, or ``scoring`` must be passed.
#         :param param_grid: Dictionary with parameters names (string) as keys and lists of parameter settings to try
#             as values.
#         :param cv: int, cross-validation generator or an iterable. Determines the cross-validation splitting strategy.
#             Possible inputs for cv are:
#                 - An iterable yielding (train, test) splits as arrays of indices.
#         :param refit:  Refit an estimator using the best found parameters on the whole dataset. When there are
#             considerations other than maximum score in choosing a best estimator, ``refit`` can be set to a function
#             which returns the selected ``best_index_`` given ``cv_results_``. The refitted estimator is made available
#             at the ``best_estimator_`` attribute and permits using ``predict`` directly on this ``GridSearchCV`` instance.
#         :param verbose: integer
#             Controls the verbosity: the higher, the more messages.
#         :param error_score: 'raise' or numeric
#             Value to assign to the score if an error occurs in estimator fitting. If set to 'raise', the error is raised.
#             If a numeric value is given, FitFailedWarning is raised. This parameter does not affect the refit
#             step, which will always raise the error. Default is 'raise' but from version 0.22 it will change to np.nan.
#         :param scoring: string, callable, list/tuple, dict or None, default: None
#             A single string (see :ref:`scoring_parameter`) or a callable (see :ref:`scoring`) to evaluate the
#             predictions on the test set. For evaluating multiple metrics, either give a list of (unique) strings or a
#             dict with names as keys and callables as values.
#
#             NOTE that when using custom scorers, each scorer should return a single value. Metric functions returning a
#             list/array of values can be wrapped into multiple scorers that return one value each.
#
#             See :ref:`multimetric_grid_search` for an example.
#
#             If None, the estimator's score method is used.
#         :param return_train_score: boolean, default=False
#             If ``False``, the ``cv_results_`` attribute will not include training scores. Computing training scores is
#             used to get insights on how different parameter settings impact the overfitting/underfitting trade-off.
#             However computing the scores on the training set can be computationally expensive and is not strictly
#             required to select the parameters that yield the best generalization performance.
#
#         Attributes:
#         -----------
#         cv_results: dict of numpy (masked) ndarrays
#             A dict with keys as column headers and values as columns, that can be imported into a pandas ``DataFrame``.
#
#             For instance the below given table
#
#             +------------+-----------+------------+-----------------+---+---------+
#             |param_kernel|param_gamma|param_degree|split0_test_score|...|rank_t...|
#             +============+===========+============+=================+===+=========+
#             |  'poly'    |     --    |      2     |       0.80      |...|    2    |
#             +------------+-----------+------------+-----------------+---+---------+
#             |  'poly'    |     --    |      3     |       0.70      |...|    4    |
#             +------------+-----------+------------+-----------------+---+---------+
#             |  'rbf'     |     0.1   |     --     |       0.80      |...|    3    |
#             +------------+-----------+------------+-----------------+---+---------+
#             |  'rbf'     |     0.2   |     --     |       0.93      |...|    1    |
#             +------------+-----------+------------+-----------------+---+---------+
#
#             will be represented by a ``cv_results_`` dict of::
#
#             {
#             'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
#                                          mask = [False False False False]...)
#             'param_gamma': masked_array(data = [-- -- 0.1 0.2],
#                                         mask = [ True  True False False]...),
#             'param_degree': masked_array(data = [2.0 3.0 -- --],
#                                          mask = [False False  True  True]...),
#             'split0_test_score'  : [0.80, 0.70, 0.80, 0.93],
#             'split1_test_score'  : [0.82, 0.50, 0.70, 0.78],
#             'mean_test_score'    : [0.81, 0.60, 0.75, 0.85],
#             'std_test_score'     : [0.01, 0.10, 0.05, 0.08],
#             'rank_test_score'    : [2, 4, 3, 1],
#             'split0_train_score' : [0.80, 0.92, 0.70, 0.93],
#             'split1_train_score' : [0.82, 0.55, 0.70, 0.87],
#             'mean_train_score'   : [0.81, 0.74, 0.70, 0.90],
#             'std_train_score'    : [0.01, 0.19, 0.00, 0.03],
#             'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
#             'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
#             'mean_score_time'    : [0.01, 0.06, 0.04, 0.04],
#             'std_score_time'     : [0.00, 0.00, 0.00, 0.01],
#             'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
#             }
#
#         """
#         self.estimator = estimator
#         self.param_grid = param_grid
#         _check_param_grid(param_grid)
#         logging.warning('GridSearchCVSaveRestore.__init__ method may not be implemented yet in entirety. Reference sklearn\'s BaseEstimator superclass.')
#         # raise NotImplementedError("__init__ method not finished being implemented yet. Reference sklearn's BaseEstimator superclass")
#
#     def _run_search(self, evaluate_candidates):
#         """
#         _run_search: Repeatedly calls ```evaluate_candidates``` to conduct a search of all candidates in the param_grid.
#         :param evaluate_candidates:
#         :return:
#         """
#         evaluate_candidates(ParameterGrid(self.param_grid))
#         raise NotImplementedError("_run_search not implemented.")
#
#     def _check_is_fitted(self, method):
#         """
#         _check_is_fitted:
#         :param method:
#         :return:
#         """
#         raise NotImplementedError
#
#     def score(self, X, y=None):
#         """
#         score: Returns the score on the given data, if the estimator has been refit. Uses the score defined by the
#             ``scoring`` parameter where provided (during instantiation), and the ``best_estimator_.score`` method otherwise.
#         :param X:
#         :param y:
#         :return:
#         """
#         raise NotImplementedError
#
#     def fit(self, X, y=None, groups=None, **fit_params):
#         """
#         Run fit with all sets of parameters.
#         :param X: array-like, shape = [n_samples, n_features]
#             Training vector, where n_samples is the number of samples and n_features is the number of features.
#         :param y: array-like, shape = [n_samples] or [n_samples, n_output], optional
#             Target relative to X for classification or regression; None for unsupervised learning.
#         :param groups: array-like, with shape (n_samples,), optional
#             Group labels for the samples used while splitting the dataset into train/test set. Only used in conjunction
#             with a "Group" `cv` instance (e.g., `GroupKFold`).
#         :param fit_params: dict of string -> object
#             Parameters passed to the ```fit``` method of the estimator.
#
#         Source: This method was copied almost verbatim (for educational usage only) from the sklearn source code
#             available here:
#             * https://github.com/scikit-learn/scikit-learn/blob/1495f69242646d239d89a5713982946b8ffcf9d9/sklearn/model_selection/_search.py#L583
#
#         :return:
#         """
#         estimator = self.estimator
#         cv = check_cv(self.cv, y, classifier=is_classifier(estimator=estimator))
#
#         logging.warning('GridSearchCVSaveRestore.fit() access to a protected member of class '
#                         'sklearn.model_selection._split (method check_cv) not yet screened.')
#
#         scorers, _ = _check_multimetric_scoring(self.estimator, scoring=self.scoring)
#         logging.warning('multimetric scoring not implemented yet. Do not attempt to use.')
#
#         refit_metric = 'score'
#         X, y, groups = indexable(X, y, groups)
#         n_splits = cv.get_n_splits(X, y, groups)
#         base_estimator = clone(self.estimator)
#
#         parallel = Parallel(n_jobs=self.n_jobs)
#
#         fit_and_score_kwargs = dict(scorer=scorers, fit_params=fit_params, return_train_score=self.return_train_score,
#                                     return_n_test_samples=True, return_times=True, return_parameters=False,
#                                     error_score=self.error_score, verbose=self.verbose)
#         results = {}
#         all_candidate_params = []
#         all_out = []
#
#         def evaluate_candidates(candidate_params):
#             candidate_params = list(candidate_params)
#             n_candidates = len(candidate_params)
#
#             if self.verbose > 0:
#                 print("Fitting {0} folds for each of {1} candidates, totalling {2} fits".format(
#                               n_splits, n_candidates, n_candidates * n_splits))
#
#
#
#         raise NotImplementedError
#
#     def save_grid_search(self, save_freq):
#         """
#         save_grid_search: Saves the GridSearchCV by serializing and writing the cv_results to disk. The cv_results will
#             be saved to the hard drive every `freq` times a model has been trained. In this manner, the
#             ```restore_grid_search``` method is able to restore the search from the last TensorFlow model to generate
#             a readable final serialized model.
#         :param save_freq: How many models should be sequentially fit and evaluated via the ```score``` method (during
#             the grid search) before performing a save operation.
#         :return:
#         """
#         raise NotImplementedError
#
#     def restore_grid_search(self):
#         """
#         restore_grid_search: Restores the GridSearchCV using the data serialized by the ```save_grid_search``` method.
#             This method will restore the cv_results of a prior interrupted grid search. Training of the last model
#             will be resumed via TensorFlow from the checkpoint file.
#         :return:
#         """
#         raise NotImplementedError


def _check_param_grid(param_grid):
    """
    _check_param_grid: Ensures the provided parameter grid is appropriately formatted for usage with sklearn's ParamGrid
        class. This function was copied verbatim (for educational usage) from sklearn's source code available here:
            * https://github.com/scikit-learn/scikit-learn/blob/1495f69242646d239d89a5713982946b8ffcf9d9/sklearn/model_selection/_search.py#L358
    :param param_grid:
    :return:
    """
    if hasattr(param_grid, 'items'):
        param_grid = [param_grid]

    for p in param_grid:
        for name, v in p.items():
            if isinstance(v, np.ndarray) and v.ndim > 1:
                raise ValueError("Parameter array should be one-dimensional.")

            if (isinstance(v, str) or
                    not isinstance(v, (np.ndarray, Sequence))):
                raise ValueError("Parameter values for parameter ({0}) need "
                                 "to be a sequence(but not a string) or"
                                 " np.ndarray.".format(name))

            if len(v) == 0:
                raise ValueError("Parameter values for parameter ({0}) need "
                                 "to be a non-empty sequence.".format(name))
