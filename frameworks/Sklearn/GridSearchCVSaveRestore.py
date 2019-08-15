"""
GridSearchCVSaveRestore.py
Similar to the Sklearn native grid search, but is capable of being interrupted and restarted with save and restore
    functionality. This class is heavily based off of:
    * https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    * https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/model_selection/_search.py#L829
    * https://scikit-learn.org/stable/modules/grid_search.html#grid-search
"""
import os
import ast
import logging
import warnings
import pickle
import time
import types
import json
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
from sklearn.utils.fixes import MaskedArray
from sklearn.metrics.scorer import _check_multimetric_scoring, check_scoring
from sklearn.utils._joblib import Parallel, delayed
from scipy.stats import rankdata
from collections import defaultdict
import Lib.copy


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

    def __init__(self, estimator, param_grid, cv_results_save_freq, cv_results_save_loc, scoring=None, n_jobs=None,
                 iid='warn', refit=True, cv='warn', verbose=0, pre_dispatch='2*n_jobs', error_score='raise-deprecating',
                 return_train_score=False):
        """
        __init__: Initialization method for objects of type GridSearchCVSaveRestore.
        :param estimator:
        :param param_grid:
        :param cv_results_save_freq: How frequently (in terms of trained models) the GridSearch's cv_results dict should
            be serialized and saved to disk.
        :param cv_results_save_loc: The file location for the saved cv_results.
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
        self.cv_results_save_loc = cv_results_save_loc
        self.cv_results = []

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

        # parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(scorer=scorers, fit_params=fit_params, return_train_score=self.return_train_score,
                                    return_n_test_samples=True, return_times=True, return_parameters=False,
                                    error_score=self.error_score, verbose=self.verbose)

        results = []
        all_candidate_params = []
        all_out = []

        # def evaluate_candidate(candidate_param):
        #     """
        #     evaluate_candidate: Similar to sklearn's evaluate_candidates method (see below) but only evaluates a single
        #         candidate. This is done in order to control the saving of the Grid Search's cv_results dictionary.
        #     :param candidate_param:
        #     :return:
        #     """
        #     n_candidates = 1
        #
        #     if self.verbose > 0:
        #         print("Fitting {0} folds for each of {1} candidates, totalling {2} fits".format(
        #                           n_splits, n_candidates, n_candidates * n_splits))
        #
        #     # out = parallel(delayed(_fit_and_score)(clone(base_estimator),
        #     #                                        X, y, train=train, test=test, parameters=parameters, **fit_and_score_kwargs)
        #     #                for parameters, (train, test) in product(candidate_param, cv.split(X, y, groups)))
        #
        #     print('list(cv.split(X, y, groups)): %s' % list(cv.split(X, y, groups)))
        #     print('candidate_param: %s' % candidate_param)
        #     print('list(product(candidate_param, cv.split(X, y, groups))): %s' % list(product(candidate_param, cv.split(X, y, groups))))
        #     # print(_fit_and_score(clone(base_estimator), X, y, ))
        #
        #     parameters, (train, test) = product(candidate_param, cv.split(X, y, groups))
        #     out = None
        #
        #     # A bit of syntactical sugar here with unrolling the Parallel call, see: https://stackoverflow.com/a/51934579/3429090
        #     # delayed(_fit_and_score)(clone(base_estimator), X, y, train=train, test=test, parameters=parameters, **fit_and_score_kwargs) for parameters, (train, test) in product(candidate_param, cv.split(X, y, groups))
        #     # _fit_and_score(base_estimator, X, y, )
        #
        #     if len(out) < 1:
        #         raise ValueError('No fits were performed. '
        #                          'Was the CV iterator empty? '
        #                          'Were there no candidates?')
        #     elif len(out) != n_candidates * n_splits:
        #         raise ValueError('cv.split and cv.get_n_splits returned '
        #                          'inconsistent results. Expected {} '
        #                          'splits, got {}'
        #                          .format(n_splits,
        #                                  len(out) // n_candidates))
        #
        #     all_candidate_params.extend(candidate_param)
        #     all_out.extend(out)
        #
        #     results = self._format_results(
        #         all_candidate_params, scorers, n_splits, all_out)
        #     logging.warning('evaluate_candidate not finished being implemented.')
        #     return results
        #
        #     # raise NotImplementedError('evaluate_candidate not finished being implemented.')

        def evaluate_candidates(candidate_params):
            if isinstance(candidate_params, dict) or isinstance(candidate_params, defaultdict):
                candidate_params = list(candidate_params)
            n_candidates = len(candidate_params)

            if self.verbose > 0:
                print("Fitting {0} folds for each of {1} remaining candidates, totalling {2} fits".format(
                              n_splits, n_candidates, n_candidates * n_splits))

            # print('list(cv.split(X, y, groups)): %s' % list(cv.split(X, y, groups)))
            # print('list(product(candidate_params, cv.split(X, y, groups))): %s' % list(product(candidate_params, cv.split(X, y, groups))))

            fold_num = 0
            for parameters, (train, test) in product(candidate_params, cv.split(X, y, groups)):
                print('product index/fold number: %d' % fold_num)
                print('\tparams: %s' % parameters)
                print('\ttrain: %s' % train)
                print('\ttest: %s' % test)
                out = _fit_and_score(estimator=clone(base_estimator), X=X, y=y, train=train, test=test, parameters=parameters, **fit_and_score_kwargs)
                print('\tout: %s' % out)

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)
                # nonlocal keyword is exactly what it sounds like, uses the outer function scope: w3schools.com/python/ref_keyword_nonlocal.asp
                nonlocal results
                # results = self._format_results(all_candidate_params, scorers, n_splits, all_out)
                result = self._format_result(candidate_param=parameters, scorer=scorers, n_splits=n_splits, out=out)
                results.append(result)
                self.cv_results.append(result)
                # Just finished training a model, should cv_results be saved?
                if fold_num % self.cv_results_save_freq == 0:
                    self._save_cv_results()
                fold_num += 1
            return self.cv_results

        self._run_search(evaluate_candidates)

        if self.refit or not self.multimetric_:
            if callable(self.refit):
                self.best_index_ = self.refit(results)
                if not isinstance(self.best_index_, (int, np.integer)):
                    raise TypeError('best_index_ returned is not an integer')
                if (self.best_index_ < 0 or self.best_index_ >= len(self.cv_results)):
                    raise IndexError('best_index_ index out of range')
            else:
                self.best_index_ = 0
                self.best_score_ = self.cv_results[0]['test_score']
                for i, cv_result in enumerate(self.cv_results):
                    if cv_result['test_score'] >= self.best_score_:
                        self.best_index_ = i
                        self.best_score_ = cv_result['test_score']
            self.best_params_ = self.cv_results[self.best_index_]['params']

        if self.refit:
            self.best_estimator_ = clone(base_estimator).set_params(**self.best_params_)
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers['score']
        self.n_splits_ = n_splits
        return self

    def _save_cv_results(self):
        # serialized_cv_results = Lib.copy.deepcopy(cv_results)
        # NOTE: self.cv_results should already be updated with the serialized file prior to this function's invocation.

        # Replace all the functions in the dictionary with a string representation of the function for serialization:
        for dictionary in self.cv_results:
            parameters = dictionary['params']
            for param, method in parameters.items():
                if not isinstance(method, int):
                    # If this is an integer, we don't need to convert to a string __repr__.
                    if not isinstance(method, str):
                        # If we are simply re-serializing a function that has already had __repr__ called, there is no
                        #   need to call __repr__ again (or we get double quotes messing up string equality checks).
                        if isinstance(method, types.ClassType):
                            # If this is one of the weird parameters that is actually a class instead of a function,
                            #   e.g. TruncatedNormal initializer, then the class can't be called with initializer.__repr__()
                            #   without arguments. So just use the string representation of the method.
                            parameters[param] = str(method)
                        else:
                            # This is not one of the weird parameters that is a class, so just call the __repr__ method
                            #   of the associated function.
                            parameters[param] = method.__repr__()
        # Dump the serialized method names to the folder for restoration if the search crashes.
        serialized_cv_results_path = os.path.join(self.cv_results_save_loc, 'cv_results.json')
        with open(serialized_cv_results_path, 'w') as fp:
            json.dump(self.cv_results, fp)

        print('Saved json serialized cv_results to: \'%s\'' % serialized_cv_results_path)
        return

    def _run_search(self, evaluate_candidates):
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
        param_grid = ParameterGrid(self.param_grid)

        ''' Was this grid search previously interrupted? If so, restore from the serialized version: '''
        serialized_cv_results_path = os.path.join(self.cv_results_save_loc, 'cv_results.json')
        serialized_cv_results = None

        if os.path.isfile(serialized_cv_results_path):
            with open(serialized_cv_results_path, 'r') as fp:
                serialized_cv_results = json.load(fp)
            self.cv_results = serialized_cv_results
        else:
            # Grid Search not previously interrupted, attempt to compute entire grid search and periodically save results:
            all_results = evaluate_candidates(param_grid)
            self.cv_results = all_results
            return

        ''' Modify the parameter grid to exclude the already completed grid searches: '''
        param_grid_list = list(param_grid)
        serialized_cv_results_list = serialized_cv_results

        param_grid_list_indices_of_already_computed_params = []
        already_computed_cv_results = []

        for i, previously_computed_params in enumerate(self.cv_results):
            for j, param_grid_params in enumerate(param_grid_list):
                activation_func = param_grid_params['activation'].__repr__()
                activation_func = activation_func.split('at')[0].strip(' ')
                previously_computed_activation_func = previously_computed_params['params']['activation'].split('at')[0].strip(' ')
                previously_computed_activation_func = previously_computed_activation_func.replace('\'', '')

                initializer_func = param_grid_params['initializer']
                if isinstance(initializer_func, types.ClassType):
                    # Type <class 'tensorflow.python.ops.init_ops.TruncatedNormal'>
                    initializer_func = str(initializer_func)
                else:
                    initializer_func = initializer_func.__repr__()
                    try:
                        initializer_func = initializer_func.split('at')[0].strip(' ')
                    except TypeError as err:
                        # <class 'tensorflow.python.ops.init_ops.TruncatedNormal'> is already in repr form:
                        pass

                previously_computed_initializer_func = previously_computed_params['params']['initializer']
                if isinstance(previously_computed_initializer_func, types.ClassType):
                    print('yay')
                    pass
                elif isinstance(previously_computed_initializer_func, str):
                    if 'TruncatedNormal' in previously_computed_initializer_func:
                        pass
                    else:
                        previously_computed_initializer_func = previously_computed_initializer_func.split('at')[0].strip(' ')
                else:
                    try:
                        previously_computed_initializer_func = previously_computed_initializer_func.split('at')[0].strip(' ')
                        # previously_computed_initializer_func = previously_computed_initializer_func.replace('\'', '')
                    except TypeError as err:
                        # <class 'tensorflow.python.ops.init_ops.TruncatedNormal'> is already in repr form:
                        pass

                optimizer_func = param_grid_params['optimizer'].__repr__()
                optimizer_func = optimizer_func.split('at')[0].strip(' ')
                previously_computed_optimizer_func = previously_computed_params['params']['optimizer'].split('at')[0].strip(' ')
                previously_computed_optimizer_func = previously_computed_optimizer_func.replace('\'', '')

                train_batch_size = param_grid_params['train_batch_size']
                previously_computed_train_batch_size = previously_computed_params['params']['train_batch_size']

                if activation_func == previously_computed_activation_func:
                    if initializer_func == previously_computed_initializer_func:
                        if optimizer_func == previously_computed_optimizer_func:
                            if train_batch_size == previously_computed_train_batch_size:
                                param_grid_list_indices_of_already_computed_params.append(i)
                                print('Already computed parameters: %s' % previously_computed_params['params'])

        param_grid_list_indices_of_already_computed_params = np.unique(param_grid_list_indices_of_already_computed_params)
        print('Detected %d/%d models already fit/trained during a prior grid search. These parameters will be excluded.'
              % (len(param_grid_list_indices_of_already_computed_params), len(param_grid_list)))

        assert len(param_grid_list_indices_of_already_computed_params) == len(self.cv_results)

        # Update the existing parameter grid by removing the parameters that have already been precomputed:
        for list_index in sorted(param_grid_list_indices_of_already_computed_params, reverse=True):
            # param_grid_list.pop(list_index)
            del param_grid_list[list_index]

        # Now restore the cv_results dictionary for sklearn's internal comparisons:
        remaining_results = evaluate_candidates(param_grid_list)
        return

    def _format_result(self, candidate_param, scorer, n_splits, out):
        """
        format_result: Similar to sklearn's _format_results method (see below) with the primary difference being that
            this method formats a single result for the purposes of serialization during save-and-restore functionality
            of a sequential based grid search.
        :source sklearn: https://github.com/scikit-learn/scikit-learn/blob/1495f69242646d239d89a5713982946b8ffcf9d9/sklearn/model_selection/_search.py#L729
        :param candidate_param: The parameters associated with this particular candidate model.
        :param scorer: The method used on the held out data to choose the best parameters for the model.
        :param n_splits: The number of cross-validation splits (folds/iterations).
        :param out: The output metrics calculated from the call to _fit_and_score(), note that this includes the score
            as determined by the scorer function. Also includes fit time in seconds.
        :return:
        """
        n_candidates = 1    # Use _format_results method instead, if this is not the case.

        # if one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            train_score_dict = out[0]
            test_score_dict = out[1]
            test_sample_count = out[2]
            fit_time_in_sec = out[3]
            score_time_in_sec = out[4]
        else:
            test_score_dict = out[0]
            test_sample_count = out[1]
            fit_time_in_sec = out[2]
            score_time_in_sec = out[3]

        # test_score_dicts and train_score dicts are lists of dictionaries and
        # we make them into dict of lists
        # test_scores = _aggregate_score_dicts(test_score_dicts)
        # if self.return_train_score:
        #     train_scores = _aggregate_score_dicts(train_score_dicts)

        result = {
            'test_score': test_score_dict['score'],
            'test_sample_count': test_sample_count,
            'fit_time_in_sec': fit_time_in_sec,
            'score_time_in_sec': score_time_in_sec,
            'params': candidate_param
        }

        # results = {}
        #
        # def _store(key_name, array, weights=None, splits=False, rank=False):
        #     """A small helper to store the scores/times to the cv_results_"""
        #     # When iterated first by splits, then by parameters
        #     # We want `array` to have `n_candidates` rows and `n_splits` cols.
        #     array = np.array(array, dtype=np.float64).reshape(n_candidates, n_splits)
        #
        #     if splits:
        #         for split_i in range(n_splits):
        #             # Uses closure to alter the results
        #             results["split%d_%s"
        #                     % (split_i, key_name)] = array[:, split_i]
        #
        #     array_means = np.average(array, axis=1, weights=weights)
        #     results['mean_%s' % key_name] = array_means
        #     # Weighted std is not directly available in numpy
        #     array_stds = np.sqrt(np.average((array - array_means[:, np.newaxis]) ** 2, axis=1, weights=weights))
        #     results['std_%s' % key_name] = array_stds
        #
        #     if rank:
        #         results["rank_%s" % key_name] = np.asarray(
        #             rankdata(-array_means, method='min'), dtype=np.int32)

        # _store('fit_time', fit_time_in_sec)
        # _store('score_time', score_time_in_sec)
        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        # param_results = defaultdict(partial(MaskedArray, np.empty(n_candidates,), mask=True, dtype=object))
        # for cand_i, params in enumerate(candidate_param):
        #     for name, value in params.items():
        #         # An all masked empty array gets created for the key
        #         # `"param_%s" % name` at the first occurrence of `name`.
        #         # Setting the value at an index also unmasks that index
        #         param_results["param_%s" % name][cand_i] = value
        #
        # results.update(param_results)
        # # Store a list of param dicts at the key 'params'
        # results['params'] = candidate_param
        #
        # # NOTE test_sample counts (weights) remain the same for all candidates
        # test_sample_counts = np.array(test_sample_counts[:n_splits], dtype=np.int)
        #
        # iid = self.iid
        # if self.iid == 'warn':
        #     warn = False
        #     for scorer_name in scorer.keys():
        #         scores = test_score[scorer_name].reshape(n_candidates,
        #                                                   n_splits)
        #         means_weighted = np.average(scores, axis=1,
        #                                     weights=test_sample_counts)
        #         means_unweighted = np.average(scores, axis=1)
        #         if not np.allclose(means_weighted, means_unweighted,
        #                            rtol=1e-4, atol=1e-4):
        #             warn = True
        #             break
        #
        #     if warn:
        #         warnings.warn("The default of the `iid` parameter will change "
        #                       "from True to False in version 0.22 and will be"
        #                       " removed in 0.24. This will change numeric"
        #                       " results when test-set sizes are unequal.",
        #                       DeprecationWarning)
        #     iid = True
        #
        # for scorer_name in scorer.keys():
        #     # Computed the (weighted) mean and std for test scores alone
        #     _store('test_%s' % scorer_name, test_score_dict[scorer_name],
        #            splits=True, rank=True,
        #            weights=test_sample_counts if iid else None)
        #     if self.return_train_score:
        #         _store('train_%s' % scorer_name, train_score_dict[scorer_name],
        #                splits=True)
        #
        # return results
        return result

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
