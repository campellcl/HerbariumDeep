import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
import numpy as np

class CrossValidationSplitter(sklearn.model_selection.ShuffleSplit):

    def __init__(self, n_splits, test_size=None, train_size=None, random_state=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def get_n_splits(self):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        for train, test in self._iter_indices(X, y, groups):
            yield train, test


class InceptionV3Estimator(BaseEstimator, ClassifierMixin, tf.keras.Model):

    def __init__(self):
        super(InceptionV3Estimator, self).__init__(name='inception_v3_estimator')
        raise NotImplementedError

    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError

    def fit(self, X_train, y_train, X_valid, y_valid, num_epochs):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError


class Driver:
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [3, 4], [5, 6]])
    y_train = np.array([1, 2, 1, 2, 1, 2])
    X_test = np.array([[3, 4], [3, 4], [6, 7], [9, 4], [2, 3], [4, 5]])
    y_test = np.array([2, 2, 1, 2, 1, 2])
    params = {
        'train_batch_size': [10, 20, 30]
    }
    num_epochs = 100
    eval_freq = 10
    ckpt_freq = 0
    keras_classifier = InceptionV3Estimator()
    custom_cv_splitter = CrossValidationSplitter()
    grid_search = GridSearchCV(keras_classifier, params, cv=CrossValidationSplitter())


if __name__ == '__main__':
    pass
