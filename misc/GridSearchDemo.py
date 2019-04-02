import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
import numpy as np



class CrossValidationSplitter(sklearn.model_selection.ShuffleSplit):

    def __init__(self, n_splits, test_size=None, train_size=None, random_state=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
        split: Yields pairs of X, y indices representing which indices are the training fold and which are the test fold
            in cross validation.
        :param X:
        :param y:
        :param groups:
        :return:
        """
        is_train_data = None
        if len(X) != self.train_size:
            assert len(X)  == self.test_size
            is_train_data = False
        else:
            is_train_data = True
        if is_train_data:
            yield([i for i in range(self.train_size)], [j for j in range(self.train_size)])
        else:
            yield([i for i in range(self.test_size)], [j for j in range(self.test_size)])
        # for i, j in zip(range(self.train_size), range(self.train_size - 1, (self.train_size + self.test_size))):
        #     yield (i, j)
        # yield ([i for i in range(self.train_size)], [j for j in range(self.train_size - 1, (self.train_size + self.test_size))])
        # yield (train_index, test_index for i, j in zip(range(self.train_size), range(self.train_size, self.train_size + self.test_size)))

        # for train, test in self._iter_indices(X, y, groups):
        #     yield train, test


class InceptionV3Estimator(BaseEstimator, ClassifierMixin, tf.keras.Model):

    def __init__(self, train_batch_size=-1, random_state=42):
        super(InceptionV3Estimator, self).__init__(name='inception_v3_estimator')
        self.train_batch_size = train_batch_size
        self.random_state = random_state
        self.eval_freq = None

    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError

    def fit(self, X_train, y_train, X_valid, y_valid, num_epochs, eval_freq, ckpt_freq, early_stopping_eval_freq, fed_bottlenecks):

        # if X_valid is not None and y_valid is not None:
        #     has_validation_data = True
        # else:
        #     has_validation_data = False
        # self.eval_freq = eval_freq
        # if self.random_state is not None:
        #     tf.set_random_seed(self.random_state)
        #     np.random.seed(self.random_state)
        #
        # base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
        #
        # for layer in base_model.layers:
        #     layer.trainable = False
        #
        # if not fed_bottlenecks:
        #     x = base_model.output
        #     bottlenecks = GlobalAveragePooling2D()(x)
        #     logits = Dense(self.num_classes, activation='elu')(bottlenecks)
        #     y_proba = Dense(self.num_classes, activation='softmax')(logits)
        #
        # self._keras_model = Model(inputs=base_model.inputs, outputs=y_proba)
        # raise NotImplementedError
        return

    def predict(self, X):
        print('X:', X)
        print('predict: %s' % (X.shape,))
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
    early_stopping_eval_freq = 1
    keras_classifier = InceptionV3Estimator()
    num_train_samples = X_train.shape[0]
    num_test_samples = X_test.shape[0]
    custom_cv_splitter = CrossValidationSplitter(train_size=num_train_samples, test_size=num_test_samples, n_splits=1)
    grid_search = GridSearchCV(keras_classifier, params, cv=custom_cv_splitter, verbose=2, refit=False, n_jobs=1)
    tf.logging.info(msg='Running GridSearch...')
    grid_search.fit(
        X=X_train,
        y=y_train,
        num_epochs=num_epochs,
        eval_freq=eval_freq,
        ckpt_freq=ckpt_freq,
        early_stopping_eval_freq=early_stopping_eval_freq,
        fed_bottlenecks=True,
        X_valid=X_test,
        y_valid=y_test
    )

if __name__ == '__main__':
    pass
