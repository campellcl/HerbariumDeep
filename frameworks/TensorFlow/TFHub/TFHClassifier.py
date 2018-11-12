"""
TFHClassifier.py
TensorFlow Hub Classifier. A generic class representing a TFHub supported model for use in sklearn's Grid Search
routines with a TensorFlow core.

:sources:
    Heavily inspired by the following URL, but with extensive modification:
        https://github.com/ageron/handson-ml/blob/master/11_deep_learning.ipynb
"""

__author__ = 'Chris Campell'
__version__ = '11/12/2018'

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
import tensorflow as tf
import tensorflow_hub as hub
from urllib.error import HTTPError


class TFHClassifier(BaseEstimator, ClassifierMixin):
    """
    TFHClassifier
    A base class bridging the interfaces between Sklearn and TensorFlow.
    See: https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
    NOTE: All estimators should specify all the parameters that can be set at the class level in their __init__ as
        explicit keyword arguments (no *args or **kwargs).
    """
    tf_session = None
    tf_graph = None     # computational graph

    def __init__(self, tfhub_module_url):
        """
        __init__: Ensures the provided module url is valid, and stores it's hyperparameters for ease of reference.
        :param tfhub_module_url: Which TensorFlow Hub module to instantiate, see the following url for some publicly
            available ones: https://github.com/tensorflow/hub/blob/r0.1/docs/modules/image.md
        """
        # Enable visible logging output:
        if tf.logging.get_verbosity() is not tf.logging.INFO:
            tf.logging.set_verbosity(tf.logging.INFO)
        ''' Attempt to load the specified TFHub module: '''
        try:
            module_spec = hub.load_module_spec(tfhub_module_url)
            tf.logging.info(msg='Loaded the provided TensorFlowHub module spec: \'%s\'' % tfhub_module_url)
        except ValueError as val_err:
            tf.logging.error('Unexpected values in the module spec URL:\n%s' % val_err)
        except tf.OpError as op_err:
            tf.logging.error('TF-File handling exception:\n%s' % op_err)
        except HTTPError as urllib_http_err:
            tfhub_base_url = 'https://tfhub.dev/google/imagenet/'
            tf.logging.error('Could not find a valid model at the provided url: \'%s\'. '
                             'No module was found at the TFHub server: \'%s\'. Received the following stack trace: %s'
                             % (tfhub_module_url, tfhub_base_url, urllib_http_err))

    def fit(self, x, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError


if __name__ == '__main__':
    inception_v3 = TFHClassifier(tfhub_module_url='https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1')




