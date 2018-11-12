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
    tfhub_module_spec = None
    tf_session = None
    tf_graph = None     # computational graph
    bottleneck_tensor = None   # TF Bottleneck Tensor

    @staticmethod
    def _instantiate_tfhub_module_computational_graph(tfhub_module_spec, module_name):
        """
        _instantiate_tfhub_module_computational_graph: Actually instantiates the provided tfhub.ModelSpec instance,
            creating a TensorFlow graph object from the provided module specification. This method is static so that it
            may be used in another script to instantiate other tfhub modules without the overhead of instantiating an
            instance of this entire class.
        :param tfhub_module_spec: <tensorflow_hub.ModuleSpec> The blueprint for the classifier to instantiate a
            computational graph for. For more information see:
            https://www.tensorflow.org/hub/api_docs/python/hub/ModuleSpec
        :param module_name: <str> The name to use for the key that will later allow this module to be retrieved via a
            Tensorflow variable_scope. This name will be displayed for this module in TensorBoard.
        :return tf_graph: The tf.Graph that was created.
        :return bottleneck_tensor: <tf.placeholder> A bottleneck tensor representing the bottlneck values output by the
            source module. This layer is the pre-activation layer just prior to the logits layer (the last fully
            connected layer that feeds into the softmax layer). In other words, if the softmax layer is the final layer
            in the neural network, then the logits layer is the dense fully-connected penultimate layer that
            precedes it, and the bottleneck tensor is the layer that precedes this logits layer.
        """
        # Define the receptive field in accordance with the chosen architecture:
        height, width = hub.get_expected_image_size(tfhub_module_spec)
        # Create a new default graph:
        with tf.Graph().as_default() as graph:
            ''' Give everything pertaining to the source model it's own variable_scope so it can be both identified and 
                retrieved as a unique entity later by its given name. This will show as a container object in 
                TensorBoard holding all Ops in the computational graph that pertain to the original source model. 
            '''
            with tf.variable_scope('source_model'):
                # Create a placeholder tensor for inputting data to the model:
                resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3], name='resized_input')
                # Separate variable scope for the tfhub model itself within this container:
                with tf.variable_scope('tfhub_module'):
                    # Actually instantiate the model that was provided to this method:
                    model = hub.Module(tfhub_module_spec, name=module_name)
                    # Create a placeholder tensor to catch the output of the (pre-activation) layer:
                    bottleneck_tensor = model(resized_input_tensor)
                    # Give a name to the newly created tensor:
                    tf.identity(bottleneck_tensor, name='bottleneck_tensor')
        return graph, bottleneck_tensor


    def __init__(self, tfhub_module_url):
        """
        __init__: Ensures the provided module url is valid, and stores it's hyperparameters for ease of reference.
        :param tfhub_module_url: Which TensorFlow Hub module to instantiate, see the following url for some publicly
            available ones: https://github.com/tensorflow/hub/blob/r0.1/docs/modules/image.md
        """
        # Enable visible logging output:
        if tf.logging.get_verbosity() is not tf.logging.INFO:
            tf.logging.set_verbosity(tf.logging.INFO)

        ''' Attempt to load the specified TFHub module spec (blueprint): '''
        try:
            # Get the module spec (model blueprint) from the provided URL:
            self.tfhub_module_spec = hub.load_module_spec(tfhub_module_url)
            tf.logging.info(msg='Loaded the provided TensorFlowHub module spec: \'%s\'' % tfhub_module_url)
        except ValueError as val_err:
            tf.logging.error('Unexpected values in the module spec URL:\n%s' % val_err)
            exit(-1)
        except tf.OpError as op_err:
            tf.logging.error('TF-File handling exception:\n%s' % op_err)
            exit(-1)
        except HTTPError as urllib_http_err:
            tfhub_base_url = 'https://tfhub.dev/google/imagenet/'
            tf.logging.error('Could not find a valid model at the provided url: \'%s\'. '
                             'No module was found at the TFHub server: \'%s\'. Received the following stack trace: %s'
                             % (tfhub_module_url, tfhub_base_url, urllib_http_err))
            exit(-1)

        ''' Perform the actual model instantiation: '''
        module_name = tfhub_module_url[tfhub_module_url.find('imagenet/') + len('imagenet/')::]
        # Actually instantiate the module blueprint and get a reference to the output bottleneck tensor for use later:
        self.tf_graph, self.bottleneck_tensor = self._instantiate_tfhub_module_computational_graph(tfhub_module_spec=self.tfhub_module_spec, module_name=module_name)



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




