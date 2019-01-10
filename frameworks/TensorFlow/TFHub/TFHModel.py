from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from datetime import datetime
from urllib.error import HTTPError
# Custom decorator methods for ease of use when attaching TensorBoard summaries for visualization:
# from frameworks.TensorFlow.TFHub.CustomTensorBoardDecorators import attach_variable_summaries


class TFHModel(BaseEstimator, ClassifierMixin):

    def __init__(self, tfhub_module_url, num_unique_classes, learning_rate, train_batch_size):
        self.tfhub_module_url = tfhub_module_url
        self.num_unique_classes = num_unique_classes
        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        self._graph = None
        self._session = None
        self._init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # Enable visible logging output:
        if tf.logging.get_verbosity() is not tf.logging.INFO:
            tf.logging.set_verbosity(tf.logging.INFO)

    def _instantiate_tfhub_module_computational_graph_and_session(self, tfhub_module_spec, module_name):
        """
        _instantiate_tfhub_module_computational_graph_and_session: Actually instantiates the provided tfhub.ModelSpec
            instance, creating a TensorFlow graph object from the provided module specification. The graph is used to
            create the sole tf.Session instance so that the lifespan of the session (and it's associated memory
            resources) persist across member functions of this class instance. For additional information regarding
            tf.Session() object scoping rules, see: https://stackoverflow.com/a/44652465/3429090
        :param tfhub_module_spec: <tensorflow_hub.ModuleSpec> The blueprint for the classifier to instantiate a
            computational graph for. For more information see:
            https://www.tensorflow.org/hub/api_docs/python/hub/ModuleSpec
        :param module_name: <str> The name to use for the key that will later allow this module to be retrieved via a
            TensorFlow variable_scope. This name will be displayed for this module in TensorBoard visualizations.
        :return graph: The sole tf.Graph instance to which
        :return bottleneck_tensor: <tf.Tensor> A bottleneck tensor representing the bottleneck values output by the
            source module. This layer is the layer just prior to the logits layer; where the logits layer is defined as
            the last fully-connected/pre-activation layer that feeds directly into the softmax layer. In other words,
            if the softmax layer is the final layer in the neural network, then the logits layer is the dense
            fully-connected penultimate layer that precedes the softmax layer. Proceeding this logits layer, is the
            "Bottleneck Tensor". For more information on "bottlenecks" see:
                https://www.tensorflow.org/hub/tutorials/image_retraining#bottlenecks
        """
        # Instantiate the sole root/container graph object for this program:
        self._graph = tf.Graph()
        # Instantiate the sole session for this program (used to persist memory across member functions):
        self._session = tf.Session(graph=self._graph)
        # Define the receptive field in accordance with the chosen architecture:
        height, width = hub.get_expected_image_size(tfhub_module_spec)
        with self._session.as_default() as sess:
            # NOTE: Can't run session initialization group yet as session graph is empty:
            # init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            # sess.run(init)

            # Create a placeholder tensor for inputting X data to the model:
            resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3], name='resized_input')

            # Initialize the model:
            model = hub.Module(tfhub_module_spec, name=module_name)
            ''' 
            Calling the model will supposedly add all model operations to the current TensorFlow graph. 
            See: https://www.tensorflow.org/hub/basics
            However, this doesn't seem to be the case within a tf.Session() context, two separate computational graphs
                still persist. 
            '''
            self._y = model(resized_input_tensor)

            # The empty default graph should now be replaced with the TFHub module graph:
            self._graph = model._graph
            with tf.variable_scope('source_model'):

                # Create a placeholder tensor for inputting y data to the model:
                # ground_truth_input_tensor = tf.placeholder(tf.int32, [None], name='ground_truth_input')

                # Obtain reference to placeholder tensor for inputting X data to the model:
                input_operation = sess.graph.get_operation_by_name('inception_v3_hub_apply_default/hub_input')
                # Obtain reference to placeholder tensor holding the prediction issued 'yhat':
                output_operation = sess.graph.get_operation_by_name('retrain_ops/final_result')

                # Separate variable scope for the tfhub model itself within this container:
                with tf.variable_scope('tfhub_module'):
                    # Actually instantiate the model that was provided to this method:
                    model = hub.Module(tfhub_module_spec, name=module_name)
                    # Create a placeholder tensor to catch the output of the (pre-activation) layer:
                    # bottleneck_tensor = model(resized_input_tensor)
        self._X = input_operation
        self._y = output_operation

    def _build_graph(self, n_inputs, n_outputs):
        """
        _build_graph: Builds and instantiates the computational graph as follows:
            1) Attempts to download the tfhub module spec provided during initialization.
            2) Builds computational graph from the provided TFHub module spec.
            3) Augments the computational graph with transfer learning domain-specific retrain operations.
            4) Augments the computational graph with evaluation operations.
        :param n_inputs:
        :param n_outputs:
        :return:
        """
        ''' Attempt to load the specified TFHub module spec (blueprint): '''
        try:
            # Download the module spec (model blueprint) from the provided URL:
            self.tfhub_module_spec = hub.load_module_spec(self.tfhub_module_url)
            tf.logging.info(msg='Loaded the provided TensorFlowHub module spec: \'%s\'' % self.tfhub_module_url)
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
                             % (self.tfhub_module_url, tfhub_base_url, urllib_http_err))
            exit(-1)
        ''' Perform the actual module instantiation from the blueprint: '''
        module_name = self.tfhub_module_url[self.tfhub_module_url.find('imagenet/') + len('imagenet/')::]
        self._instantiate_tfhub_module_computational_graph_and_session(
            tfhub_module_spec=self.tfhub_module_spec,
            module_name=module_name
        )
        tf.logging.info(msg='Defined computational graph from the TensorFlow Hub module spec.')
        with self._session.as_default() as sess:
            (train_step, eval_metric, bottleneck_input,
             ground_truth_input, final_tensor) = self._add_final_retrain_ops(
                num_unique_classes=self.num_unique_classes, bottleneck_tensor=self.bottleneck_tensor)
            tf.logging.info('Added final retrain Ops to the module source graph.')

    def fit(self, X, y, n_epochs=10, eval_step_interval=None):
        """
        fit: Fits the model using the provided training data over the specified number of epochs. The process is as
            follows:
            1) A TFHub module spec is downloaded.
            2) The TFHub module spec is used to build a computational graph.
            3) The computational graph is augmented with transfer learning retrain Ops.
            4) The computational graph is augmented with evaluation metric Ops.
            5) The computational graph is used to create a sole tf.Session() instance that will house memory
                resources persisting across member functions of this class.
            6) References/handles are obtained to crucial Tensors, and they are stored as class attributes for ease
                of access across member functions.
            7) Data is fed via feedict into the necessary Tensors and the prediction is captured in an output
                Tensor (previously added during step 4).
        :param self:
        :param X:
        :return:
        """
        # The session will not have been initialized yet, so no need to close it.
        n_inputs = X.shape[1]
        n_outputs = self.num_unique_classes
        self.classes_ = np.unique(y)
        if eval_step_interval is not None:
            self._eval_step_interval = eval_step_interval
        else:
            self._eval_step_interval = n_epochs // len(str(n_epochs))
        self._build_graph(n_inputs=n_inputs, n_outputs=n_outputs)
        raise NotImplementedError


if __name__ == '__main__':
    pass
