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
# Custom decorator methods for ease of use when attaching TensorBoard summaries for visualization:
# from frameworks.TensorFlow.TFHub.CustomTensorBoardDecorators import attach_variable_summaries


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
    graph = None     # TensorFlow computational graph
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
            TensorFlow variable_scope. This name will be displayed for this module in TensorBoard visualizations.
        :return graph: The tf.Graph that was created.
        :return bottleneck_tensor: <tf.Tensor> A bottleneck tensor representing the bottleneck values output by the
            source module. This layer is the layer just prior to the logits layer; where the logits layer is defined as
            the last fully-connected/pre-activation layer that feeds directly into the softmax layer. In other words,
            if the softmax layer is the final layer in the neural network, then the logits layer is the dense
            fully-connected penultimate layer that precedes the softmax layer. Proceeding this logits layer, is the
            "Bottleneck Tensor". For more information on "bottlenecks" see:
                https://www.tensorflow.org/hub/tutorials/image_retraining#bottlenecks
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

    @staticmethod
    def _attach_variable_summaries(tf_variable):
        """
        _attach_variable_summaries: Attaches several TensorBoard summaries to the provided Tensor (for TensorBoard
            visualization).
        :param tf_variable: <tf.Variable (tf.Tensor wrapper)>  A TensorFlow tensor to which various TensorBoard
            summaries are to be attached to. When a summary writer is created using this variable, these attached
            summaries are evaluated and exported for visualization in TensorBoard.
        :return None: Upon completion, the provided tf_variable will have had summary statistics attached to it that
            will show in TensorBoard when exported via a summary writer.
        """
        with tf.name_scope('variable_summaries'):
            mean = tf.reduce_mean(tf_variable)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(tf_variable - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(tf_variable))
            tf.summary.scalar('min', tf.reduce_min(tf_variable))
            tf.summary.histogram('histogram', tf_variable)
        return

    def _add_final_retrain_ops(self, num_unique_classes, bottleneck_tensor):
        """
        add_final_retrain_ops: Adds a new softmax and fully-connected layer for training and model evaluation. In order
        to use the TFHub model as a fixed feature extractor, we need to retrain the top fully connected layer of the
        graph that we previously added in the '_instantiate_tfhub_module_computational_graph' method. This function adds
        the right ops to the graph, along with some variables to hold the weights, and then sets up all the gradients
        for the backward pass.

        The set up for the softmax and fully-connected layers is based on:
            * https://www.tensorflow.org/tutorials/mnist/beginners/index.html
        Also see the following additional resources section (with custom tags for ease of meta-level code analysis
            parsing).
        :resource: https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py
        :param num_unique_classes: <int> The number of unique classes in the training, validation, and testing datasets.
        :param bottleneck_tensor: <tf.Tensor> A bottleneck tensor representing the bottleneck values output by the
            source module. This layer is the layer just prior to the logits layer; where the logits layer is defined as
            the last fully-connected/pre-activation layer that feeds directly into the softmax layer. In other words,
            if the softmax layer is the final layer in the neural network, then the logits layer is the dense
            fully-connected penultimate layer that precedes the softmax layer (pre-activation). Before this logits
            layer, is the "Bottleneck Tensor" which holds the output from the original source model's forward
            propagation for every sample in the bottleneck vector. For more information on "bottlenecks" see:
                https://www.tensorflow.org/hub/tutorials/image_retraining#bottlenecks
        :returns A series of tf.Tensor objects (symbolic handles to the result of operations) to be used during the main
            training loop at runtime. See below:
        :return train_step: <tf.Tensor> A tensor representing the result of a series of operations that constitute a
            single step during training.
        :return cross_entropy_mean: <tf.Tensor> A tensor holding the result of the application of the cross_entropy
            evaluation metric. This can be applied to either training, testing, or validation datasets at runtime.
        :return bottleneck_input: <tf.Tensor> A placeholder tensor which is provided the current mini-batch during
            runtime.
        :return ground_truth_input: <tf.Tensor> A tensor to hold the ground truths of the batch provided at runtime.
        :return final_tensor: <tf.Tensor> A tensor (symbolic handle) to the computation produced as the result of a
            forward propagation through the augmented network. This is akin to the softmax layer when using softmax
            as an activation function 'phi' on the last layer.
        """
        '''
        The batch size is determined during runtime depending on the first value passed to the bottleneck tensor 
        during invocation:
        '''
        batch_size, bottleneck_tensor_size = self.bottleneck_tensor.get_shape().as_list()
        '''
        This assert statement makes sure that the last layer of the TFHub module was constructed with the first
        dimension set to None (i.e. the tensor's dimensionality was constructed with mini-batch usage in mind):
        '''
        assert batch_size is None
        ''' Tensor Declarations: '''
        # For TensorBoard (and ability to retrieve collection by name) this is a variable_scope instead of a name_scope:
        with tf.variable_scope('retrain_ops'):
            # Child elements of the retrain_ops/ variable scope are named independently for TensorBoard ease of viz.:
            with tf.name_scope('input'):
                # Create a placeholder Tensor of same type as bottleneck_tensor to cache output from TFHub module:
                bottleneck_input = tf.placeholder_with_default(
                    bottleneck_tensor,
                    shape=[batch_size, bottleneck_tensor_size],
                    name='BottleneckInputPlaceholder'
                )
                # Another placeholder Tensor to hold the true class labels:
                ground_truth_input = tf.placeholder(
                    tf.int64,
                    shape=[batch_size],
                    name='GroundTruthInput'
                )
        # Additional organization for TensorBoard:
        layer_name = 'final_retrain_ops'
        with tf.name_scope(layer_name):
            # Every layer has the following items:
            with tf.name_scope('weights'):
                # Output random values from truncated normal distribution:
                if self.init_type == 'he':
                    # TODO: Add he initialization
                    tf.logging.error('Requested initialization method \'he\' not supported yet.')
                    raise NotImplementedError
                elif self.init_type == 'xavier':
                    # TODO: Add xavier initialization.
                    tf.logging.error('Requested initialization method \'xavier\' not supported yet.')
                    raise NotImplementedError
                else:
                    # TODO: Add support for different distributions other than truncated normal.
                    stddev = 0.001
                    initial_value = tf.truncated_normal(
                        shape=[bottleneck_tensor_size, num_unique_classes],
                        stddev=stddev
                    )
                    tf.logging.info(msg='Computational Graph Construction: Defined weight initialization method as '
                                        'truncated normal distribution with a stddev=%.4f' % stddev)
                layer_weights = tf.Variable(initial_value=initial_value, name='final_weights')
                self._attach_variable_summaries(layer_weights)

            with tf.name_scope('biases'):
                layer_biases = tf.Variable(initial_value=tf.zeros([num_unique_classes]), name='final_biases')
                self._attach_variable_summaries(layer_biases)

            # pre-activations (z vector):
            with tf.name_scope('Wx_plus_b'):
                '''
                NOTE: The "bottleneck layer" is the layer that feeds directly into the 'layer that performs the actual 
                classification' (in this case the softmax layer). During instantiation, the bottleneck_tensor is 
                created to hold the output of the bottleneck-layer/logits-layer computation. To be explicitly clear, 
                the bottleneck tensor is the output of the layer preceding the logits layer. The logits layer is 'z', 
                that is: the weighted sum of the values in the bottleneck tensor (plus a bias term) BEFORE applying the 
                activation function phi (in this case softmax). The softmax layer is the result of applying phi to the 
                preceding logits layer. 
                The terminology can be a bit confusing, for example see:
                https://stackoverflow.com/questions/41455101/what-is-the-meaning-of-the-word-logits-in-tensorflow/47010867#47010867
                '''
                logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
                # Logits are the result of calculating z using the bottleneck tensor, BEFORE applying phi:
                tf.summary.histogram('logits', logits)
        ''' Add final layer to TensorBoard histograms to track the histogram values over time '''
        final_tensor = tf.nn.softmax(logits, name='softmax')
        tf.summary.histogram('activations', final_tensor)

        ''' Add evaluation metric tensors here: '''
        # TODO: Patch constructor input here for supporting multiple evaluation metrics.
        with tf.name_scope('cross_entropy'):
            cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
                labels=ground_truth_input, logits=logits
            )
        tf.summary.scalar('cross_entropy', cross_entropy_mean)

        ''' Define the series of computations that constitute a single training step in the network '''
        with tf.name_scope('train'):
            # TODO: Patch constructor input here for supporting multiple optimization methods.
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            train_step = optimizer.minimize(cross_entropy_mean)

        return train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor

    def __init__(self, tfhub_module_url, init_type, learning_rate, num_unique_classes):
        """
        __init__: Ensures the provided module url is valid, and stores it's hyperparameters for ease of reference.
        NOTE: All estimators should specify all the parameters that can be set at the class level in their __init__ as
            explicit keyword arguments (no *args or **kwargs).
            See: https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        :param tfhub_module_url: <str> Which TensorFlow Hub module to instantiate, see the following url for some publicly
            available ones: https://github.com/tensorflow/hub/blob/r0.1/docs/modules/image.md
        :param init_type: <str> The chosen weight initialization technique: {he, xavier, random}.
        :param learning_rate: <float> The chosen learning rate (eta), range [0,1] inclusive.
        :param num_unique_classes: <int> The number of unique classes in the training, validation, and testing datasets.
        """
        # Make the important operations available easily through instance variables:
        self.init_type = init_type
        self.learning_rate = learning_rate
        self.num_unique_classes = num_unique_classes

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

        ''' Perform the actual module instantiation from the blueprint: '''
        module_name = tfhub_module_url[tfhub_module_url.find('imagenet/') + len('imagenet/')::]
        # Actually instantiate the module blueprint and get a reference to the output bottleneck tensor for use later:
        self.graph, self.bottleneck_tensor = self._instantiate_tfhub_module_computational_graph(
            tfhub_module_spec=self.tfhub_module_spec,
            module_name=module_name
        )
        tf.logging.info(msg='Defined computational graph from the TensorFlow Hub module spec.')
        ''' Add the re-train operations necessary for the model to function in the target domain. Get back tf.Tensor 
            objects that provide a handle to the result of the corresponding operation in the computational graph during
            runtime: 
        '''
        with self.graph.as_default():
            (train_step, cross_entropy, bottleneck_input,
             ground_truth_input, final_tensor) = self._add_final_retrain_ops(
                num_unique_classes=self.num_unique_classes, bottleneck_tensor=self.bottleneck_tensor)

        # Add more important operations to the list of easily available instance variables:
        self._init = tf.global_variables_initializer()
        self._saver = tf.train.Saver()

        #



        # self._fine_tune_and_train_model(num_unique_classes,)






    def fit(self, x, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError


if __name__ == '__main__':

    inception_v3 = TFHClassifier(tfhub_module_url='https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1', init_type='random', learning_rate=0.01, num_unique_classes=)




