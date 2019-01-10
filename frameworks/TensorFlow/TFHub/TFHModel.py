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
        self._init = None
        # Enable visible logging output:
        if tf.logging.get_verbosity() is not tf.logging.INFO:
            tf.logging.set_verbosity(tf.logging.INFO)

    def _create_module_graph(self, tfhub_module_spec):
        """
        _create_module_graph: Builds the computational graph from the downloaded TFHub module spec/blueprint.
        :param tfhub_module_spec: A downloaded and cached TFHub module spec/blueprint.
        :returns :
            :return resized_input_tensor: The tensor to be used as input to the TFHub module. This holds images that
                have already been resized to the height and width specified by the chosen hub module. The TFHub module
                will not perform image resizing, and the graph must be augmented with Ops pertaining to this.
            :return bottleneck_tensor: The final layer of the TFHub module (pre-activation) layer. This tensor is used
                to catch bottleneck values prior to them being passed to the final target domain re-train Operations.
        """
        self._graph = tf.Graph()
        height, width = hub.get_expected_image_size(tfhub_module_spec)
        with self._graph.as_default() as graph:
            # Create a placeholder tensor for input to the model (recall that None = minibatch size at runtime)
            resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3], name='resized_input')
            # Declare the model in accordance with the chosen architecture:
            m = hub.Module(tfhub_module_spec, name='inception_v3_hub')
            # Create a placeholder tensor to catch the output of the pre-activation layer:
            bottleneck_tensor = m(resized_input_tensor)
        return resized_input_tensor, bottleneck_tensor

    # def _instantiate_tfhub_module_computational_graph_and_session(self, tfhub_module_spec, module_name):
    #     """
    #     _instantiate_tfhub_module_computational_graph_and_session: Actually instantiates the provided tfhub.ModelSpec
    #         instance, creating a TensorFlow graph object from the provided module specification. The graph is used to
    #         create the sole tf.Session instance so that the lifespan of the session (and it's associated memory
    #         resources) persist across member functions of this class instance. For additional information regarding
    #         tf.Session() object scoping rules, see: https://stackoverflow.com/a/44652465/3429090
    #     :param tfhub_module_spec: <tensorflow_hub.ModuleSpec> The blueprint for the classifier to instantiate a
    #         computational graph for. For more information see:
    #         https://www.tensorflow.org/hub/api_docs/python/hub/ModuleSpec
    #     :param module_name: <str> The name to use for the key that will later allow this module to be retrieved via a
    #         TensorFlow variable_scope. This name will be displayed for this module in TensorBoard visualizations.
    #     :return graph: The sole tf.Graph instance to which
    #     :return bottleneck_tensor: <tf.Tensor> A bottleneck tensor representing the bottleneck values output by the
    #         source module. This layer is the layer just prior to the logits layer; where the logits layer is defined as
    #         the last fully-connected/pre-activation layer that feeds directly into the softmax layer. In other words,
    #         if the softmax layer is the final layer in the neural network, then the logits layer is the dense
    #         fully-connected penultimate layer that precedes the softmax layer. Proceeding this logits layer, is the
    #         "Bottleneck Tensor". For more information on "bottlenecks" see:
    #             https://www.tensorflow.org/hub/tutorials/image_retraining#bottlenecks
    #     """
    #     # Instantiate the sole root/container graph object for this program:
    #     self._graph = tf.Graph()
    #     # Instantiate the sole session for this program (used to persist memory across member functions):
    #     self._session = tf.Session(graph=self._graph)
    #     # Define the receptive field in accordance with the chosen architecture:
    #     height, width = hub.get_expected_image_size(tfhub_module_spec)
    #
    #     with self._session.as_default() as sess:
    #         # NOTE: Can't run session initialization group yet as session graph is empty:
    #         # init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #         # sess.run(init)
    #
    #         # Create a placeholder tensor for inputting X data to the model:
    #         resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3], name='resized_input')
    #
    #         # Initialize the model:
    #         model = hub.Module(tfhub_module_spec, name=module_name, trainable=True)
    #         '''
    #         Calling the model will supposedly add all model operations to the current TensorFlow graph.
    #         See: https://www.tensorflow.org/hub/basics
    #         However, this doesn't seem to be the case within a tf.Session() context, two separate computational graphs
    #             still persist.
    #         '''
    #         y_hat = model(resized_input_tensor)
    #
    #     #     # The empty default graph should now be replaced with the TFHub module graph:
    #     #     self._graph = model._graph
    #     #     with tf.variable_scope('source_model'):
    #     #
    #     #         # Create a placeholder tensor for inputting y data to the model:
    #     #         # ground_truth_input_tensor = tf.placeholder(tf.int32, [None], name='ground_truth_input')
    #     #
    #     #         # Obtain reference to placeholder tensor for inputting X data to the model:
    #     #         input_operation = sess.graph.get_operation_by_name('inception_v3_hub_apply_default/hub_input')
    #     #         # Obtain reference to placeholder tensor holding the prediction issued 'yhat':
    #     #         output_operation = sess.graph.get_operation_by_name('retrain_ops/final_result')
    #     #
    #     #         # Separate variable scope for the tfhub model itself within this container:
    #     #         with tf.variable_scope('tfhub_module'):
    #     #             # Actually instantiate the model that was provided to this method:
    #     #             model = hub.Module(tfhub_module_spec, name=module_name)
    #     #             # Create a placeholder tensor to catch the output of the (pre-activation) layer:
    #     #             # bottleneck_tensor = model(resized_input_tensor)
    #     # self._X = input_operation
    #     # self._y = output_operation


    def _add_final_retrain_ops(self, class_count, output_tensor_name, bottleneck_tensor, is_training=False):
        """
        _add_final_retrain_ops: Augments the default computational graph (which should be identical to the TFHub
            computational graph at this point) with tf.Operation(s) required to fine-tune the source classifier and
            ensure it is run-able in the target domain. This method adds the ops to the graph, along with variables for
            training in the source domain, and then sets up all the gradients for the backward pass.

            The set up for the softmax and fully-connected layers is based on:
                * https://www.tensorflow.org/tutorials/mnist/beginners/index.html

            This code has been modified by Chris Campell from its original source:
                * https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py

            Augmented Input Operations:
                1) 'input_placeholders/bottleneck_input': Feeds into the pre-activation bottleneck tensor of the
                    source TFHub module, in order to bypass the bottleneck generating forward pass (in the event that
                    bottleneck values have already been pre-computed). This saves valuable time by bypassing bottleneck
                    generation when already previously performed.
                2) 'input_placeholders/ground_truth_input': Used to compute loss metrics during training and evaluation
                    metrics when training is complete. Input the ground truth labels of data into this tensor when
                    training or evaluating.
            Augmented Transfer Learning (Target Domain Specific) Retrain Operations:
                1) 'final_retrain_ops/weights': Holds the weights of nodes pertaining to fine tuning in the target
                    domain.
                2) 'final_retrain_ops/biases': Holds the biases of nodes pertaining to fine tuning in the target domain.
                3) 'final_retrain_ops/Wx_plus_b': Holds the pre-activations (logits) prior to application of softmax.
            Augmented Output Operations:
                1) 'output_tensor': Holds the output of the logits when softmax is applied (e.g. holds the value of
                    the tensor 'final_retrain_ops/Wx_plus_b' after applying softmax).
        :param class_count: The number of unique classes in the training, testing, and validation datasets. This is
            assumed to be equal for obvious reasons.
        :param output_tensor_name: <str> The desired name of the tensor containing the final predictions issued by the
            model after fine-tuning for the source domain.
        :param bottleneck_tensor: <tf.Tensor> The final 'output' tensor representing the pre-activation logits of the
            TFHub source model.
        :param is_training: <bool> A boolean indicating if the graph augmentations should include loss and optimizer
            operations (necessary for training), or exclude these operations (for evaluation purposes).
        :returns train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, output_tensor, optimizer_info:
            A series of references to tensors which will be fed via feed-dict during runtime.
            :return train_step: <tf.Operation> The operation to invoke during training that will initialize and execute
                a loss function and minimize it via a tf.Optim instance.
            :return cross_entropy_mean: <tf.Operation> A handle to the loss operation which is to be invoked when
                computing loss during training.
            :return bottleneck_input: <tf.Tensor> A handle to the tensor right before the source models final bottleneck
                tensor. This can be used to bypass the forward pass that computes the bottleneck values in the event
                that they have been already pre-computed.
            :return ground_truth_input: <Tf.Tensor> A handle to the tensor used to capture y values input to the model
                during training.
            :return output_tensor: <tf.Tensor> A handle to the tensor containing the fine-tuned model's predictions in
                the target domain.
            :return optimizer_info: <?> A reference to the optimizer used during training. This information will be used
                in TensorBoard to record the hyperparameters used during training that directly pertain to the optimizer.
        """
        # The batch size is determined by the shape of the bottleneck tensor at runtime:
        batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
        assert batch_size is None, 'We want to work with arbitrary batch size when ' \
                               'constructing fully-connected and softmax layers for fine-tuning.'

        ''' Add placeholder Tensors that feed into the TFHub computational graph '''
        with tf.name_scope('input_placeholders'):
            '''
            This appears odd because we already have a handle to the bottleneck tensor (final pre-activation tensor in
            the TFHub source module). However, we need another handle to the Tensor right before the bottleneck Tensor. 
            Why? Because if we have pre-computed the bottleneck values, we can pass them directly into the bottleneck 
            tensor via this handle (thereby bypassing the forward pass generating the bottleneck tensors).  
            '''
            # Create a placeholder Tensor to feed into the bottleneck tensor (see above comment):
            bottleneck_input = tf.placeholder_with_default(
                bottleneck_tensor,
                shape=[batch_size, bottleneck_tensor_size],
                name='bottleneck_input'
            )
            # Create a placeholder Tensor to hold the ground truth class labels:
            ground_truth_input = tf.placeholder(
                tf.int64,
                shape=[batch_size],
                name='ground_truth_input'
            )

        ''' Add transfer learning target domain final retrain operations: '''
        final_layer_name = 'final_retrain_ops'
        with tf.variable_scope(final_layer_name):
            # The final layer of target domain re-train Operations is composed of the following:
            with tf.name_scope('weights'):
                # Output random values from truncated normal distribution:
                initial_value = tf.truncated_normal(
                    shape=[bottleneck_tensor_size, class_count],
                    stddev=0.001
                )
                layer_weights = tf.Variable(initial_value=initial_value, name='final_weights')

            with tf.name_scope('biases'):
                layer_biases = tf.Variable(initial_value=tf.zeros([class_count]), name='final_biases')

            # pre-activations:
            with tf.name_scope('Wx_plus_b'):
                logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
                tf.summary.histogram('pre_activation_logits', logits)

        # This is the tensor that will hold the predictions of the fine-tuned (re-trained) model:
        output_tensor = tf.nn.softmax(logits=logits, name=output_tensor_name)

        # If this is an eval graph and not the training graph, we don't need loss Ops or an Optimizer:
        if not is_training:
            return None, None, bottleneck_input, ground_truth_input, output_tensor, 'No optimizer'

        # If this is a training graph, we need to add a loss function and an Optimizer for training:
        with tf.name_scope('cross_entropy'):
            cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(labels=ground_truth_input, logits=logits)

        tf.summary.scalar('cross_entropy', cross_entropy_mean)

        with tf.name_scope('train'):
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)
            # TODO: Make momentum a command line argument.
            if optimizer.get_name() == 'Momentum':
                optimizer_info = optimizer.get_name() + '{momentum=%.2f}' % optimizer._momentum
            else:
                optimizer_info = optimizer.get_name() + '{%s}' % (optimizer.get_slot_names())
            train_step = optimizer.minimize(cross_entropy_mean)

        return train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, output_tensor, optimizer_info

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
        resized_input_tensor, bottleneck_tensor = self._create_module_graph(tfhub_module_spec=self.tfhub_module_spec)
        tf.logging.info(msg='Defined computational graph from the TensorFlow Hub module spec.')

        ''' Add transfer learning retrain Operations to the module source graph '''
        with self._graph.as_default() as graph:
            (train_step, eval_metric, bottleneck_input,
             ground_truth_input, final_tensor, optimizer_info) = self._add_final_retrain_ops(
                class_count=self.num_unique_classes, output_tensor_name='predictions',
                bottleneck_tensor=bottleneck_tensor, is_training=True
            )
            # TODO: Calling the above ^ with is_training=True is wasteful. Should have two separate graphs.
            tf.logging.info('Added final retrain Ops to the module source graph.')

        # TODO: Add train_saver instance here for TensorBoard.
        self._init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self._training_op, self._cross_entropy = train_step, eval_metric
        self._bottleneck_input, self._ground_truth_input = bottleneck_input, ground_truth_input
        self._output = final_tensor

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
