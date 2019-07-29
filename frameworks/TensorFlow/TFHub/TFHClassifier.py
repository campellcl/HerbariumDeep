from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow_hub as hub
import numpy as np
import os
import pycm
import json

he_init = tf.variance_scaling_initializer()
# he_init = tf.initializers.he_normal


class TFHClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, dataset, class_labels, optimizer=tf.train.AdamOptimizer, train_batch_size=-1, val_batch_size=-1,
                 activation=tf.nn.elu, initializer=he_init,
                 batch_norm_momentum=None, dropout_rate=None, random_state=None, tb_logdir='C:\\tmp\\summaries\\',
                 ckpt_dir='C:\\tmp', saved_model_dir='C:\\tmp\\summaries\\', refit=False):
        """
        __init__: Initializes the TensorFlow Hub Classifier (TFHC) by storing all hyperparameters.
        :param dataset: The dataset type: {BOON, GoingDeeper, SERNEC, debug}
        :param optimizer: The type of optimizer to use during training (tf.train.AdamOptimizer by default).
        :param train_batch_size:
        :param val_batch_size:
        :param activation:
        :param initializer:
        :param batch_norm_momentum:
        :param dropout_rate:
        :param random_state:
        :param tb_logdir: <str> The directory to export training and validation summaries to for tensorboard analysis.
        :param ckpt_dir: <str> The directory to export model snapshots (checkpoints) during training to.
         :param refit: <bool> True if this is the second time fitting a TFHClassifier with the same ParameterGrid, False
            otherwise. This flag is particularly useful when utilizing SKLearn's GridSearchCV, which involves refitting
            the model using the most successful hyperparameter combination from the GridSearch. This boolean flag can
            be used to ensure the TensorBoard logging directories previously populated during the GridSearch are wiped
            clean, before being repopulated with results pertaining exactly to the final refit operation.
        """
        """Initialize the DNNClassifier by simply storing all the hyperparameters."""
        self.dataset = dataset
        # TFHub module spec instance:
        self._module_spec = None
        self.class_labels = class_labels
        self.num_classes = len(class_labels)
        self.optimizer = optimizer
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.activation = activation
        self.initializer = initializer
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.classes_ = None
        self._epoch_train_losses = []

        # TensorFlow Low-Level API: tf.Graph(), tf.Session(), and training graph initializers:
        self._train_graph = None
        self._train_graph_global_init = None
        self._train_session = None
        # TensorFlow Low-Level API: tf.Graph(), tf.Session(), and evaluation graph initializers:
        self._eval_graph = None
        self._eval_graph_global_init = None
        self._eval_session = None

        # Create a FileWriter object to export tensorboard information:
        self._train_writer = None
        self._val_writer = None
        ''' TensorBoard Related Variables: '''
        self._train_graph_merged_summaries = None
        self._eval_graph_merged_summaries = None
        self.relative_ckpt_dir = None
        self.saved_model_dir = saved_model_dir
        self.relative_model_export_dir = None
        self.tb_logdir = tb_logdir
        self.refit = refit

    # def _build_graphs(self, n_inputs, n_outputs):
    #     """
    #     _build_graph: Builds and returns a TensorFlow graph containing the TFHub module sub-graph augmented with
    #         transfer learning re-train Operations, as well as evaluation step operations. Do NOT invoke this method
    #         without care, see the note below.
    #     CRITICAL: This method is to be called only within the scope of the global 'tf.Graph()' context manager associated
    #         with this class instance (self). Failing to call this method from within the associated global context
    #         manager's scope will result in the augmented graph components being added to a separate computational graph.
    #         The implications of this will become apparent when the tf.train.Saver() instance fails to restore the
    #         checkpoint for evaluation and inference due to conflicting graph definitions.
    #     :param n_inputs:
    #     :param n_outputs:
    #     :return:
    #     """
    #     if self.random_state is not None:
    #         tf.set_random_seed(self.random_state)
    #         np.random.seed(self.random_state)
    #
    #     if self.batch_norm_momentum or self.dropout_rate:
    #         self._training = tf.placeholder_with_default(False, shape=(), name='training')
    #     else:
    #         self._training = None
    #
    #     # Load TFHub module spec/blueprint:
    #     tfhub_module_spec = hub.load_module_spec('https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1')
    #     tf.logging.info(msg='Loaded the provided TensorFlowHub module spec: \'%s\'' % tfhub_module_spec)
    #     self._module_spec = tfhub_module_spec
    #
    #     self._train_graph = tf.Graph()
    #     # self._train_graph.name_scope('train_graph')
    #     self._eval_graph = tf.Graph()
    #     # self._eval_graph.name_scope('eval_graph')
    #
    #     # Create the TensorFlow-Hub Module Graphs:
    #     augmented_train_graph, self._train_graph_bottleneck_tensor, self._train_graph_resized_input_tensor = TFHClassifier.create_module_graph(graph=self._train_graph, module_spec=self._module_spec)
    #     augmented_eval_graph, self._eval_graph_bottleneck_tensor, self._eval_graph_resized_input_tensor = TFHClassifier.create_module_graph(graph=self._eval_graph, module_spec=self._module_spec)
    #
    #     # Add transfer learning re-train Ops to training graph:
    #     with augmented_train_graph.as_default() as further_augmented_train_graph:
    #         with further_augmented_train_graph.name_scope('train_graph') as scope:
    #             (training_op, xentropy, X_tensor, y_tensor, logits_tensor, y_proba_tensor) = self._add_final_retrain_ops(
    #                 bottleneck_tensor=self._train_graph_bottleneck_tensor,
    #                 is_training=True,
    #                 final_tensor_name='y_proba'
    #             )
    #             acc_eval_step, top_five_acc_eval_step, predictions = self._add_evaluation_step(
    #                 y_proba_tensor=y_proba_tensor,
    #                 y_tensor=y_tensor
    #             )
    #     augmented_train_graph = further_augmented_train_graph
    #     tf.reset_default_graph()
    #
    #     # Add transfer learning re-train Ops to evaluation graph:
    #     with augmented_eval_graph.as_default() as further_augmented_eval_graph:
    #         with further_augmented_eval_graph.name_scope('eval_graph') as scope:
    #             # Add the transfer learning re-train layers:
    #             (_, _, X_tensor, y_tensor, logits_tensor, y_proba_tensor) = self._add_final_retrain_ops(
    #                 bottleneck_tensor=self._eval_graph_bottleneck_tensor,
    #                 is_training=False,
    #                 final_tensor_name='y_proba'
    #             )
    #
    #             # TODO: Can't restore values from the training graph to eval graph on instantiation.
    #             # Restore the values from the training graph to the eval graph:
    #             # tf.train.Saver().restore(eval_sess, self.ckpt_dir)
    #
    #             # TODO: Will need to add the prediction Ops to the eval session graph for export after restore ^:
    #             # acc_eval_step, top_five_acc_eval_step, predictions = self._add_evaluation_step(
    #             #     y_proba_tensor=y_proba_tensor,
    #             #     y_tensor=y_tensor
    #             # )
    #     augmented_eval_graph = further_augmented_eval_graph
    #     tf.reset_default_graph()
    #     return augmented_train_graph, augmented_eval_graph

    def _build_train_graph(self):
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        if self.batch_norm_momentum or self.dropout_rate:
            self._training = tf.placeholder_with_default(False, shape=(), name='training')
        else:
            self._training = None

        # Load TFHub module spec/blueprint:
        tfhub_module_spec = hub.load_module_spec('https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1')
        tf.logging.info(msg='Loaded the provided TensorFlowHub module spec: \'%s\'' % tfhub_module_spec)
        self._module_spec = tfhub_module_spec

        self._train_graph = tf.Graph()

        # Create the TensorFlow-Hub Module Graphs:
        augmented_train_graph, self._train_graph_bottleneck_tensor, self._train_graph_resized_input_tensor = \
            TFHClassifier.create_module_graph(graph=self._train_graph, module_spec=self._module_spec)

        # Add transfer learning re-train Ops to training graph:
        with augmented_train_graph.as_default() as further_augmented_train_graph:
            with further_augmented_train_graph.name_scope('train_graph') as scope:
                (training_op, xentropy, X_tensor, y_tensor, logits_tensor, y_proba_tensor) = self._add_final_retrain_ops(
                    bottleneck_tensor=self._train_graph_bottleneck_tensor,
                    is_training_graph=True,
                    final_tensor_name='y_proba'
                )
                acc_eval_step, top_five_acc_eval_step, predictions = self._add_evaluation_step(
                    y_proba_tensor=y_proba_tensor,
                    y_tensor=y_tensor
                )
        augmented_train_graph = further_augmented_train_graph
        tf.reset_default_graph()
        return augmented_train_graph

    def _build_eval_graph(self):
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        self._eval_graph = tf.Graph()

        # Create the TensorFlow-Hub Module Graph:
        augmented_eval_graph, self._eval_graph_bottleneck_tensor, self._eval_graph_resized_input_tensor = \
            TFHClassifier.create_module_graph(graph=self._eval_graph, module_spec=self._module_spec)

        self._eval_session = tf.Session(graph=augmented_eval_graph)

        # Add transfer learning re-train Ops to evaluation graph:
        with augmented_eval_graph.as_default() as further_augmented_eval_graph:
            with further_augmented_eval_graph.name_scope('eval_graph') as scope:
                # Add the transfer learning re-train layers:
                (_, _, X_tensor, y_tensor, logits_tensor, y_proba_tensor) = self._add_final_retrain_ops(
                    bottleneck_tensor=self._eval_graph_bottleneck_tensor,
                    is_training_graph=False,
                    final_tensor_name='y_proba'
                )

                # TODO: Can't restore values from the training graph to eval graph on instantiation.
                # Restore the values from the training graph to the eval graph:
                # self._train_saver.restore(self._eval_session, os.path.join(self.ckpt_dir, 'model.ckpt'))
                tf.train.Saver().restore(self._eval_session, os.path.join(self.relative_ckpt_dir, 'model.ckpt'))

                # TODO: Will need to add the prediction Ops to the eval session graph for export after restore ^:
                acc_eval_step, top_five_acc_eval_step, predictions = self._add_evaluation_step(
                    y_proba_tensor=y_proba_tensor,
                    y_tensor=y_tensor
                )
        augmented_eval_graph = further_augmented_eval_graph
        tf.reset_default_graph()
        return augmented_eval_graph

    def close_train_session(self):
        if self._train_session:
            self._train_session.close()

    def close_eval_session(self):
        if self._eval_session:
            self._eval_session.close()

    def _get_model_params(self):
        """Get all variable values (used for early stopping, faster than saving to disk)"""
        with self._train_graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self._train_session.run(gvars))}

    def _restore_model_params(self, model_params):
        """Set all variables to the given values (for early stopping, faster than loading from disk)"""
        gvar_names = list(model_params.keys())
        assign_ops = {gvar_name: self._train_graph.get_operation_by_name(gvar_name + "/Assign")
                      for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        self._train_session.run(assign_ops, feed_dict=feed_dict)

    @staticmethod
    def create_module_graph(graph, module_spec):
        """
        create_module_graph: Creates a tensorflow graph from the provided TFHub module.
        source: https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py
        :param module_spec: the hub.ModuleSpec for the image module being used.
        :returns:
            :return graph: The tf.Graph that was created.
            :return bottleneck_tensor: The bottleneck values output by the module.
            :return resized_input_tensor: The input images, resized as expected by the module.
            :return wants_quantization: A boolean value, whether the module has been instrumented with fake quantization
                ops.
        """
        # tf.reset_default_graph()
        # Define the receptive field in accordance with the chosen architecture:
        height, width = hub.get_expected_image_size(module_spec)
        # Create a new default graph:
        with graph.as_default() as source_model_graph:
            with source_model_graph.name_scope('source_model'):
                # Create a placeholder tensor for input to the model.
                resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3], name='resized_input')
                with tf.variable_scope('pre_trained_hub_module'):
                    # Declare the model in accordance with the chosen architecture:
                    m = hub.Module(module_spec, name='inception_v3_hub')
                    # Create another place holder tensor to catch the output of the pre-activation layer:
                    bottleneck_tensor = m(resized_input_tensor)
        augmented_graph = source_model_graph
        return augmented_graph, bottleneck_tensor, resized_input_tensor

    def _add_final_retrain_ops(self, bottleneck_tensor, is_training_graph, final_tensor_name='y_proba'):
        """
        add_final_retrain_ops: Adds a new softmax and fully-connected layer for training and model evaluation. In order to
            use the TFHub model as a fixed feature extractor, we need to retrain the top fully connected layer of the graph
            that we previously added in the 'create_module_graph' method. This function adds the right ops to the graph,
            along with some variables to hold the weights, and then sets up all the gradients for the backward pass.

            The set up for the softmax and fully-connected layers is based on:
            https://www.tensorflow.org/tutorials/mnist/beginners/index.html
        :source https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py
        :modified_by: Chris Campell
        :param num_classes: The number of unique class labels in both training and validation sets.
        :param final_tensor_name: A name string for the final node that produces the fine-tuned results.
        :param bottleneck_tensor: The output of the main CNN graph (the specified TFHub module).
        :param is_training_graph: Boolean, specifying whether the newly add layer is for the training graph, or
            eval/inference graph.
        :returns : The tensors for the training and cross entropy results, tensors for the
            bottleneck input and ground truth input, a reference to the optimizer for archival purposes and use in the
            hyper-string representation of this training run.
        """
        batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
        assert batch_size is None, 'We want to work with arbitrary batch size when ' \
                               'constructing fully-connected and softmax layers for fine-tuning.'
        with tf.variable_scope('retrain_ops'):
            with tf.name_scope('input'):
                # Create a placeholder Tensor of same type as bottleneck_tensor to cache output from TFHub module:
                X = tf.placeholder_with_default(
                    bottleneck_tensor,
                    shape=[batch_size, bottleneck_tensor_size],
                    name='X'
                )
                # Another placeholder Tensor to hold the true class labels
                y = tf.placeholder(
                    tf.int64,
                    shape=[batch_size],
                    name='y'
                )
                # Scalar placeholder for batch index relevant in epoch accuracy computations across batches:
                batch_index = tf.placeholder(
                    tf.int32,
                    shape=[],
                    name='batch_index'
                )
                predictions = tf.placeholder(
                    tf.int64,
                    shape=[batch_size],
                    name='predictions'
                )
            with tf.variable_scope('final_retrain_ops'):
                # The final layer of target domain re-train operations is composed of the following:
                logits = tf.layers.dense(X, self.num_classes, activation=self.activation, use_bias=True, kernel_initializer=self.initializer, trainable=True, name='logits')
                # tf.summary.histogram('logits', logits)
                # This is the tensor that will hold the predictions of the fine-tuned (re-trained) model:
                y_proba = tf.nn.softmax(logits=logits, name=final_tensor_name)

        if is_training_graph:
            # Take care not to overwrite these with the eval graph's call to this method, hence the conditional:
            self._X, self._y, self._predictions = X, y, predictions
            self._y_proba = y_proba

            if self.dataset == 'SERNEC':
                # epoch_train_losses = tf.placeholder(
                #     tf.int64,
                #     shape=[],
                #     name='epoch_train_losses'
                # )
                with tf.variable_scope('final_retrain_ops'):
                    preds = tf.math.argmax(y_proba, axis=1)

                    batch_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
                    xentropy_summary = tf.summary.histogram('xentropy', batch_xentropy)

                    batch_loss = tf.reduce_mean(batch_xentropy, name='batch_loss')
                    loss_summary = tf.summary.scalar('loss', batch_loss)
                    # self._epoch_train_losses.append(batch_loss)
                    minimization_op = self.optimizer.minimize(batch_loss)

                    # TODO: Move this logic to add_eval_step method?:
                    batch_correct = tf.nn.in_top_k(predictions=y_proba, targets=y, k=1)
                    batch_accuracy = tf.reduce_mean(tf.cast(batch_correct, tf.float32), name='batch_accuracy')
                    acc_summary = tf.summary.scalar('accuracy', batch_accuracy)

                    top_five_predictions = tf.nn.in_top_k(predictions=y_proba, targets=y, k=5)
                    batch_top_five_acc = tf.reduce_mean(tf.cast(top_five_predictions, tf.float32))
                    top_five_acc_summary = tf.summary.scalar('top_five_accuracy', batch_top_five_acc)

                    # Maintain sums for moving averages:
                    batch_loss_sum = tf.Variable(0.0, dtype=tf.float32, name='batch_loss_sum')
                    batch_accuracy_sum = tf.Variable(0.0, dtype=tf.float32, name='batch_acc_sum')
                    batch_top5_acc_sum = tf.Variable(0.0, dtype=tf.float32, name='batch_top5_acc_sum')

                    # Moving average calculation ops:
                    batch_loss_moving_average = batch_loss_sum / tf.cast(batch_index, tf.float32)
                    average_batch_loss_summary = tf.summary.scalar('average_batch_loss', batch_loss_moving_average)
                    batch_acc_moving_average = batch_accuracy_sum / tf.cast(batch_index, tf.float32)
                    average_batch_acc_summary = tf.summary.scalar('average_batch_acc', batch_acc_moving_average)
                    batch_top5_acc_moving_average = batch_top5_acc_sum / tf.cast(batch_index, tf.float32)
                    average_batch_top_five_acc_summary = tf.summary.scalar('average_batch_top_five_acc', batch_top5_acc_moving_average)

                    # Maintain clear operations for resetting running batch averages at every new epoch:
                    clear_batch_loss_moving_average_op = tf.assign(batch_loss_sum, 0.0)
                    clear_batch_acc_moving_average_op = tf.assign(batch_accuracy_sum, 0.0)
                    clear_batch_top5_acc_average_op = tf.assign(batch_top5_acc_sum, 0.0)
                    clear_batch_running_averages_op = tf.group(clear_batch_loss_moving_average_op, clear_batch_acc_moving_average_op, clear_batch_top5_acc_average_op)

                    # Enforce running average updates on training op execution:
                    with tf.control_dependencies([minimization_op]):
                        increment_batch_loss_op = batch_loss_sum.assign_add(batch_loss)
                        increment_batch_acc_op = batch_accuracy_sum.assign_add(batch_accuracy)
                        increment_batch_top5_acc_op = batch_top5_acc_sum.assign_add(batch_top_five_acc)
                        training_op = tf.group(increment_batch_loss_op, increment_batch_acc_op, increment_batch_top5_acc_op)

                    self._batch_loss_moving_average, self._batch_acc_moving_average, self._batch_top5_acc_moving_average = batch_loss_moving_average, batch_acc_moving_average, batch_top5_acc_moving_average
                    self._clear_batch_running_averages_op = clear_batch_running_averages_op
                    self._batch_index = batch_index
                    # When forward propagating the entire validation dataset (which fits in GPU memory), treat these ops as synonomous:
                    self._preds, self._loss = preds, batch_loss
                    self._accuracy, self._top_five_acc = batch_accuracy, batch_top_five_acc
            else:
                # Training graph, add loss Ops and an optimizer:
                with tf.variable_scope('final_retrain_ops'):
                    preds = tf.math.argmax(y_proba, axis=1)

                    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
                    tf.summary.histogram('xentropy', xentropy)

                    loss = tf.reduce_mean(xentropy, name='loss')
                    tf.summary.scalar('loss', loss)

                    training_op = self.optimizer.minimize(loss)

                    # TODO: Move this logic to add_eval_step method:
                    # correct = tf.nn.in_top_k(logits, y, 1)
                    correct = tf.nn.in_top_k(predictions=y_proba, targets=y, k=1)
                    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
                    tf.summary.scalar('accuracy', accuracy)

                    top_five_predictions = tf.nn.in_top_k(predictions=y_proba, targets=y, k=5)
                    top_five_acc = tf.reduce_mean(tf.cast(top_five_predictions, tf.float32))
                    tf.summary.scalar('top_five_accuracy', top_five_acc)

                    self._preds, self._loss = preds, loss
                    self._accuracy, self._top_five_acc = accuracy, top_five_acc

            # init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            train_graph_global_init = tf.global_variables_initializer()

            # Merge all TensorBoard summaries into one object:
            if self.dataset == 'SERNEC':
                # If SERNEC dataset, need different summaries for training and validation datasets:
                # train_graph_training_merged_summaries = tf.summary.merge(['average_batch_loss', 'average_batch_acc', 'average_batch_top_five_acc'])
                train_graph_training_merged_summaries = tf.summary.merge([average_batch_loss_summary, average_batch_acc_summary, average_batch_top_five_acc_summary])
                # train_graph_val_merged_summaries = tf.summary.merge(['xentropy', 'loss', 'accuracy', 'top_five_accuracy'])
                train_graph_val_merged_summaries = tf.summary.merge([xentropy_summary, loss_summary, acc_summary, top_five_acc_summary])
                self._train_graph_training_merged_summaries = train_graph_training_merged_summaries
                self._train_graph_val_merged_summaries = train_graph_val_merged_summaries
            else:
                # One training graph will suffice if both training and validation datasets fit into GPU memory:
                train_graph_merged_summaries = tf.summary.merge_all()
                self._train_graph_merged_summaries = train_graph_merged_summaries

            # Create a saver for checkpoint file creation and restore:
            # ON RESUME: Run to debug here, then explore tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) need some way
            #     to get the logits name to stay consistent. The problem is this graph is not the same as the eval graph,
            #     so when the saver calls restore it is confused. Can run the program and see what tensor names the restore call fails to find.
            # saved_vars_mapping = {logits.name: logits}
            # train_saver = tf.train.Saver(saved_vars_mapping)
            train_saver = tf.train.Saver()

            self._train_graph_global_init = train_graph_global_init
            self._training_op = training_op,
            self._train_saver = train_saver
            if self.dataset == 'SERNEC':
                return training_op, batch_xentropy, X, y, logits, y_proba
            else:
                return training_op, xentropy, X, y, logits, y_proba
        else:
            # Evaluation graph, no need to add loss Ops and an optimizer.

            # init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            eval_graph_global_init = tf.global_variables_initializer()

            # Merge all TensorBoard summaries into one object:
            # eval_graph_merged_summaries = tf.summary.merge_all()

            self._eval_graph_global_init = eval_graph_global_init

            X_tensor = X
            y_tensor = y
            y_proba_tensor = y_proba
            return None, None, X_tensor, y_tensor, logits, y_proba_tensor

    @staticmethod
    def _add_evaluation_step(y_proba_tensor, y_tensor):
        """
        _add_evaluation_step
        :param y_proba_tensor:
        :param y_tensor:
        :return:
        """
        # Create a tensor containing the predicted class label for each training sample (the argmax of the probability tensor)
        predictions = tf.math.argmax(y_proba_tensor, axis=1)

        correct = tf.nn.in_top_k(predictions=y_proba_tensor, targets=y_tensor, k=1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

        top_five_predicted = tf.nn.in_top_k(predictions=y_proba_tensor, targets=y_tensor, k=5)
        top_five_acc = tf.reduce_mean(tf.cast(top_five_predicted, tf.float32))

        acc_eval_step = accuracy
        top_five_acc_eval_step = top_five_acc

        return acc_eval_step, top_five_acc_eval_step, predictions

    def _build_train_session_for_model_export(self):
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)
        if not self._train_graph:
            tf.logging.warning('Expected the training graph to be initialized prior to invocation of _build_train_session_for_model_export()...')
            self._train_graph = tf.Graph()
        if not self._train_session:
            tf.logging.warning('Expected the training session to be initialized prior to invocation of _build_train_session_for_model_export()...')
            augmented_train_graph, self._train_graph_bottleneck_tensor, self._train_graph_resized_input_tensor = TFHClassifier.create_module_graph(graph=self._train_graph, module_spec=self._module_spec)
            self._train_session = tf.Session(graph=augmented_train_graph)
        # Transfer learning re-train Ops are added to the training graph during fit (prior to this method's invocation).
        # TODO: retrieve X_tensor (X:0), y_tensor (y:0), acc_eval_step, top_five_acc_eval_step, predictions
        return self._train_session, self._train_graph_resized_input_tensor, self._train_graph_bottleneck_tensor,


    def _build_eval_session(self):
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        self._eval_graph = tf.Graph()
        # if not self._eval_session:
            # eval_graph, bottleneck_tensor, resized_input_tensor = TFHClassifier.create_module_graph(graph=self._eval_graph, module_spec=self._module_spec)
            # eval_sess = tf.Session(graph=eval_graph)

        augmented_eval_graph, self._eval_graph_bottleneck_tensor, self._eval_graph_resized_input_tensor = TFHClassifier.create_module_graph(graph=self._eval_graph, module_spec=self._module_spec)
        self._eval_session = tf.Session(graph=augmented_eval_graph)

        # Add transfer learning re-train Ops to the evaluation graph:
        with self._eval_session.graph.as_default() as further_augmented_eval_graph:
            with further_augmented_eval_graph.name_scope('eval_graph') as scope:
                # Add the transfer learning re-train layers:
                (_, _, X_tensor, y_tensor, logits_tensor, y_proba_tensor) = self._add_final_retrain_ops(
                    final_tensor_name='y_proba',
                    bottleneck_tensor=self._eval_graph_bottleneck_tensor,
                    is_training_graph=False
                )

                # Restore the trained values form the training graph to the eval graph:
                tf.train.Saver().restore(self._eval_session, os.path.join(self.relative_ckpt_dir, 'model.ckpt'))

                # Add the prediction operations to the eval session graph for export:
                acc_eval_step, top_five_acc_eval_step, predictions = self._add_evaluation_step(
                    y_proba_tensor=y_proba_tensor,
                    y_tensor=y_tensor
                )
        tf.reset_default_graph()
        return self._eval_session, self._eval_graph_resized_input_tensor, X_tensor, y_tensor, acc_eval_step, top_five_acc_eval_step, predictions

    def export_model(self, saved_model_dir, human_readable_class_labels, final_tensor_name='y_proba'):
        """
        Exports a trained model for use with TensorFlow serving.

        Args:
          module_spec: The hub.ModuleSpec for the image module being used.
          class_count: The number of classes.
          saved_model_dir: Directory in which to save exported model and variables.
        """
        # Export eval graph for inference:
        self._eval_session, self._eval_graph_resized_input_tensor, _, _, _, _, _ = self._build_eval_session()
        with self._eval_session.graph.as_default() as graph:
            inputs = {'resized_image_input_tensor': tf.saved_model.utils.build_tensor_info(self._eval_graph_resized_input_tensor)}
            out_classes = graph.get_tensor_by_name('eval_graph/retrain_ops/final_retrain_ops/%s:0' % final_tensor_name)
            outputs = {'y_proba': tf.saved_model.utils.build_tensor_info(out_classes)}
            signature = tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs,
                outputs=outputs,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
            builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(saved_model_dir, 'inference'))
            builder.add_meta_graph_and_variables(
                self._eval_session, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
                },
                main_op=tf.tables_initializer()
            )
            builder.save()
        # training_saved_model_dir is the path to the saved version of the model used for resuming training if interrupted:
        training_saved_model_dir = os.path.join(saved_model_dir, 'training')
        train_builder = tf.saved_model.builder.SavedModelBuilder(training_saved_model_dir)
        with self._train_session as sess:
            inputs = {
                'resized_image_input_tensor': tf.saved_model.utils.build_tensor_info(self._train_graph_resized_input_tensor)
                # 'training_op': tf.saved_model.utils.build_tensor_info(self._training_op[0])
            }
            out_classes = sess.graph.get_tensor_by_name('train_graph/retrain_ops/final_retrain_ops/%s:0' % final_tensor_name)
            outputs = {'y_proba': tf.saved_model.utils.build_tensor_info(out_classes)}
            training_signatures = tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs,
                outputs=outputs,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
            train_builder.add_meta_graph_and_variables(
                self._train_session, [tf.saved_model.tag_constants.TRAINING],
                signature_def_map={
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: training_signatures
                },
                main_op=tf.tables_initializer()
            )
            # train_builder.add_meta_graph(sess, [tf.saved_model.tag_constants.TRAINING], training_signatures, strip_default_attrs=True)
        train_builder.save()

        # Export labels as text file for use in inference:
        with tf.gfile.GFile(os.path.join(saved_model_dir, 'class_labels.txt'), 'w') as fp:
            fp.write('\n'.join(human_readable_class_labels) + '\n')
            # Build signature definition map:
            # feature_configs = {
            #     'resized_input_tensor': tf.FixedLenFeature(shape=self._eval_graph_resized_input_tensor.shape, dtype=tf.float32)
            # }
            # serialized_eval_graph_resized_input_tensor = tf.parse_example(
            #     self._eval_graph_resized_input_tensor,
            #     feature_configs
            # )
            # model_inputs = tf.saved_model.utils.build_tensor_info(serialized_eval_graph_resized_input_tensor)
            # classification_inputs = {
            #     'resized_input_image': tf.saved_model.utils.build_tensor_info(self._eval_graph_resized_input_tensor)
            # }
            # classification_output_classes = {
            #     'y_proba': self._eval_session.graph.get_tensor_by_name('eval_graph/retrain_ops/final_retrain_ops/%s:0' % final_tensor_name)
            # }
            # outputs = {
            #     'y_proba': tf.saved_model.utils.build_tensor_info()
            # }
            # classification_signature = (
            #     tf.saved_model.signature_def_utils.build_signature_def(
            #         inputs={
            #             tf.saved_model.signature_constants.CLASSIFY_INPUTS: classification_inputs
            #         },
            #         outputs={
            #             tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: classification_outputs
            #         },
            #         method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME
            #     )
            # )
            # tensor_info_x = tf.saved_model.utils.build_tensor_info(self._eval_graph.get_tensor_by_name('source_model/pre_trained_hub_module/inception_v3_hub/InceptionV3/input:0'))

            # builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
            # builder.add_meta_graph_and_varaibles(
            #     sess, [tf.saved_model.tag_constants.SERVING],
            #     signature_def_map={
            #         'predict_images':
            #     }
            # )
        return


    def simple_save_model(self):
        pass

    def save_graph_to_file(self, graph_file_name, module_spec, class_count):
        """
        save_graph_to_file: Saves a tensorflow computational graph to a file for use in tensorboard.
        :param graph_file_name: The file name that will be used when saving the resulting graph.
        :param module_spec: The TFHub module specification (i.e. blueprint).
        :param class_count: The number of unique classes in the training and testing datasets (assumed to be identical so
            that the model is not evaluated on classes it hasn't seen).
        :return:
        """
        graph = self._session.graph
        tf.train.write_graph(self._session.graph_def, logdir=self.tb_logdir, name='session_graph_def.meta', as_text=True)

        # Convert variables to constants:
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            self._session, graph.as_graph_def(), ['Y_proba']
        )

        with tf.gfile.GFile(graph_file_name, 'wb') as fp:
            fp.write(output_graph_def.SerializeToString())

    # def print_multiclass_acc(self, y, y_pred):
    #     """
    #     print_multiclass_acc: Prints the accuracy for each class in human readable form
    #     :param y:
    #     :param y_pred:
    #     :return:
    #     """
    #     cm = pycm.ConfusionMatrix(actual_vector=y, predict_vector=y_pred)
    #     out = '\t{'
    #     for clss, acc in cm.ACC.items():
    #         out = out + '\'%s\': %.2f%%,' % (self.class_labels[int(clss)], acc*100)
    #     out = out[:-1]
    #     out = out + '}'
    #     print(out)

    @staticmethod
    def _shuffle_batch(X, y, batch_size):
        """
        _shuffle_batch: Yields an iterable of batch tuples that has been permuted and shuffled so that sampling without
            replacement occurs and each sample is included in at least one batch.
        :source HandsOnML: https://github.com/ageron/handson-ml/blob/master/13_convolutional_neural_networks.ipynb
        :param X: The training data.
        :param y: The target data.
        :param batch_size: The desired batch size. The last array will have len(array) <= batch_size, all others will
            have len(batch_size) length.
        :return:
        """
        # If the batch size is set via shorthand to -1, infer the batch size from dim(0) of X:
        if batch_size == -1:
            batch_size = len(X)
        # Yield batch tuples until exhausted:
        rnd_idx = np.random.permutation(len(X))
        n_batches = len(X) // batch_size
        for batch_idx in np.array_split(rnd_idx, n_batches):
            X_batch, y_batch = X[batch_idx], y[batch_idx]
            yield X_batch, y_batch

    def perform_instance_specific_directory_setup(self):
        """
        perform_instance_specific_directory_setup: Sets up the directories on the hard drive that are specific to the
        hyperparameters passed into the fit method of this instance. As a result, this method must be invoked upon calls
        to .fit() and not with the rest of the directory initialization code found in the GoingDeeper driver class.
        :return:
        """

        '''Some TensorBoard setup: '''
        # TB TrainWriter logging directory:
        if not self.refit:
            self.relative_tb_log_dir_train = os.path.join(self.tb_logdir, 'gs')
            self.relative_tb_log_dir_train = os.path.join(self.relative_tb_log_dir_train, 'train')
        else:
            self.relative_tb_log_dir_train = os.path.join(self.tb_logdir, 'gs_winner\\train')
        self.relative_tb_log_dir_train = os.path.join(self.relative_tb_log_dir_train, self.__repr__())

        # TB ValidationWriter logging directory:
        if not self.refit:
            self.relative_tb_log_dir_val = os.path.join(self.tb_logdir, 'gs')
            self.relative_tb_log_dir_val = os.path.join(self.relative_tb_log_dir_val, 'val')
        else:
            self.relative_tb_log_dir_val = os.path.join(self.tb_logdir, 'gs_winner\\val')
        self.relative_tb_log_dir_val = os.path.join(self.relative_tb_log_dir_val, self.__repr__())

        ''' Setup directories related to checkpointing and model save and restore: '''

        # Create grid search training directory:
        if not os.path.exists(self.relative_tb_log_dir_train):
            try:
                os.mkdir(self.relative_tb_log_dir_train)
            except OSError as err:
                print('FATAL ERROR: Could not create train checkpoint dir manually. Received error: %s' % err)
        # Create grid search validation directory:
        if not os.path.exists(self.relative_tb_log_dir_val):
            try:
                os.mkdir(self.relative_tb_log_dir_val)
            except OSError as err:
                print('FATAL ERROR: Could not create val checkpoint dir manually. Received error: %s' % err)
        # Checkpoint directory relative to this specific instance's hyperparameter combination:
        self.relative_ckpt_dir = os.path.join(self.relative_tb_log_dir_train, 'checkpoints')
        # Attempt to manually create the directory that will later be used by checkpoint Summary writers:
        if not os.path.exists(self.relative_ckpt_dir):
            try:
                os.mkdir(self.relative_ckpt_dir)
            except OSError as err:
                print('ERROR: Failed to create relative hyperparameter directory for checkpoint storage. Received error: %s' % err)
        else:
            print('WARNING: The relative checkpoint write directory somehow already exists prior to FileWriter invocation. Ensure Tensor Board summary writers have no conflict.')
        # Ensure that the directory used by model export code DOES NOT already exist:
        self.relative_model_export_dir = os.path.join(self.relative_tb_log_dir_train, 'trained_model')
        if os.path.exists(self.relative_model_export_dir):
           tf.logging.error(msg='Fatal error. Ensure model export directory: \'%s\' does not exist prior to save of eval graph' % self.relative_model_export_dir)

        # if self.refit:
        #     grid_search_winner_saved_model_dir = os.path.join(self.saved_model_dir, 'gs_winner\\trained_model')
        #     if not os.path.exists(grid_search_winner_saved_model_dir):
        #         try:
        #             os.mkdir(grid_search_winner_saved_model_dir)
        #         except OSError as err:
        #             print('FATAL ERROR: Could not save winning grid search model. Recieved error: %s' % err)
        return self.relative_tb_log_dir_train, self.relative_tb_log_dir_val



    def fit(self, X, y, n_epochs=100, X_valid=None, y_valid=None, eval_freq=1, ckpt_freq=1, early_stop_eval_freq=1):
        """
        fit: Fits the model to the training data. If X_valid and y_valid are provided, validation metrics are reported
            instead of training metrics, and early stopping is employed.
        :param X: The training data.
        :param y: The training targets.
        :param n_epochs: The number of passes over the entire training dataset during fitting.
        :param X_valid: The validation dataset.
        :param y_valid: The validation targets.
        :param eval_freq: How many epochs to train for, before running and displaying the results of an evaluation step.
        :param ckpt_freq: How many epochs to train for, before saving a model checkpoint.
        :param early_stop_eval_freq: This parameter determines how many 'eval_freq' triggering epochs (evaluations on the
            entire validation dataset) should occur before checking to see if the criterion function has decreased.
            Recall that the model is evaluated against the entire X_valid and y_valid datasets every 'eval_freq' epochs
            (if X_valid and y_valid are provided). The early stopping criterion is also updated every 'eval_freq' epochs.
            For instance, if the eval_freq is set to 20 epochs, and the early_stop_eval_freq is set to 1; then on the
            0th epoch, the best encountered criterion function value will be updated by the performance of the model on
            the validation dataset. The next evaluation will then occur on epoch 20 (as dictated by 'eval_freq'). Since,
            early_stop_eval_freq was set to 1, if the criterion function produces a worse value than what was previously
            recorded as the best encountered value, early stopping will be triggered this epoch.
        :return:
        """
        ''' Generic cleanup, and some deferred instantiation: '''
        # Close the session in case it was previously left open for some reason:
        self.close_train_session()
        self.close_eval_session()

        # If the batch size was set to -1 during initialization, infer the training size at runtime from the X matrix:
        if self.train_batch_size == -1:
            self.train_batch_size = len(X)
        if self.val_batch_size == -1:
            if X_valid is not None:
                self.val_batch_size = len(X_valid)
            else:
                self.val_batch_size = None

        # infer n_inputs and n_outputs from the training set:
        n_inputs = X.shape[1]
        self.classes_ = np.unique(y)
        n_outputs = len(self.classes_)

        # needed in case of early stopping:
        max_checks_without_progress = early_stop_eval_freq
        checks_without_progress = 0
        best_loss = np.infty
        best_params = None

        ''' Instance relative directory setup '''
        tb_log_dir_train, tb_log_dir_val = self.perform_instance_specific_directory_setup()

        ''' Build the computational graphs: '''
        self._train_graph = tf.Graph()
        self._eval_graph = tf.Graph()
        # with self._graph.as_default():
        # self._train_graph, self._eval_graph = self._build_graphs(n_inputs, n_outputs)
        self._train_graph = self._build_train_graph()
        # extra ops for batch normalization
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        ''' Train the model! '''
        self._train_session = tf.Session(graph=self._train_graph)
        # self._eval_session = tf.Session(graph=self._eval_graph)
        with self._train_session.as_default() as sess:
            self._train_writer = tf.summary.FileWriter(tb_log_dir_train, sess.graph)
            self._val_writer = tf.summary.FileWriter(tb_log_dir_val)
            self._train_graph_global_init.run()
            # self.running_vars_init.run()

            for epoch in range(n_epochs):
                is_last_step = (epoch + 1 == n_epochs)
                self._epoch_train_losses = []
                # batch partitioning:
                for batch_num, (X_batch, y_batch) in enumerate(self._shuffle_batch(X, y, batch_size=self.train_batch_size)):
                    if self.dataset == 'SERNEC':
                        # Dataset too large to calculate training summaries on entire dataset, maintain them here.
                        # Recall that self._training_op is defined differently for SERNEC to maintain moving average updates across batches:
                        train_summary, _ = sess.run([self._train_graph_training_merged_summaries, self._training_op], feed_dict={self._X: X_batch, self._y: y_batch, self._batch_index: batch_num})
                    else:
                        # Dataset is small enough to calculate training summaries on entire dataset, so run a train op; but don't capture and retain training summaries on the batch level:
                        _ = sess.run([self._training_op], feed_dict={self._X: X_batch, self._y: y_batch})

                # Check to see if eval metrics should be computed this epoch:
                if (epoch % eval_freq == 0) or is_last_step:
                    if self.dataset != 'SERNEC':
                        # Compute evaluation metrics on the entire training dataset:
                        train_summary, train_preds = sess.run([self._train_graph_merged_summaries, self._preds], feed_dict={self._X: X, self._y: y})
                        self._train_writer.add_summary(train_summary, epoch)
                    else:
                        # Dataset is too large for eval metrics to be computed here. Instead, running averages have been
                        #   maintained by a forced control dependency on the training_op.
                        self._train_writer.add_summary(train_summary, epoch)

                    if X_valid is not None and y_valid is not None:
                        if self.dataset == 'SERNEC':
                            # different summary writer:
                            val_summary, loss_val, acc_val, top5_acc, val_preds = sess.run(
                                [self._train_graph_val_merged_summaries, self._loss, self._accuracy, self._top_five_acc, self._preds],
                                feed_dict={self._X: X_valid, self._y: y_valid}
                            )
                        else:
                            # Run eval metrics on the entire validation dataset:
                            val_summary, loss_val, acc_val, top5_acc, val_preds = sess.run(
                                [self._train_graph_merged_summaries, self._loss, self._accuracy, self._top_five_acc, self._preds],
                                feed_dict={self._X: X_valid, self._y: y_valid}
                            )

                        if is_last_step:
                            # last step, no early stopping, export stats:
                            train_cm = pycm.ConfusionMatrix(actual_vector=y, predict_vector=train_preds)
                            val_cm = pycm.ConfusionMatrix(actual_vector=y_valid, predict_vector=val_preds)
                            val_one_hot_class_label_as_chars = val_cm.classes
                            mapping = {one_hot_label: clss_name for one_hot_label, clss_name in zip(val_one_hot_class_label_as_chars, self.class_labels)}
                            # print(mapping)
                            # val_cm.relabel(mapping=mapping)
                            with open(os.path.join(tb_log_dir_train, 'mappings.json'), 'w') as fp:
                                json.dump(mapping, fp, indent=0)
                            with open(os.path.join(tb_log_dir_val, 'mappings.json'), 'w') as fp:
                                json.dump(mapping, fp, indent=0)
                            train_cm.save_html(os.path.join(tb_log_dir_train, 'confusion_matrix'))
                            train_cm.save_csv(os.path.join(tb_log_dir_train, 'confusion_matrix'))
                            val_cm.save_html(os.path.join(tb_log_dir_val, 'confusion_matrix'))
                            val_cm.save_csv(os.path.join(tb_log_dir_val, 'confusion_matrix'))

                        # Update TensorBoard on the results:
                        self._val_writer.add_summary(val_summary, epoch)

                        # Early stopping logic:
                        if loss_val < best_loss:
                            best_params = self._get_model_params()
                            best_loss = loss_val
                            checks_without_progress = 0
                        else:
                            checks_without_progress += 1
                        print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%\tTop-5 Accuracy: {:.2f}%".format(
                            epoch, loss_val, best_loss, acc_val * 100, top5_acc * 100))
                        if checks_without_progress > max_checks_without_progress:
                            print("Early stopping!")
                            if self.dataset != 'SERNEC':
                                train_cm = pycm.ConfusionMatrix(actual_vector=y, predict_vector=train_preds)
                                val_cm = pycm.ConfusionMatrix(actual_vector=y_valid, predict_vector=val_preds)
                                one_hot_class_label_as_chars = val_cm.classes
                                mapping = {int(one_hot_label_char): clss_name for one_hot_label_char, clss_name in zip(one_hot_class_label_as_chars, self.class_labels)}
                                # print(mapping)
                                # cm.relabel(mapping=mapping)
                                with open(os.path.join(tb_log_dir_train, 'mappings.json'), 'w') as fp:
                                    json.dump(mapping, fp, indent=0)
                                with open(os.path.join(tb_log_dir_val, 'mappings.json'), 'w') as fp:
                                    json.dump(mapping, fp, indent=0)
                                train_cm.save_html(os.path.join(tb_log_dir_train, 'confusion_matrix'))
                                train_cm.save_csv(os.path.join(tb_log_dir_train, 'confusion_matrix'))
                                val_cm.save_html(os.path.join(tb_log_dir_val, 'confusion_matrix'))
                                val_cm.save_csv(os.path.join(tb_log_dir_val, 'confusion_matrix'))
                            # Since we are early stopping, need to save checkpoint for restore:
                            is_last_step = True
                            # tf.logging.info(msg='Writing checkpoint (model snapshot) to \'%s\'' % os.path.join(self.relative_ckpt_dir, 'model.ckpt'))
                            # self._train_saver.save(sess, os.path.join(self.relative_ckpt_dir, 'model.ckpt'))
                            # break
                    else:
                        if self.dataset != 'SERNEC':
                            # Run eval metrics on the entire training dataset (as no validation set is available):
                            loss_train, acc_train = sess.run([self._loss, self._accuracy],
                                                                 feed_dict={self._X: X,
                                                                            self._y: y})
                            # print("{}\tLast training batch loss: {:.6f}\tAccuracy: {:.2f}%".format(
                            #     epoch, loss_train, acc_train * 100))
                            print("{}\tLoss on all of X_train: {:.6f}\tAccuracy: {:.2f}%".format(
                                epoch, loss_train, acc_train * 100))
                        else:
                            # Report running average of eval metrics on training dataset
                            # print()
                            # ON RESUME: self._epoch_train_losses has only one tensor in it. The append appears to be failing.
                            # Need to debug this. Continue on adapting code to use minibatches for training due to SERNEC dataset size.
                            epoch_average_batch_loss = sess.run([self._batch_loss_moving_average], feed_dict={self._batch_index: batch_num})
                            # print('type(epoch_average_batch_loss): %s' % type(epoch_average_batch_loss))
                            # print('len(epoch_average_batch_loss): %s' % len(epoch_average_batch_loss))
                            # print("%d\tAverage xentropy loss across all training batches this epoch: %.4f" % (epoch, epoch_average_batch_loss[0]))
                            print("{}\tAverage xentropy loss across all training mini-batches this epoch: {:.4f}%".format(epoch, epoch_average_batch_loss[0]))

                # Check to see if a checkpoint should be recorded this epoch:
                if is_last_step:
                    tf.logging.info(msg='Writing checkpoint (model snapshot) to \'%s\'' % os.path.join(self.relative_ckpt_dir, 'model.ckpt'))
                    self._train_saver.save(sess, os.path.join(self.relative_ckpt_dir, 'model.ckpt'))
                    break
                    # Constant OP for tf.Serving export code of stand-alone model goes here:
                    # tf.logging.info(msg='Writing computational graph with constant-op conversion to \'%s\'' % self.tb_logdir)
                    # intermediate_file_name = (self.ckpt_dir + 'intermediate_' + str(epoch) + '.pb')
                    # self.save_graph_to_file(graph_file_name=intermediate_file_name, module_spec=self._module_spec, class_count=n_outputs)
                else:
                    if ckpt_freq != 0 and epoch > 0 and ((epoch % ckpt_freq == 0)):
                        tf.logging.info(msg='Writing checkpoint (model snapshot) to \'%s\'' % os.path.join(self.relative_ckpt_dir, 'model.ckpt'))
                        self._train_saver.save(sess, os.path.join(self.relative_ckpt_dir, 'model.ckpt'))
                    else:
                        # Don't save checkpoint
                        pass
                # If this is the SERNEC dataset, we have running average counters to reset:
                if self.dataset == 'SERNEC':
                    # Reset the running averages across batches:
                    sess.run(self._clear_batch_running_averages_op)
            # If we used early stopping then rollback to the best model found
            if best_params:
                # print('Restoring model to parameter set of best performing epoch: %s' % best_params)
                self._restore_model_params(best_params)

            # Export the trained model for use with serving:
            # saved_model_dir = os.path.join('C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\frameworks\\TensorFlow\\TFHub\\tmp\\trained_models', self.__repr__())
            self.export_model(saved_model_dir=self.relative_model_export_dir, human_readable_class_labels=self.class_labels, final_tensor_name='y_proba')
            return self

    def predict_proba(self, X):
        '''
        predict_proba: This is called by sklearn internally during training to score accuracy metrics for model ranking.
            The custom CVSplitter class should yield the right data to this method, but this can be tested by examining
            the input shape of X and comparing it to the training subset dimensions. Note that this method uses the
            eval graph to perform evaluations, however this is permissible since the eval_graph is identical to the
            train_graph with the primary difference being the exclusion of an optimizer, and the gradients themselves.
            Recall that the eval_graph is built by restoring the checkpoints of the last train_graph weights.
        :param X:
        :return:
        '''
        if not self._train_session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        if self._train_session._closed:
            # Use the eval graph on the training data:
            with self._eval_session as sess:
                eval_sess_y_proba = sess.graph.get_tensor_by_name('eval_graph/retrain_ops/final_retrain_ops/y_proba:0')
                eval_sess_X = sess.graph.get_tensor_by_name('eval_graph/retrain_ops/input/X:0')
                return eval_sess_y_proba.eval(feed_dict={eval_sess_X: X})

            ''' Use this code if you want to restore the training session from last checkpoint and eval with the training graph instead. '''
            # tf.reset_default_graph()
            # train_sess = tf.Session(graph=self._train_graph)
            # with train_sess.as_default() as sess:
            #     self._train_saver.restore(sess, os.path.join(self.relative_ckpt_dir, 'model.ckpt'))
            #     # y_proba = sess.graph.get_tensor_by_name('train_graph/retrain_ops/final_retrain_ops/y_proba:0')
            #     return self._y_proba.eval(feed_dict={self._X: X})

    def predict(self, X):
        # ON RESUME: OFF BY ON ERROR WHEN max(class_indices) yeilds 996 instead of 995 (zero based index)
        tf.logging.info(msg='predict called with X.shape: %s' % (X.shape,))
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        # Off by one index reduction:
        class_indices_corrected = [1 - len(self.classes_) if class_index == len(self.classes_) else class_index for class_index in class_indices]
        # tf.logging.info(msg='Size of class indices: %s' % (class_indices.shape,))
        # tf.logging.info(msg='Maximum value in class indices: %s' % max(class_indices))
        # tf.logging.info(msg='Size of self.classes_: %s' % (len(self.classes_)))
        # tf.logging.info(msg='self.classes_: %s' % self.classes_)
        return np.array([[self.classes_[class_index]]
                         for class_index in class_indices_corrected], np.int32)

    def save(self, path):
        # Save model checkpoint:
        self._train_saver.save(self._train_session, path)

    @staticmethod
    def _get_initializer_repr(initializer):
        function_repr = str(initializer)
        if 'random_uniform' in function_repr:
            return 'INIT_UNIFORM'
        elif 'random_normal' in function_repr:
            return 'INIT_NORMAL'
        elif 'init_ops.TruncatedNormal' in function_repr:
            return 'INIT_NORMAL_TRUNCATED'
        elif 'he_normal' in function_repr or 'init_ops.VarianceScaling' in function_repr:
            if initializer.distribution == 'uniform':
                # He uniform
                return 'INIT_HE_UNIFORM'
            else:
                # He normal
                return 'INIT_HE_NORMAL'
        else:
            return 'INIT_UNKNOWN'

    @staticmethod
    def _get_optimizer_repr(optimizer):
        class_repr = str(optimizer)
        if 'GradientDescentOptimizer' in class_repr:
            return 'OPTIM_GRAD_DESCENT'
        elif 'AdamOptimizer' in class_repr:
            return 'OPTIM_ADAM'
        elif 'MomentumOptimizer' in class_repr:
            if optimizer._use_nesterov:
                return 'OPTIM_NESTEROV'
            else:
                return 'OPTIM_MOMENTUM'
        elif 'AdaDelta' in class_repr:
            return 'OPTIM_ADADELTA'
        else:
            return 'OPTIM_UNKNOWN'

    @staticmethod
    def _get_activation_repr(activation):
        function_repr = str(activation)
        if 'leaky_relu' in function_repr:
            return 'ACTIVATION_LEAKY_RELU'
        elif 'elu' in function_repr:
            return 'ACTIVATION_ELU'
        else:
            return 'ACTIVATION_UNKNOWN'

    def __repr__(self):
        tfh_repr = '%s,%s,%s,TRAIN_BATCH_SIZE__%d' % (
            self._get_initializer_repr(self.initializer),
            self._get_optimizer_repr(self.optimizer),
            self._get_activation_repr(self.activation),
            self.train_batch_size
        )
        return tfh_repr


# if __name__ == '__main__':
#     n_inputs = 28 * 28 # MNIST
#     n_outputs = 5
#     (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
#     X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
#     X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
#     y_train = y_train.astype(np.int32)
#     y_test = y_test.astype(np.int32)
#     X_valid, X_train = X_train[:5000], X_train[5000:]
#     y_valid, y_train = y_train[:5000], y_train[5000:]
#     X_train1 = X_train[y_train < 5]28
#     y_train1 = y_train[y_train < 5]
#     X_valid1 = X_valid[y_valid < 5]
#     y_valid1 = y_valid[y_valid < 5]
#     X_test1 = X_test[y_test < 5]
#     y_test1 = y_test[y_test < 5]
#     dnn_clf = TFHClassifier(random_state=42)
#     dnn_clf.fit(X_train1, y_train1, n_epochs=10, X_valid=X_valid1, y_valid=y_valid1)
#     y_pred = dnn_clf.predict(X_test1)
#     # accuracy_score(y_test1, y_pred)
#     print(accuracy_score(y_test1, y_pred))
