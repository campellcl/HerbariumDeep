from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow_hub as hub
import numpy as np
import os
import shutil
import pycm

he_init = tf.variance_scaling_initializer()
# he_init = tf.initializers.he_normal


class TFHClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, class_labels, optimizer=tf.train.AdamOptimizer, train_batch_size=-1, val_batch_size=-1,
                 activation=tf.nn.elu, initializer=he_init,
                 batch_norm_momentum=None, dropout_rate=None, random_state=None, tb_logdir='tmp/summaries/',
                 ckpt_dir='tmp/', saved_model_dir='tmp/trained_model/', refit=False):
        """
        __init__: Initializes the TensorFlow Hub Classifier (TFHC) by storing all hyperparameters.
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
        self._module_spec = None
        self.class_labels = class_labels
        self.optimizer = optimizer
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.activation = activation
        self.initializer = initializer
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self._session = None
        self._graph = None
        # Create a FileWriter object to export tensorboard information:
        self._train_writer = None
        self._val_writer = None
        ''' TensorBoard Related Variables: '''
        self.ckpt_dir = ckpt_dir
        self.saved_model_dir = saved_model_dir
        self.tb_logdir = tb_logdir
        self.refit = refit

    @staticmethod
    def _get_initializer_repr(initializer):
        function_repr = str(initializer)
        if 'random_uniform' in function_repr:
            return 'INIT_UNIFORM'
        elif 'random_normal' in function_repr:
            return 'INIT_NORMAL'
        elif 'truncated_normal' in function_repr:
            return 'INIT_NORMAL_TRUNCATED'
        elif 'he_normal' in function_repr or 'init_ops.VarianceScaling' in function_repr:
            # He normal
            return 'INIT_HE_NORMAL'
        elif 'he_uniform' in function_repr:
            return 'INIT_HE_UNIFORM'
        else:
            return 'INIT_UNKNOWN'

    @staticmethod
    def _get_optimizer_repr(optimizer):
        class_repr = str(optimizer)
        if 'GradientDescentOptimizer' in class_repr:
            return 'OPTIM_GRAD_DESCENT'
        elif 'AdamOptimizer' in class_repr:
            return 'OPTIM_ADAM'
        else:
            return 'OPTIM_UNKNOWN'

    def _build_graph(self, n_inputs, n_outputs):
        # I added this to control TB writting, not sure session persistance is the issue depends on SKLearn inner workings.
        # if self._session is not None:
        #     self.close_session()

        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        # X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        # y = tf.placeholder(tf.int32, shape=(None), name="y")

        if self.batch_norm_momentum or self.dropout_rate:
            self._training = tf.placeholder_with_default(False, shape=(), name='training')
        else:
            self._training = None

        # Load module spec/blueprint:
        tfhub_module_spec = hub.load_module_spec('https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1')
        self._module_spec = tfhub_module_spec
        height, width = hub.get_expected_image_size(tfhub_module_spec)
        tf.logging.info(msg='Loaded the provided TensorFlowHub module spec: \'%s\'' % tfhub_module_spec)

        # Create a placeholder tensor for image input to the model (when bottleneck has not been pre-computed).
        resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3], name='resized_input')

        # Declare the model in accordance with the chosen architecture:
        m = hub.Module(tfhub_module_spec)

        # Create a placeholder tensor to catch the output of the pre-activation layer:
        bottleneck_tensor = m(resized_input_tensor)

        '''
        Add re-train operations:
        '''
        batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
        assert batch_size is None, 'We want to work with arbitrary batch size when ' \
                               'constructing fully-connected and softmax layers for fine-tuning.'

        X = tf.placeholder_with_default(
            bottleneck_tensor,
            shape=[batch_size, bottleneck_tensor_size],
            name='X'
        )
        y = tf.placeholder(
            tf.int64,
            shape=[batch_size],
            name='y'
        )
        num_classes = tf.placeholder(tf.int32, shape=(), name='NumClasses')
        predictions = tf.placeholder(
            tf.int64,
            shape=[batch_size],
            name='predictions'
        )

        ''' Add transfer learning target domain final retrain operations: '''
        final_layer_name = 'final_retrain_ops'
        with tf.variable_scope(final_layer_name):
            # The final layer of target domain re-train Operations is composed of the following:
            logits = tf.layers.dense(X, n_outputs, activation=self.activation, use_bias=True, kernel_initializer=self.initializer, trainable=True, name='logits')
            tf.summary.histogram('logits', logits)

            # with tf.name_scope('weights'):
            #     # Output random values from the initializer:
            #     if 'random_uniform' in str(self.initializer):
            #     # if self.initializer_repr == 'INIT_UNIFORM':
            #         # Random uniform distribution initializer doesn't need stddev:
            #         initial_value = self.initializer(
            #             shape=[bottleneck_tensor_size, n_outputs]
            #         )
            #     elif 'he_normal' in str(self.initializer) or 'init_ops.VarianceScaling' in str(self.initializer):
            #     # elif self.initializer_repr == 'INIT_HE_NORMAL':
            #         # He normal initializer doesn't need a stddev:
            #         initial_value = self.initializer(
            #             shape=[bottleneck_tensor_size, n_outputs]
            #         )
            #     # elif self.initializer_repr == 'INIT_HE_UNIFORM':
            #     elif 'he_uniform' in str(self.initializer):
            #         initial_value = self.initializer()(shape=[bottleneck_tensor_size, n_outputs])
            #     else:
            #         initial_value = self.initializer(
            #             shape=[bottleneck_tensor_size, n_outputs],
            #             stddev=0.001
            #         )
            #     # Output random values from truncated normal distribution:
            #     # initial_value = tf.truncated_normal(
            #     #     shape=[bottleneck_tensor_size, n_outputs],
            #     #     stddev=0.001
            #     # )
            #     layer_weights = tf.Variable(initial_value=initial_value, name='final_weights')

            # with tf.name_scope('biases'):
            #     layer_biases = tf.Variable(initial_value=tf.zeros([n_outputs]), name='final_biases')

            # # pre-activations:
            # with tf.name_scope('Wx_plus_b'):
            #     logits = tf.matmul(X, layer_weights) + layer_biases
            #     # For TensorBoard histograms:
            #     tf.summary.histogram('Wx_plus_b', logits)


        # logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer=he_init, name="logits")
        Y_proba = tf.nn.softmax(logits, name="Y_proba")

        # The logic here is for computing a running accuracy on a class-by-class basis:
        # Create a tensor containing the predicted class label for each training sample (the argmax of the probability tensor)
        preds = tf.math.argmax(Y_proba, axis=1)
        # Create a confusion matrix:
        confusion_matrix = tf.confusion_matrix(y, predictions, num_classes=num_classes, dtype=tf.float32, name='BatchConfusionMatrix')
        # Create an accumulator variable to hold the counts:
        # confusion = tf.Variable(tf.zeros([len(self.classes_), len(self.classes_)], dtype=tf.float32), name='ConfusionMatrixAccumulator')
        # print(confusion)
        # Create an update op for doing a += accumulation on the batch:
        # confusion_update = confusion.assign(confusion + batch_confusion)
        # Sanity check:
        # acc = tf.reduce_mean(confusion_matrix)
        # per-class acc:
        # per_class_acc = tf.reduce_mean(batch_confusion, axis=1)
        # tf.logging.info('per_class_acc: %s' % per_class_acc)

        # Create a tensor to store a running counter of the number of times each class was chosen as the target:
        # class_counts = tf.zeros(shape=[len(self.classes_)], dtype=tf.float32)
        # Create a tensor to store a running counter of the number of times each class was predicted correctly:
        # class_counts_correct = tf.zeros(shape=[len(self.classes_)], dtype=tf.float32)

        # # Create a boolean tensor containing values of True where the predicted label matches the class label:
        # is_correct = tf.math.equal(y, preds)
        # # Convert this tensor to zero's and one's and then update the class_counts tensor
        # is_correct_as_int = tf.cast(is_correct, tf.float32)
        # for i in range(class_counts.shape[0]):
        #     class_counts[i] = sum(tf.cast(tf.equal(preds, i), tf.float32))
        # class_counts_correct = class_counts_correct + is_correct_as_int


        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                  logits=logits)
        tf.summary.histogram('xentropy', xentropy)

        loss = tf.reduce_mean(xentropy, name="loss")
        tf.summary.scalar('loss', loss)

        # optim_class_repr = str(self.optimizer)
        # if 'momentum.MomentumOptimizer' in optim_class_repr:
        #     # optimizer = self.optimizer(learning_rate=self.learning_rate, momentum=)
        #     # Optimizer is an already instantiated MomentumOptimizer, do not attempt to re-instantiate:
        #     optimizer = self.optimizer
        # elif 'adagrad.AdagradOptimizer' in optim_class_repr:
        #     # Optimizer is an already instantiated AdagradOptimizer, do not attempt to re-instantiate:
        #     optimizer = self.optimizer
        # elif 'adadelta.AdadeltaOptimizer' in optim_class_repr:
        #     # Optimizer is an already instantiated AdadeltaOptimizer, do not attempt to re-instantiate:
        #     optimizer = self.optimizer
        # else:
        #     optimizer = self.optimizer(learning_rate=self.learning_rate)
        training_op = self.optimizer.minimize(loss)

        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
        tf.summary.scalar('accuracy', accuracy)

        top5_pred= tf.nn.in_top_k(predictions=Y_proba, targets=y, k=5)
        top5_acc = tf.reduce_mean(tf.cast(top5_pred, tf.float32))
        tf.summary.scalar('top5_accuracy', top5_acc)

        # Shape mis-match incompatible shapes: [5850] vs. [650]
        # per_class_acc, per_class_acc_update = tf.metrics.mean_per_class_accuracy(y, predictions, len(self.classes_), name='PerClassAcc')
        # tf.summary.scalar('per_class_acc', per_class_acc)

        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='PerClassAcc')
        running_vars_init = tf.variables_initializer(var_list=running_vars)

        # init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        init = tf.global_variables_initializer()

        # Merge all tensorboard summaries into one object:
        tb_merged_summaries = tf.summary.merge_all()

        # Create a saver for checkpoint file creation and restore:
        train_saver = tf.train.Saver()

        # Make the important operations available easily through instance variables
        self._X, self._y, self._Y_proba, self._predictions = X, y, Y_proba, predictions
        self._Y_proba, self._preds, self._loss = Y_proba, preds, loss
        self._training_op, self._accuracy, self._top_five_acc = training_op, accuracy, top5_acc
        # self._per_class_acc_update_op = per_class_acc_update
        self.num_classes, self.confusion_matrix = num_classes, confusion_matrix
        self._init, self.running_vars_init, self._train_saver = init, running_vars_init, train_saver
        self._merged = tb_merged_summaries

    def close_session(self):
        if self._session:
            self._session.close()

    def _get_model_params(self):
        """Get all variable values (used for early stopping, faster than saving to disk)"""
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}

    def _restore_model_params(self, model_params):
        """Set all variables to the given values (for early stopping, faster than loading from disk)"""
        gvar_names = list(model_params.keys())
        assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign")
                      for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)

    # def export_model(self, module_spec, class_count, saved_model_dir):
    #     """
    #     Exports a trained model for use with TensorFlow serving.
    #
    #     Args:
    #       module_spec: The hub.ModuleSpec for the image module being used.
    #       class_count: The number of classes.
    #       saved_model_dir: Directory in which to save exported model and variables.
    #     """
    #     # The SavedModel should hold the eval graph.
    #     eval_sess, resized_input_image, bottleneck_input, ground_truth_input, acc_evaluation_step, \
    #         top5_acc_eval_step, prediction = _build_eval_session(module_spec, class_count)
    #     graph = eval_sess.graph
    #     with graph.as_default():
    #         inputs = {'image': tf.saved_model.utils.build_tensor_info(resized_input_image)}
    #
    #         out_classes = eval_sess.graph.get_tensor_by_name('retrain_ops/final_result:0')
    #         outputs = {
    #             'prediction': tf.saved_model.utils.build_tensor_info(out_classes)
    #         }
    #
    #         signature = tf.saved_model.signature_def_utils.build_signature_def(
    #             inputs=inputs,
    #             outputs=outputs,
    #             method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
    #
    #         legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    #
    #         # Save out the SavedModel.
    #         builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
    #         builder.add_meta_graph_and_variables(
    #             eval_sess, [tf.saved_model.tag_constants.SERVING],
    #             signature_def_map={
    #                 tf.saved_model.signature_constants.
    #                     DEFAULT_SERVING_SIGNATURE_DEF_KEY:
    #                     signature
    #             },
    #             legacy_init_op=legacy_init_op)
    #         builder.save()

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

    # def get_hyperparameter_string(self):
    #     if self.initializer == tf.random_normal:
    #         hyper_string = 'INIT_rand_norm,'
    #     elif self.initializer == tf.random_uniform:
    #         hyper_string = 'INIT_rand_unif,'
    #     elif self.initializer == tf.truncated_normal:
    #         hyper_string = 'INIT_trunc_norm,'
    #     elif self.initializer == tf.initializers.he_normal:
    #     # elif 'tf.python.ops.init_ops.VarianceScaling' in str(self.initializer):
    #         hyper_string = 'INIT_he_norm,'
    #     else:
    #         hyper_string = 'INIT_unknown,'
    #     return hyper_string

    def print_multiclass_acc(self, y, y_pred):
        """
        print_multiclass_acc: Prints the accuracy for each class in human readable form
        :param y:
        :param y_pred:
        :return:
        """
        cm = pycm.ConfusionMatrix(actual_vector=y, predict_vector=y_pred)
        out = '\t{'
        for clss, acc in cm.ACC.items():
            out = out + '\'%s\': %.2f%%,' % (self.class_labels[int(clss)], acc*100)
        out = out[:-1]
        out = out + '}'
        print(out)


    def fit(self, X, y, n_epochs=100, X_valid=None, y_valid=None, eval_freq=1, ckpt_freq=1):
        """
        fit: Fits the model to the training data.
        :param X:
        :param y:
        :param n_epochs: The number of passes over the entire training dataset during fitting.
        :param X_valid:
        :param y_valid:
        :param eval_freq: How many epochs to train for, before running and displaying the results of an evaluation step.
        :param ckpt_freq: How many epochs to train for, before saving a model checkpoint.
        :return:
        """
        """Fit the model to the training set. If X_valid and y_valid are provided, use early stopping."""
        self.close_session()
        if self.train_batch_size == -1:
            self.train_batch_size = len(X)
        if self.val_batch_size == -1:
            self.val_batch_size = len(X_valid)

        # TB TrainWriter logging directory:
        tb_log_path_train = self.tb_logdir + '/train/' + self.__repr__()
        # TB ValidationWriter logging directory:
        tb_log_path_val = self.tb_logdir + '/val/' + self.__repr__()

        # If this is a refit of an existing hyperparameter combination, purge TB logging directories:
        if self.refit:
            # Remove the TB logging directory from the first training fit with these parameters:
            shutil.rmtree(tb_log_path_train, ignore_errors=True)
            # Remove the TB logging directory from the first validation fit with these parameters:
            shutil.rmtree(tb_log_path_val, ignore_errors=True)
            # Re-create both directories:
            # os.mkdir(tb_log_path_train)
            # os.mkdir(tb_log_path_val)

        # infer n_inputs and n_outputs from the training set.
        n_inputs = X.shape[1]
        self.classes_ = np.unique(y)
        # self._num_classes = len(self.classes_)
        n_outputs = len(self.classes_)
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph(n_inputs, n_outputs)
            # extra ops for batch normalization
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # needed in case of early stopping
        max_checks_without_progress = 20
        checks_without_progress = 0
        best_loss = np.infty
        best_params = None

        # Now train the model!
        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            # self._hyper_string = str(self._get_model_params())
            self._train_writer = tf.summary.FileWriter(tb_log_path_train, sess.graph)
            self._val_writer = tf.summary.FileWriter(tb_log_path_val)
            self._init.run()
            self.running_vars_init.run()
            for epoch in range(n_epochs):
                is_last_step = (epoch + 1 == n_epochs)
                # Minibatch partitioning:
                rnd_idx = np.random.permutation(len(X))
                for rnd_indices in np.array_split(rnd_idx, len(X) // self.train_batch_size):
                    X_batch, y_batch = X[rnd_indices], y[rnd_indices]

                    # Run a training step and capture the results in self._training_op
                    train_summary, _ = sess.run([self._merged, self._training_op], feed_dict={self._X: X_batch, self._y: y_batch})

                    # Export the results to the TensorBoard logging directory:
                    self._train_writer.add_summary(train_summary, epoch)

                    # Check to see if a checkpoint should be recorded this epoch:
                    if ckpt_freq != 0 and epoch > 0 and (epoch % ckpt_freq == 0) or is_last_step:
                        tf.logging.info(msg='Writing checkpoint (model snapshot) to \'%s\'' % self.ckpt_dir)
                        self._train_saver.save(sess, self.ckpt_dir)
                        # Constant OP for tf.Serving export code goes here:
                        # tf.logging.info(msg='Writing computational graph with constant-op conversion to \'%s\'' % self.tb_logdir)
                        # intermediate_file_name = (self.ckpt_dir + 'intermediate_' + str(epoch) + '.pb')
                        # self.save_graph_to_file(graph_file_name=intermediate_file_name, module_spec=self._module_spec, class_count=n_outputs)

                if X_valid is not None and y_valid is not None:
                    # Run eval metrics, and write the result.
                    val_summary, loss_val, acc_val, top5_acc, preds = sess.run(
                        [self._merged, self._loss, self._accuracy, self._top_five_acc, self._preds],
                        feed_dict={self._X: X_valid, self._y: y_valid}
                    )
                    # Compute confusion matrix for multiclass accuracy:
                    # confusion_matrix = sess.run(self.confusion_matrix, feed_dict={self._y: y_valid, self._predictions: preds, self.num_classes: len(self.classes_)})
                    # cm = pycm.ConfusionMatrix(actual_vector=y_valid, predict_vector=preds)
                    # tf.summary.histogram(cm.ACC, 'multiclassAcc')
                    self.print_multiclass_acc(y=y_valid, y_pred=preds)
                    # true_positives = confusion_matrix.diagonal()
                    # print()
                    # print(confusion_matrix)
                    # print('preds: %s' % preds)
                    # sess.run(self.batch_confusion, feed_dict={self._y: y_valid, self._predictions: preds, self.num_classes: len(self.classes_)})
                    # per_class_acc = sess.run(self.per_class_acc, feed_dict={self._y: y_valid, self._predictions: preds, self.num_classes: len(self.classes_)})
                    # sess.run(self._per_class_acc_update_op, feed_dict={self._y: y_valid, self._predictions: preds})
                    # per_class_acc = sess.run(self.per_class_acc, feed_dict={self._y: y_valid, self._predictions: preds})
                    # print('per_class_acc: %s' % per_class_acc)
                    self._val_writer.add_summary(val_summary, epoch)
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
                        break
                else:
                    # Report on training accuracy since no validation dataset
                    if (epoch % eval_freq) == 0 or is_last_step:
                        loss_train, acc_train = sess.run([self._loss, self._accuracy],
                                                         feed_dict={self._X: X_batch,
                                                                    self._y: y_batch})
                        print("{}\tLast training batch loss: {:.6f}\tAccuracy: {:.2f}%".format(
                            epoch, loss_train, acc_train * 100))

            # If we used early stopping then rollback to the best model found
            if best_params:
                self._restore_model_params(best_params)

            # Export the trained model for use with serving:
            # export_model(module_spec=self._module_spec, class_count=n_outputs, saved_model_dir=self.saved_model_dir)

            return self

    def predict_proba(self, X):
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self._Y_proba.eval(feed_dict={self._X: X})

    def predict(self, X):
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return np.array([[self.classes_[class_index]]
                         for class_index in class_indices], np.int32)

    def save(self, path):
        self._train_saver.save(self._session, path)

    def __repr__(self):
        tfh_repr = '%s,%s' % (self._get_initializer_repr(self.initializer), self._get_optimizer_repr(self.optimizer))
        # tfh_repr = '%s,' % str(self.initializer)
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
#     X_train1 = X_train[y_train < 5]
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
