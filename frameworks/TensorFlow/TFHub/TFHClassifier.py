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
    def __init__(self, class_labels, optimizer=tf.train.AdamOptimizer, train_batch_size=-1, val_batch_size=-1,
                 activation=tf.nn.elu, initializer=he_init,
                 batch_norm_momentum=None, dropout_rate=None, random_state=None, tb_logdir='tmp\\summaries\\',
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
        self.classes_ = None
        # Create a FileWriter object to export tensorboard information:
        self._train_writer = None
        self._val_writer = None
        ''' TensorBoard Related Variables: '''
        self.ckpt_dir = ckpt_dir
        self.saved_model_dir = saved_model_dir
        self.tb_logdir = tb_logdir
        self.refit = refit

    def _build_graph(self, n_inputs, n_outputs):
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

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

        Y_proba = tf.nn.softmax(logits, name="Y_proba")

        ''' The logic here is for computing a running accuracy on a class-by-class basis: '''
        # Create a tensor containing the predicted class label for each training sample (the argmax of the probability tensor)
        preds = tf.math.argmax(Y_proba, axis=1)
        # Create a confusion matrix:
        # confusion_matrix = tf.confusion_matrix(y, predictions, num_classes=num_classes, dtype=tf.float32, name='confusion_matrix')
        # tf.summary.tensor_summary('confusion_matrix', confusion_matrix)

        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                  logits=logits)
        tf.summary.histogram('xentropy', xentropy)

        loss = tf.reduce_mean(xentropy, name="loss")
        tf.summary.scalar('loss', loss)

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

        # init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        init = tf.global_variables_initializer()

        # Merge all tensorboard summaries into one object:
        tb_merged_summaries = tf.summary.merge_all()

        # Create a saver for checkpoint file creation and restore:
        train_saver = tf.train.Saver()

        # Make the important operations available easily through instance variables:
        self._X, self._y, self._Y_proba, self._predictions = X, y, Y_proba, predictions
        self._Y_proba, self._preds, self._loss = Y_proba, preds, loss
        self._training_op, self._accuracy, self._top_five_acc = training_op, accuracy, top5_acc
        # self._per_class_acc_update_op = per_class_acc_update
        self.num_classes = num_classes
        self._init, self._train_saver = init, train_saver
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

    def fit(self, X, y, n_epochs=100, X_valid=None, y_valid=None, eval_freq=1, ckpt_freq=1):
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
        :return:
        """
        ''' Generic cleanup, and some lazy instantiation: '''
        # Close the session in case it was previously left open for some reason:
        self.close_session()

        # If the batch size was set to -1 during initialization, infer the training size at runtime from the X matrix:
        if self.train_batch_size == -1:
            self.train_batch_size = len(X)
        if self.val_batch_size == -1:
            self.val_batch_size = len(X_valid)

        # infer n_inputs and n_outputs from the training set:
        n_inputs = X.shape[1]
        self.classes_ = np.unique(y)
        n_outputs = len(self.classes_)

        # needed in case of early stopping:
        max_checks_without_progress = 20
        checks_without_progress = 0
        best_loss = np.infty
        best_params = None

        '''Some TensorBoard setup: '''
        # TB TrainWriter logging directory:
        if not self.refit:
            tb_log_dir_train = os.path.join(self.tb_logdir, 'gs')
            tb_log_dir_train = os.path.join(tb_log_dir_train, 'train')
        else:
            tb_log_dir_train = os.path.join(self.tb_logdir, 'train')
        tb_log_dir_train = os.path.join(tb_log_dir_train, self.__repr__())
        # TB ValidationWriter logging directory:
        if not self.refit:
            tb_log_dir_val = os.path.join(self.tb_logdir, 'gs')
            tb_log_dir_val = os.path.join(tb_log_dir_val, 'val')
        else:
            tb_log_dir_val = os.path.join(self.tb_logdir, 'val')
        tb_log_dir_val = os.path.join(tb_log_dir_val, self.__repr__())

        ''' Build the computational graph: '''
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph(n_inputs, n_outputs)
            # extra ops for batch normalization
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        ''' Train the model! '''
        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            self._train_writer = tf.summary.FileWriter(tb_log_dir_train, sess.graph)
            self._val_writer = tf.summary.FileWriter(tb_log_dir_val)
            self._init.run()
            # self.running_vars_init.run()

            for epoch in range(n_epochs):
                is_last_step = (epoch + 1 == n_epochs)
                # batch partitioning:
                for X_batch, y_batch in self._shuffle_batch(X, y, batch_size=self.train_batch_size):
                    # Run a training step, but don't capture the results at the mini-batch level:
                    # _ = sess.run(self._training_op, feed_dict={self._X: X_batch, self._y: y_batch})
                    # Run a training step and capture the results in self._training_op
                     _ = sess.run(self._training_op, feed_dict={self._X: X_batch, self._y: y_batch})

                    # Export the results to the TensorBoard logging directory:
                    # self._train_writer.add_summary(train_summary, epoch)

                # Check to see if eval metrics should be computed this epoch:
                if (epoch % eval_freq == 0) or is_last_step:
                    train_summary, train_preds = sess.run([self._merged, self._preds], feed_dict={self._X: X, self._y: y})
                    self._train_writer.add_summary(train_summary, epoch)

                    if X_valid is not None and y_valid is not None:
                        # Run eval metrics on the entire validation dataset:
                        val_summary, loss_val, acc_val, top5_acc, val_preds = sess.run(
                            [self._merged, self._loss, self._accuracy, self._top_five_acc, self._preds],
                            feed_dict={self._X: X_valid, self._y: y_valid}
                        )

                        if is_last_step:
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
                            break
                    else:
                        # Run eval metrics on the entire training dataset (as no validation set is available):
                        loss_train, acc_train = sess.run([self._loss, self._accuracy],
                                                             feed_dict={self._X: X,
                                                                        self._y: y})
                        # print("{}\tLast training batch loss: {:.6f}\tAccuracy: {:.2f}%".format(
                        #     epoch, loss_train, acc_train * 100))
                        print("{}\tLoss on all of X_train: {:.6f}\tAccuracy: {:.2f}%".format(
                            epoch, loss_train, acc_train * 100))

                # Check to see if a checkpoint should be recorded this epoch:
                if ckpt_freq != 0 and epoch > 0 and ((epoch % ckpt_freq == 0) or is_last_step):
                    tf.logging.info(msg='Writing checkpoint (model snapshot) to \'%s\'' % self.ckpt_dir)
                    self._train_saver.save(sess, self.ckpt_dir)
                    # Constant OP for tf.Serving export code of stand-alone model goes here:
                    # tf.logging.info(msg='Writing computational graph with constant-op conversion to \'%s\'' % self.tb_logdir)
                    # intermediate_file_name = (self.ckpt_dir + 'intermediate_' + str(epoch) + '.pb')
                    # self.save_graph_to_file(graph_file_name=intermediate_file_name, module_spec=self._module_spec, class_count=n_outputs)

            # If we used early stopping then rollback to the best model found
            if best_params:
                # print('Restoring model to best parameter set: %s' % best_params)
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
