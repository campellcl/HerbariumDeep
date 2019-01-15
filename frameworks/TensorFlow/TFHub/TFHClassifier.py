from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os

he_init = tf.variance_scaling_initializer()


class TFHClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, optimizer_class=tf.train.AdamOptimizer,
                 learning_rate=0.01, batch_size=20, activation=tf.nn.elu, initializer=he_init,
                 batch_norm_momentum=None, dropout_rate=None, random_state=None, tb_logdir=None,
                 ckpt_dir=None, saved_model_dir=None):
        """
        __init__: Initializes the TensorFlow Hub Classifier (TFHC) by storing all hyperparameters.
        :param optimizer_class: The type of optimizer to use during training (AdamOptimizer by default)
        :param learning_rate:
        :param batch_size:
        :param activation:
        :param initializer:
        :param batch_norm_momentum:
        :param dropout_rate:
        :param random_state:
        :param tb_logdir: <str> The directory to export training and validation summaries to for tensorboard analysis.
        :param ckpt_dir: <str> The directory to export model snapshots (checkpoints) during training to.
        """
        """Initialize the DNNClassifier by simply storing all the hyperparameters."""
        self._module_spec = None
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation = activation
        self.initializer = initializer
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self._session = None
        # Create a FileWriter object to export tensorboard information:
        self._train_writer = None
        # TensorBoard directory assignments:
        if tb_logdir is None:
            self.tb_logdir = 'tmp/summaries/'
        if ckpt_dir is None:
            self.ckpt_dir = 'tmp/'
        if saved_model_dir is None:
            self.saved_model_dir = 'tmp/trained_model/'

    def _build_graph(self, n_inputs, n_outputs):
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

        ''' Add transfer learning target domain final retrain operations: '''
        final_layer_name = 'final_retrain_ops'
        with tf.variable_scope(final_layer_name):
            # The final layer of target domain re-train Operations is composed of the following:
            with tf.name_scope('weights'):
                # Output random values from truncated normal distribution:
                initial_value = tf.truncated_normal(
                    shape=[bottleneck_tensor_size, n_outputs],
                    stddev=0.001
                )
                layer_weights = tf.Variable(initial_value=initial_value, name='final_weights')

            with tf.name_scope('biases'):
                layer_biases = tf.Variable(initial_value=tf.zeros([n_outputs]), name='final_biases')

            # pre-activations:
            with tf.name_scope('Wx_plus_b'):
                logits = tf.matmul(X, layer_weights) + layer_biases
                # For TensorBoard histograms:
                tf.summary.histogram('Wx_plus_b', logits)

        # logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer=he_init, name="logits")
        Y_proba = tf.nn.softmax(logits, name="Y_proba")

        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                  logits=logits)
        tf.summary.histogram('xentropy', xentropy)

        loss = tf.reduce_mean(xentropy, name="loss")
        tf.summary.scalar('loss', loss)

        optimizer = self.optimizer_class(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(loss)

        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
        tf.summary.scalar('accuracy', accuracy)

        init = tf.global_variables_initializer()

        # Merge all tensorboard summaries into one object:
        tb_merged_summaries = tf.summary.merge_all()

        # Create a saver for checkpoint file creation and restore:
        train_saver = tf.train.Saver()

        # Make the important operations available easily through instance variables
        self._X, self._y = X, y
        self._Y_proba, self._loss = Y_proba, loss
        self._training_op, self._accuracy = training_op, accuracy
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
    #     """Exports model for serving.
    #
    # Args:
    #   module_spec: The hub.ModuleSpec for the image module being used.
    #   class_count: The number of classes.
    #   saved_model_dir: Directory in which to save exported model and variables.
    # """
    #     # The SavedModel should hold the eval graph.
    #     eval_sess, resized_input_image, bottleneck_input, ground_truth_input, acc_evaluation_step, \
    #         top5_acc_eval_step, prediction = build_eval_session(module_spec, class_count)
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

    def fit(self, X, y, n_epochs=10, X_valid=None, y_valid=None, eval_freq=1, ckpt_freq=1):
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

        # infer n_inputs and n_outputs from the training set.
        n_inputs = X.shape[1]
        self.classes_ = np.unique(y)
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
            self._train_writer = tf.summary.FileWriter(self.tb_logdir + '/train', sess.graph)
            self._init.run()
            for epoch in range(n_epochs):
                is_last_step = (epoch + 1 == n_epochs)
                # Run a training step and capture the results in self._training_op
                train_summary, _ = sess.run([self._merged, self._training_op], feed_dict={self._X: X, self._y: y})
                # Export the results to the TensorBoard logging directory:
                self._train_writer.add_summary(train_summary, epoch)

                # Check to see if a checkpoint should be recorded this epoch:
                if epoch > 0 and (epoch % ckpt_freq == 0) or is_last_step:
                    tf.logging.info(msg='Writing checkpoint (model snapshot) to \'%s\'' % self.ckpt_dir)
                    self._train_saver.save(sess, self.ckpt_dir)
                    # tf.logging.info(msg='Writing computational graph with constant-op conversion to \'%s\'' % self.tb_logdir)
                    # intermediate_file_name = (self.ckpt_dir + 'intermediate_' + str(epoch) + '.pb')
                    # self.save_graph_to_file(graph_file_name=intermediate_file_name, module_spec=self._module_spec, class_count=n_outputs)

                if X_valid is not None and y_valid is not None:
                    loss_val, acc_val = sess.run([self._loss, self._accuracy],
                                                 feed_dict={self._X: X_valid,
                                                            self._y: y_valid})
                    if loss_val < best_loss:
                        best_params = self._get_model_params()
                        best_loss = loss_val
                        checks_without_progress = 0
                    else:
                        checks_without_progress += 1
                    print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
                        epoch, loss_val, best_loss, acc_val * 100))
                    if checks_without_progress > max_checks_without_progress:
                        print("Early stopping!")
                        break
                else:
                    if (epoch % eval_freq) == 0 or is_last_step:
                        loss_train, acc_train = sess.run([self._loss, self._accuracy],
                                                         feed_dict={self._X: X,
                                                                    self._y: y})
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


if __name__ == '__main__':
    n_inputs = 28 * 28 # MNIST
    n_outputs = 5
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    X_valid, X_train = X_train[:5000], X_train[5000:]
    y_valid, y_train = y_train[:5000], y_train[5000:]
    X_train1 = X_train[y_train < 5]
    y_train1 = y_train[y_train < 5]
    X_valid1 = X_valid[y_valid < 5]
    y_valid1 = y_valid[y_valid < 5]
    X_test1 = X_test[y_test < 5]
    y_test1 = y_test[y_test < 5]
    dnn_clf = TFHClassifier(random_state=42)
    dnn_clf.fit(X_train1, y_train1, n_epochs=10, X_valid=X_valid1, y_valid=y_valid1)
    y_pred = dnn_clf.predict(X_test1)
    # accuracy_score(y_test1, y_pred)
    print(accuracy_score(y_test1, y_pred))
