from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

he_init = tf.variance_scaling_initializer()


class HandsOnMLClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, optimizer_class=tf.train.AdamOptimizer, initializer=he_init, learning_rate=0.01, random_state=None):
        self.optimizer_class = optimizer_class
        self.initializer_class = initializer
        self.random_state = random_state
        self.learning_rate = learning_rate
        self._train_graph = None
        self._train_session = None
        self.bottleneck_input = None
        self.ground_truth_input = None
        self.loss = None
        self.accuracy = None
        self.training_op = None
        self.extra_update_ops = None
        self._init = None

    def _build_train_graph(self, n_classes):
        self._train_graph = tf.Graph()
        with self._train_graph.as_default():
            # Load module spec/blueprint:
            tfhub_module_spec = hub.load_module_spec('https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1')
            height, width = hub.get_expected_image_size(tfhub_module_spec)

            # Create a placeholder tensor for image input to the model (when bottleneck has not been pre-computed).
            resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3], name='resized_input')

            # m_reg = hub.Module(tfhub_module_spec, trainable=True, tags={'train'})
            m_reg = hub.Module(tfhub_module_spec, trainable=False, tags={'train'})
            # m = hub.Module(tfhub_module_spec)

            # Create a placeholder tensor to catch the output of the pre-activation layer:
            # bottleneck_tensor = m(resized_input_tensor)
            # bottleneck_tensor_reg = m_reg(resized_input_tensor)
            bottleneck_tensor_reg = m_reg()

            batch_size, bottleneck_tensor_size = bottleneck_tensor_reg.get_shape().as_list()
            self.bottleneck_input = tf.placeholder_with_default(
                bottleneck_tensor_reg,
                shape=[batch_size, bottleneck_tensor_size],
                name='BottleneckInputPlaceholder'
            )
            self.ground_truth_input = tf.placeholder(
                tf.int64, [batch_size], name='GroundTruthInput'
            )
            with tf.name_scope('weights'):
                initial_value = tf.variance_scaling_initializer()(shape=[bottleneck_tensor_size, n_classes])
                layer_weights = tf.Variable(initial_value=initial_value, name='final_weights')
            with tf.name_scope('biases'):
                layer_biases = tf.Variable(initial_value=tf.zeros([n_classes]), name='final_biases')
            with tf.name_scope('Wx_plus_b'):
                logits = tf.matmul(self.bottleneck_input, layer_weights) + layer_biases
            Y_proba = tf.nn.softmax(logits, name='Y_proba')
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ground_truth_input, logits=logits)
            self.loss = tf.reduce_mean(xentropy, name='xentropy_loss')
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.training_op = optimizer.minimize(self.loss)
            correct = tf.nn.in_top_k(logits, self.ground_truth_input, 1)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
            # Declare initializer:
            self._init = tf.global_variables_initializer()
            # extra ops for batch normalization
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def fit(self, X, y, n_epochs=10, X_valid=None, y_valid=None, ckpt_freq=0):
        n_classes = X.shape[1]
        self._build_train_graph(n_classes=n_classes)
        self._train_session = tf.Session(graph=self._train_graph)
        with self._train_session as sess:
            self._init.run(session=sess)
            for epoch in range(100):
                sess.run([self.training_op], feed_dict={self.bottleneck_input: X, self.ground_truth_input: y})
                # if self.extra_update_ops:
                #     sess.run([self.extra_update_ops], feed_dict={self.bottleneck_input: X, self.ground_truth_input: y})

                # if X_valid and y_valid:
                #     loss_val, acc_val = sess.run([self.loss, self.accuracy],
                #                                  feed_dict={
                #                                      self.bottleneck_input: X_valid,
                #                                      self.ground_truth_input: y_valid
                #                                  })

        return self

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
