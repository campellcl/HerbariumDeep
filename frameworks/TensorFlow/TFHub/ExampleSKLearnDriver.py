from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
from frameworks.TensorFlow.TFHub.ExampleSKLearnDNN import DNNClassifier

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
    dnn_clf = DNNClassifier(random_state=42)
    dnn_clf.fit(X_train1, y_train1, n_epochs=10, X_valid=X_valid1, y_valid=y_valid1)
    y_pred = dnn_clf.predict(X_test1)
    # accuracy_score(y_test1, y_pred)
    print(accuracy_score(y_test1, y_pred))
