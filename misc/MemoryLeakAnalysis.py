import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D

print('TensorFlow Version: %s' % tf.VERSION)
print('tf.keras Version: %s' % tf.keras.__version__)

tf.logging.set_verbosity(tf.logging.INFO)

num_params = 100
num_classes = 1000
train_from_bottlenecks = True
activations = ['elu', 'relu', 'tanh']

for i in range(num_params):
    tf.logging.warning('Cleared Keras\' back-end session.')
    tf.keras.backend.clear_session()

    current_activation = activations[i % len(activations)]
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))

    for layer in base_model.layers:
        layer.trainable = False

    if not train_from_bottlenecks:
        x = base_model.output
        bottlenecks = GlobalAveragePooling2D()(x)
        logits = Dense(num_classes, activation=current_activation, name='logits')(bottlenecks)
        y_proba = Dense(num_classes, activation='softmax')(logits)
        _keras_model = Model(inputs=base_model.input, outputs=y_proba)
    else:
        bottlenecks = Input(shape=(base_model.output_shape[-1],), name='bottleneck')
        # bottlenecks = Dense(self.num_classes, input_shape=(base_model.output_shape[-1],))
        logits = Dense(num_classes, activation=current_activation, name='logits')(bottlenecks)
        y_proba = Dense(num_classes, activation='softmax', name='y_proba')(logits)
        # This is the model that is actually trained, if bottlenecks are being fed from memory:
        _keras_model = Model(inputs=bottlenecks, outputs=y_proba)
