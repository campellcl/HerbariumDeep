"""
CustomCallbacks.py
"""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard


class FileWritersTensorBoardCallback(TensorBoard):
    """
    FileWritersTensorBoardCallback
    :source url: https://stackoverflow.com/a/48393723/3429090
    """
    train_log_dir = None
    val_log_dir = None
    val_writer = None
    train_writer = None

    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original 'TensorBoard' log to a subdirectory 'train'
        self.train_log_dir = os.path.join(log_dir, 'train')
        super(FileWritersTensorBoardCallback, self).__init__(self.train_log_dir, **kwargs)
        # Log validation metrics to val dir:
        self.val_log_dir = os.path.join(log_dir, 'val')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        # Call to super to setup training metrics writer:
        super(FileWritersTensorBoardCallback, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass
