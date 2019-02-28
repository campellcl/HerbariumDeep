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
    counter = None

    def __init__(self, hyperparameter_string_repr, log_dir='./logs', **kwargs):
        self.hyperparameter_string_repr = hyperparameter_string_repr
        # Make the original 'TensorBoard' log to a subdirectory 'train'
        self.train_log_dir = os.path.join(log_dir, 'train')
        self.train_log_dir = os.path.join(self.train_log_dir, self.hyperparameter_string_repr)
        super(FileWritersTensorBoardCallback, self).__init__(self.train_log_dir, **kwargs)
        # Log validation metrics to val dir:
        self.val_log_dir = os.path.join(log_dir, 'val')
        self.val_log_dir = os.path.join(self.val_log_dir, self.hyperparameter_string_repr)
        self.counter = 0

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        # Call to super to setup training metrics writer:
        super(FileWritersTensorBoardCallback, self).set_model(model)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', 'batch_'): v for k, v in logs.items() if k.startswith('val_')}
        self.counter += 1
        for name, value in val_logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, self.counter)
        self.val_writer.flush()
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(FileWritersTensorBoardCallback, self).on_batch_end(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        #  Grab the val_writer logs. Rename the keys so that they can be plotted on the same figure with training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', 'epoch_'): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining train_writer logs to `TensorBoard.on_epoch_end`:
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(FileWritersTensorBoardCallback, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(FileWritersTensorBoardCallback, self).on_train_end(logs)
        self.val_writer.close()
