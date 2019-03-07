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

    @staticmethod
    def _ensure_dir_read_write_exist(dir):
        if os.path.exists(dir):
            if os.access(dir, os.R_OK) and os.access(dir, os.W_OK):
                return True
            else:
                raise PermissionError('The provided directory: \'%s\' is not readable or writeable.' % dir)
        else:
            raise NotADirectoryError('The provided directory: \'%s\' does not exist.' % dir)

    def __init__(self, hyperparameter_string_repr, write_freq, is_refit, log_dir='./logs', **kwargs):
        self.hyperparameter_string_repr = hyperparameter_string_repr
        self.write_freq = write_freq
        # If this is a refit of an existing hyperparameter set by sklearn, then create a new relative directory:
        if is_refit:
            # Make the original 'TensorBoard' log to a subdirectory 'train'
            self.train_log_dir = os.path.join(log_dir, 'train')
            # Custom implementation for val logging:
            self.val_log_dir = os.path.join(log_dir, 'val')
        else:
            # Not a refit operation, this is part of the grid search:
            self.train_log_dir = os.path.join(log_dir, 'gs')
            self.train_log_dir = os.path.join(self.train_log_dir, 'train')
            self.val_log_dir = os.path.join(log_dir, 'gs')
            self.val_log_dir = os.path.join(self.val_log_dir, 'val')

        self.train_log_dir = os.path.join(self.train_log_dir, self.hyperparameter_string_repr)
        super(FileWritersTensorBoardCallback, self).__init__(self.train_log_dir, **kwargs)
        self.val_log_dir = os.path.join(self.val_log_dir, self.hyperparameter_string_repr)
        self.counter = 0

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        # Call to super to setup training metrics writer:
        super(FileWritersTensorBoardCallback, self).set_model(model)

    # def _is_logging(self, epoch, batch=None):
    #     is_first_epoch = (epoch == 0)
    #     if is_first_epoch:
    #         return True
    #     elif (self.counter % self.write_freq) == 0:
    #         return True
    #     else:
    #         return False

    # def on_batch_end(self, batch, logs=None):
    #     logs = logs or {}
    #     val_logs = {k.replace('val_', 'batch_'): v for k, v in logs.items() if k.startswith('val_')}
    #     self.counter += 1
    #     for name, value in val_logs.items():
    #         if name in ['batch', 'size']:
    #             continue
    #         summary = tf.Summary()
    #         summary_value = summary.value.add()
    #         summary_value.simple_value = value.item()
    #         summary_value.tag = name
    #         self.val_writer.add_summary(summary, self.counter)
    #     self.val_writer.flush()
    #     logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
    #     super(FileWritersTensorBoardCallback, self).on_batch_end(batch, logs)

    def on_epoch_begin(self, epoch, logs=None):
        super(FileWritersTensorBoardCallback, self).on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        is_first_epoch = (epoch == 0)
        if (self.counter % self.write_freq == 0) or is_first_epoch:
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
        self.counter += 1

    def on_train_begin(self, logs=None):
        super(FileWritersTensorBoardCallback, self).on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', 'epoch_'): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, self.counter)
        self.val_writer.flush()
        logs = {k: v for k, v in logs.items() if k.startswith('val_')}
        super(FileWritersTensorBoardCallback, self).on_train_end(logs)
        self.val_writer.close()
