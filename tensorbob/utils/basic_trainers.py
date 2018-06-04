import tensorflow as tf
import numpy as np
import os
from .trainer import Trainer
from .training import create_train_op, create_finetune_train_op
from .training_utils import TrainDatasetFeedDictHook, evaluate_on_single_scale, ValidationDatasetEvaluationHook

__all__ = ['BaseClassificationTrainer']


class BaseClassificationTrainer(Trainer):
    def __init__(self, num_classes=1000,
                 training_crop_size=224, val_crop_size=384,
                 fine_tune_steps=None, fine_tune_var_list=None,
                 **kwargs):
        # {
        #     'num_classes': 1000,
        #     'training_crop_size': 224,
        #     'val_crop_size': 384,
        #     'fine_tune_steps': None,
        #     'fine_tune_var_list': None,
        #
        #     'batch_size': 32,
        #     'weight_decay': 0.00005,
        #     'keep_prob': 0.5,
        #     'learning_rate_start': 0.001,
        #     'lr_decay_rate': 0.5,
        #     'lr_decay_steps': 40000 * 10,
        #     'lr_staircase': False,
        #     'step_ckpt_dir': './logs/ckpt/',
        #     'train_logs_dir': './logs/train/',
        #     'val_logs_dir': './logs/val/',
        #     'best_val_ckpt_dir': './logs/best_val/',
        #     'metrics_collection': 'val_metrics',
        #     'metrics_update_ops_collection': 'update_ops',
        #     'metrics_reset_ops_collection': 'reset_ops',
        #     'use_mean_metrics': False,
        #     'logging_every_n_steps': 1000,
        #     'summary_every_n_steps': 1000,
        #     'save_every_n_steps': 1000,
        #     'evaluate_every_n_steps': 10000,
        #     'max_steps': None
        # }
        super().__init__(**kwargs)
        self._training_crop_size = training_crop_size
        self._val_crop_size = val_crop_size
        self._num_classes = num_classes
        self._fine_tune_steps = fine_tune_steps
        self._fine_tune_var_list = fine_tune_var_list
        if fine_tune_steps is not None and fine_tune_var_list is None:
            raise ValueError('fine_tune_var_list must not be None if fine_tune_steps is not None.')

    def _get_training_dataset(self):
        raise NotImplementedError

    def _get_val_dataset(self):
        raise NotImplementedError

    def _get_model(self):
        raise NotImplementedError

    def _get_optimizer(self):
        raise NotImplementedError

    def _get_scaffold(self):
        raise NotImplementedError

    def _get_learning_rate(self):
        return tf.train.exponential_decay(self._learning_rate_start,
                                          tf.train.get_or_create_global_step(),
                                          self._lr_decay_steps,
                                          self._lr_decay_rate,
                                          self._lr_staircase,
                                          'learning_rate')

    def _get_loss(self, logits):
        tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=self._ph_y)
        return tf.losses.get_total_loss()

    def _get_metrics(self, logits, total_loss):
        predictions = tf.argmax(tf.nn.softmax(logits), axis=1)
        mean_loss, _ = tf.metrics.mean(total_loss,
                                       metrics_collections=[self._metrics_collection],
                                       updates_collections=[self._metrics_update_ops_collection],
                                       name='mean_loss')
        non_mean_loss = total_loss
        loss = tf.cond(self._ph_use_mean_metrics,
                       lambda: mean_loss,
                       lambda: non_mean_loss,
                       name='loss')
        tf.summary.scalar('loss', loss)

        mean_accuracy, _ = tf.metrics.accuracy(self._ph_y, predictions,
                                               metrics_collections=[self._metrics_collection],
                                               updates_collections=[self._metrics_update_ops_collection],
                                               name='mean_accuracy')
        non_mean_accuracy = tf.reduce_mean(tf.cast(tf.equal(self._ph_y, predictions), tf.float32))
        accuracy = tf.cond(self._ph_use_mean_metrics,
                           lambda: mean_accuracy,
                           lambda: non_mean_accuracy,
                           name='accuracy')

        for metric in tf.get_collection(tf.GraphKeys.METRIC_VARIABLES):
            tf.add_to_collection(self._metrics_reset_ops_collection,
                                 tf.assign(metric, tf.zeros(metric.get_shape(), metric.dtype)))
        tf.summary.scalar('accuracy', accuracy)

        return [accuracy, loss]

    def _get_train_op(self, total_loss, optimizer):
        global_step = tf.train.get_or_create_global_step()
        if self._fine_tune_steps is None:
            return create_train_op(total_loss, optimizer, global_step,
                                   update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        train_op1 = create_train_op(total_loss, optimizer, global_step,
                                    update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS),
                                    variables_to_train=self._fine_tune_var_list)
        train_op2 = create_train_op(total_loss, optimizer, global_step,
                                    update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS),
                                    variables_to_train=self._fine_tune_var_list)
        return create_finetune_train_op(train_op1, train_op2, self._fine_tune_steps, global_step)

    def _get_train_feed_fn(self):
        return {self._ph_is_training: True,
                self._ph_image_size: self._training_crop_size,
                self._ph_use_mean_metrics: self._use_mean_metrics}

    def _get_hooks(self):
        train_dataset_hook = TrainDatasetFeedDictHook(self._train_dataset, self._ph_x, self._ph_y)

        evaluate_feed_dict = {self._ph_image_size: self._val_crop_size,
                              self._ph_is_training: False,
                              self._ph_use_mean_metrics: True}
        evaluate_fn = evaluate_on_single_scale(self._val_crop_size, self._ph_x, self._ph_y,
                                               evaluate_feed_dict, None,
                                               tf.get_collection(self._metrics_reset_ops_collection),
                                               tf.get_collection(self._metrics_update_ops_collection),
                                               main_metric=self._main_metric)
        summary_feed_dict = {
            self._ph_image_size: self._val_crop_size,
            self._ph_is_training: False,
            self._ph_use_mean_metrics: True,
            self._ph_x: np.zeros([self._batch_size, self._val_crop_size, self._val_crop_size, 3]),
            self._ph_y: np.zeros([self._batch_size])
        }
        val_summary_writer = tf.summary.FileWriter(self._val_logs_dir, tf.get_default_graph())
        validation_evaluate_hook = ValidationDatasetEvaluationHook(self._val_dataset,
                                                                   self._evaluate_every_n_steps,
                                                                   summary_op=tf.summary.merge_all(),
                                                                   summary_writer=val_summary_writer,
                                                                   saver_file_prefix=os.path.join(self._best_val_ckpt_dir,
                                                                                                  'model.ckpt'),
                                                                   evaluate_fn=evaluate_fn,
                                                                   summary_feed_dict=summary_feed_dict)
        return [train_dataset_hook, validation_evaluate_hook]
