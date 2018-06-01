import tensorflow as tf
from .training import train
from tensorflow.python.platform import tf_logging as logging

__all__ = ['Trainer']


class Trainer:
    def __init__(self,
                 batch_size=32, weight_decay=0.00005, keep_prob=0.5,

                 learning_rate_start=0.001, lr_decay_rate=0.5, lr_decay_steps=40000 * 10, lr_staircase=False,

                 step_ckpt_dir='./logs/ckpt/', train_logs_dir='./logs/train/',
                 val_logs_dir='./logs/val/', best_val_ckpt_dir='./logs/best_val/',

                 metrics_collection='val_metrics', metrics_update_ops_collection='update_ops',
                 metrics_reset_ops_collection='reset_ops', use_mean_metrics=False,

                 logging_every_n_steps=1000, summary_every_n_steps=1000, save_every_n_steps=1000,
                 evaluate_every_n_steps=10000, max_steps=None):
        self._logging_every_n_steps = logging_every_n_steps
        self._summary_every_n_steps = summary_every_n_steps
        self._save_every_n_steps = save_every_n_steps
        self._evaluate_every_n_steps = evaluate_every_n_steps
        self._max_steps = max_steps

        self._step_ckpt_dir = step_ckpt_dir
        self._train_logs_dir = train_logs_dir
        self._val_logs_dir = val_logs_dir
        self._best_val_ckpt_dir = best_val_ckpt_dir

        self._metrics_collection = metrics_collection
        self._metrics_update_ops_collection = metrics_update_ops_collection
        self._metrics_reset_ops_collection = metrics_reset_ops_collection
        self._use_mean_metrics = use_mean_metrics

        self._ph_x = tf.placeholder(tf.float32, name='x')
        self._ph_y = tf.placeholder(tf.int64, name='y')
        self._ph_image_size = tf.placeholder(tf.int32, name='image_sizes')
        self._ph_is_training = tf.placeholder(tf.bool, name='is_training')
        self._ph_use_mean_metrics = tf.placeholder(tf.bool, name='use_mean_metrics')

        self._batch_size = batch_size
        self._weight_decay = weight_decay
        self._keep_prob = keep_prob

        self._learning_rate_start = learning_rate_start
        self._lr_decay_rate = lr_decay_rate
        self._lr_decay_steps = lr_decay_steps
        self._lr_staircase = lr_staircase

        self._train_dataset = None
        self._val_dataset = None
        self._main_metric = None

    def _get_training_dataset(self):
        raise NotImplementedError

    def _get_val_dataset(self):
        raise NotImplementedError

    def _get_optimizer(self):
        raise NotImplementedError

    def _get_learning_rate(self):
        raise NotImplementedError

    def _get_model(self):
        raise NotImplementedError

    def _get_loss(self, logits):
        raise NotImplementedError

    def _get_metrics(self, logits, total_loss):
        raise NotImplementedError

    def _get_hooks(self):
        raise NotImplementedError

    def _get_train_op(self, total_loss, optimizer):
        raise NotImplementedError

    def _get_scaffold(self):
        raise NotImplementedError

    def _get_train_feed_fn(self):
        raise NotImplementedError

    def train(self):
        # 获取数据集
        logging.debug('creating datasets')
        self._train_dataset = self._get_training_dataset()
        self._val_dataset = self._get_val_dataset()
        logging.debug('successfully get datasets with size %d and %d' % (self._train_dataset.size,
                                                                         self._val_dataset.size))
        # 建立模型，得到结果
        logits, end_points = self._get_model()
        logging.debug('successfully getting logits')

        # 获取损失函数
        total_loss = self._get_loss(logits)
        logging.debug('successfully getting total loss')

        # 构建优化器
        optimizer = self._get_optimizer()
        logging.debug('successfully getting optimizer')

        # 性能指标相关操作
        metrics = self._get_metrics(logits, total_loss)
        if metrics is not None:
            self._main_metric = metrics[0]
        logging.debug('successfully getting metrics')

        # 构建train_op, hooks, scaffold
        train_op = self._get_train_op(total_loss, optimizer)
        logging.debug('successfully getting train_op')
        hooks = self._get_hooks()
        logging.debug('successfully getting hooks')
        scaffold = self._get_scaffold()
        logging.debug('successfully getting scaffold')

        # train summary writer
        summary_writer = tf.summary.FileWriter(self._train_logs_dir, graph=tf.get_default_graph())

        logging.debug('training start')
        train(train_op,
              self._step_ckpt_dir,
              scaffold=scaffold,
              hooks=hooks,
              max_steps=self._max_steps,
              logging_tensors=metrics,
              logging_every_n_steps=self._logging_every_n_steps,
              feed_fn=self._get_train_feed_fn,
              summary_writer=summary_writer, summary_every_n_steps=self._summary_every_n_steps,
              save_every_n_steps=self._save_every_n_steps)
