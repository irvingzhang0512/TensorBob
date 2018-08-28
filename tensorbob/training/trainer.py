import tensorflow as tf
import os
from tensorflow.python.platform import tf_logging as logging
from tensorbob.utils.variables import get_variables_to_restore
from tensorbob.training.training_utils import train, create_train_op, create_finetune_train_op, \
    evaluate_on_single_scale, ValidationDatasetEvaluationHook
from tensorbob.training.trainer_utils import learning_rate_exponential_decay, learning_rate_val_evaluation, \
    learning_rate_steps_dict

__all__ = ['Trainer', 'BaseClassificationTrainer', 'BaseSegmentationTrainer']


class Trainer:
    def __init__(self,
                 batch_size=32, weight_decay=0.00005, keep_prob=0.5,

                 learning_rate_type=1, learning_rate_start=0.001,
                 lr_decay_rate=0.5, lr_decay_steps=40000 * 10, lr_staircase=False,  # learning_rate_exponential_decay
                 steps_to_lr_dict=None, min_lr=0.000001,  # learning_rate_steps_dict
                 lr_shrink_epochs=3, lr_shrink_by_number=10.0,

                 base_logs_dir='./logs/',
                 step_ckpt_dir='ckpt', train_logs_dir='train',
                 val_logs_dir='val', best_val_ckpt_dir='best_val',

                 metrics_collection='val_metrics', metrics_update_ops_collection='update_ops',
                 metrics_reset_ops_collection='reset_ops', use_mean_metrics=False,

                 logging_every_n_steps=1000, summary_every_n_steps=1000, save_every_n_steps=1000,
                 evaluate_every_n_steps=10000, max_steps=None):
        self._logging_every_n_steps = logging_every_n_steps
        self._summary_every_n_steps = summary_every_n_steps
        self._save_every_n_steps = save_every_n_steps
        self._evaluate_every_n_steps = evaluate_every_n_steps
        self._max_steps = max_steps

        self._base_logs_dir = base_logs_dir
        self._step_ckpt_dir = step_ckpt_dir
        self._train_logs_dir = train_logs_dir
        self._val_logs_dir = val_logs_dir
        self._best_val_ckpt_dir = best_val_ckpt_dir

        self._metrics_collection = metrics_collection
        self._metrics_update_ops_collection = metrics_update_ops_collection
        self._metrics_reset_ops_collection = metrics_reset_ops_collection
        self._use_mean_metrics = use_mean_metrics

        self._ph_is_training = tf.placeholder(tf.bool, name='is_training')
        self._ph_use_mean_metrics = tf.placeholder(tf.bool, name='use_mean_metrics')

        self._batch_size = batch_size
        self._weight_decay = weight_decay
        self._keep_prob = keep_prob

        self._learning_rate_type = learning_rate_type
        self._learning_rate_start = learning_rate_start
        self._lr_decay_rate = lr_decay_rate
        self._lr_decay_steps = lr_decay_steps
        self._lr_staircase = lr_staircase
        self._steps_to_lr_dict = steps_to_lr_dict
        self._min_lr = min_lr
        self._lr_shrink_epochs = lr_shrink_epochs
        self._lr_shrink_by_number = lr_shrink_by_number

        self._train_dataset = None
        self._val_dataset = None
        self._main_metric = None
        self._x = None
        self._y = None

    def _get_training_dataset(self):
        """
        BaseDataset 实例，要求 next_batch 返回两个参数，分别是 数据（图像）和标签
        :return:    BaseDataset
        """
        raise NotImplementedError

    def _get_val_dataset(self):
        """
        BaseDataset 实例，要求 next_batch 返回两个参数，分别是 数据（图像）和标签
        :return:    BaseDataset
        """
        raise NotImplementedError

    def _get_optimizer(self):
        raise NotImplementedError

    def _get_model(self):
        raise NotImplementedError

    def _get_loss(self, logits):
        """
        根据模型输出的 logits 计算误差
        :param logits:      _get_model 返回结果
        :return:            误差
        """
        raise NotImplementedError

    def _get_metrics(self, logits, total_loss):
        """
        训练时用到的各种性能指标，用于后续 logging hook
        list对象，第一个 metrics 为其主要性能指标（如分类任务中的 accuracy，图像分割任务中的 mean IoU）
        :param logits:          _get_model 返回结果
        :param total_loss:      _get_loss 返回结果
        :return:
        """
        raise NotImplementedError

    def _get_hooks(self):
        raise NotImplementedError

    def _get_train_op(self, total_loss, optimizer):
        raise NotImplementedError

    def _get_scaffold(self):
        """
        主要用于导入 pre-trained model
        :return:        tf.train.Scaffold 对象，默认可以是None
        """
        raise NotImplementedError

    def _get_train_feed_fn(self):
        """
        返回一个函数，该函数的返回值就是每次训练时要用到的参数
        如 is_training 等
        :return:
        """
        raise NotImplementedError

    def _get_learning_rate(self):
        """
        获取学习率，有三种模式
        1：learning_rate_exponential_decay
        2：到规定 steps 修改 learning rate，参考 learning_rate_steps_dict 函数。
        3：根据验证集结果进行学习率衰减，参考 learning_rate_val_evaluation 函数。
        :return:    学习率
        """
        if self._learning_rate_type not in [1, 2, 3]:
            raise ValueError('learning_rate_type must in [1, 2, 3]')
        if self._learning_rate_type == 1:
            if self._learning_rate_start is None \
                    or self._lr_decay_steps is None \
                    or self._lr_decay_rate is None \
                    or self._lr_staircase is None:
                raise ValueError('learning rate vars error')
            lr = learning_rate_exponential_decay(self._learning_rate_start,
                                                 tf.train.get_or_create_global_step(),
                                                 self._lr_decay_steps,
                                                 self._lr_decay_rate,
                                                 self._lr_staircase)
            self._learning_rate_tensor = lr
        elif self._learning_rate_type == 2:
            if self._steps_to_lr_dict is None or self._min_lr is None:
                raise ValueError('learning rate vars error')
            lr = learning_rate_steps_dict(self._steps_to_lr_dict,
                                          self._min_lr,
                                          tf.train.get_or_create_global_step())
            self._learning_rate_tensor = lr
        else:
            if self._learning_rate_start is None \
                    or self._lr_shrink_epochs is None \
                    or self._lr_shrink_by_number is None:
                raise ValueError('learning rate vars error')
            lr = learning_rate_val_evaluation(self._learning_rate_start)
            self._learning_rate_tensor = lr
        tf.summary.scalar('learning_rate', lr)
        return lr

    def train(self):
        # 获取数据集
        logging.debug('creating datasets')
        self._train_dataset = self._get_training_dataset()
        self._x, self._y = self._train_dataset.next_batch
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
        logging_tensors = metrics
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
              os.path.join(self._base_logs_dir, self._step_ckpt_dir),
              scaffold=scaffold,
              hooks=hooks,
              max_steps=self._max_steps,
              logging_tensors=logging_tensors,
              logging_every_n_steps=self._logging_every_n_steps,
              feed_fn=self._get_train_feed_fn,
              summary_writer=summary_writer, summary_every_n_steps=self._summary_every_n_steps,
              save_every_n_steps=self._save_every_n_steps)


class BaseClassificationTrainer(Trainer):
    def __init__(self, num_classes=1000,
                 training_crop_size=224, val_crop_size=384,
                 fine_tune_steps=None,
                 fine_tune_var_include=None,
                 fine_tune_var_exclude=None,
                 **kwargs):
        # {
        #     'num_classes': 1000,
        #     'training_crop_size': 224,
        #     'val_crop_size': 384,
        #     'fine_tune_steps': None,
        #     'fine_tune_var_include': None,
        #     'fine_tune_var_exclude': None,
        #
        #     'batch_size': 32,
        #     'weight_decay': 0.00005,
        #     'keep_prob': 0.5,
        #     'leraning_rate_type': 1,
        #     'learning_rate_start': 0.001,
        #     'lr_decay_rate': 0.5,
        #     'lr_decay_steps': 40000 * 10,
        #     'lr_staircase': False,
        #     'steps_to_lr_dict': None,
        #     'min_lr': 0.000001,
        #     'lr_shrink_epochs': 3,
        #     'lr_shrink_by_number': 10.0,
        #     'base_logs_dir': 'logs'
        #     'step_ckpt_dir': 'ckpt',
        #     'train_logs_dir': 'train',
        #     'val_logs_dir': 'val',
        #     'best_val_ckpt_dir': 'best_val',
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
        self._fine_tune_var_include = fine_tune_var_include
        self._fine_tune_var_exclude = fine_tune_var_exclude
        if fine_tune_steps is not None and fine_tune_var_include is None:
            raise ValueError('fine_tune_var_include must not be None if fine_tune_steps is not None.')

    def _get_training_dataset(self):
        pass

    def _get_val_dataset(self):
        pass

    def _get_optimizer(self):
        pass

    def _get_model(self):
        pass

    def _get_scaffold(self):
        pass

    def _get_loss(self, logits):
        tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=self._y)
        return tf.losses.get_total_loss()

    def _get_metrics(self, logits, total_loss):
        predictions = tf.argmax(logits, axis=1)
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

        mean_accuracy, _ = tf.metrics.accuracy(self._y, predictions,
                                               metrics_collections=[self._metrics_collection],
                                               updates_collections=[self._metrics_update_ops_collection],
                                               name='mean_accuracy')
        non_mean_accuracy = tf.reduce_mean(tf.cast(tf.equal(self._y, predictions), tf.float32))
        accuracy = tf.cond(self._ph_use_mean_metrics,
                           lambda: mean_accuracy,
                           lambda: non_mean_accuracy,
                           name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

        return [accuracy, loss]

    def _get_train_op(self, total_loss, optimizer):
        global_step = tf.train.get_or_create_global_step()
        if self._fine_tune_steps is None:
            return create_train_op(total_loss, optimizer, global_step,
                                   update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        train_op1 = create_train_op(total_loss, optimizer, global_step,
                                    update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS),
                                    variables_to_train=get_variables_to_restore(self._fine_tune_var_include,
                                                                                self._fine_tune_var_exclude))
        train_op2 = create_train_op(total_loss, optimizer, global_step,
                                    update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        return create_finetune_train_op(train_op1, train_op2, self._fine_tune_steps, global_step)

    def _get_train_feed_fn(self):
        return {self._ph_is_training: True,
                self._ph_use_mean_metrics: self._use_mean_metrics}

    def _get_hooks(self):
        evaluate_feed_dict = {self._ph_is_training: False,
                              self._ph_use_mean_metrics: True}
        evaluate_fn = evaluate_on_single_scale(self._x, self._y,
                                               evaluate_feed_dict,
                                               tf.get_collection(self._metrics_reset_ops_collection),
                                               tf.get_collection(self._metrics_update_ops_collection),
                                               main_metric=self._main_metric)
        summary_feed_dict = {
            self._ph_is_training: False,
            self._ph_use_mean_metrics: True,
        }
        val_summary_writer = tf.summary.FileWriter(self._val_logs_dir, tf.get_default_graph())
        val_set_shrink_lr_flag = (self._learning_rate_type == 3)
        validation_evaluate_hook = ValidationDatasetEvaluationHook(self._val_dataset,
                                                                   self._evaluate_every_n_steps,
                                                                   summary_op=tf.summary.merge_all(),
                                                                   summary_writer=val_summary_writer,
                                                                   saver_file_prefix=os.path.join(
                                                                       self._best_val_ckpt_dir,
                                                                       'model.ckpt'),
                                                                   evaluate_fn=evaluate_fn,
                                                                   summary_feed_dict=summary_feed_dict,
                                                                   shrink_learning_rate=val_set_shrink_lr_flag,
                                                                   shrink_by_number=self._lr_shrink_by_number,
                                                                   shrink_epochs=self._lr_shrink_epochs)
        return [validation_evaluate_hook]


class BaseSegmentationTrainer(Trainer):
    def __init__(self, num_classes=1000,
                 training_crop_size=224, val_crop_size=224,
                 **kwargs):
        # {
        #     'num_classes': 1000,
        #     'training_crop_size': 224,
        #     'val_crop_size': 384,
        #
        #     'batch_size': 32,
        #     'weight_decay': 0.00005,
        #     'keep_prob': 0.5,
        #     'leraning_rate_type': 1,
        #     'learning_rate_start': 0.001,
        #     'lr_decay_rate': 0.5,
        #     'lr_decay_steps': 40000 * 10,
        #     'lr_staircase': False,
        #     'steps_to_lr_dict': None,
        #     'min_lr': 0.000001,
        #     'lr_shrink_epochs': 3,
        #     'lr_shrink_by_number': 10.0,
        #     'base_logs_dir': 'logs'
        #     'step_ckpt_dir': 'ckpt',
        #     'train_logs_dir': 'train',
        #     'val_logs_dir': 'val',
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

    def _get_training_dataset(self):
        pass

    def _get_val_dataset(self):
        pass

    def _get_optimizer(self):
        pass

    def _get_model(self):
        pass

    def _get_scaffold(self):
        pass

    def _get_loss(self, logits):
        tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=self._y)
        return tf.losses.get_total_loss()

    def _get_metrics(self, logits, total_loss):
        predictions = tf.argmax(logits, axis=-1)
        _, loss = tf.metrics.mean(total_loss,
                                  metrics_collections=[self._metrics_collection],
                                  updates_collections=[self._metrics_update_ops_collection],
                                  name='loss')
        _, accuracy = tf.metrics.accuracy(self._y, predictions,
                                          metrics_collections=[self._metrics_collection],
                                          updates_collections=[self._metrics_update_ops_collection],
                                          name='accuracy')
        _, confused_matrix = tf.metrics.mean_iou(tf.reshape(self._y, [-1]),
                                                 tf.reshape(predictions, [-1]),
                                                 self._num_classes,
                                                 metrics_collections=[self._metrics_collection],
                                                 updates_collections=[self._metrics_update_ops_collection],
                                                 name='confused_matrix')
        mean_iou = compute_mean_iou_by_confusion_matrix('mean_iou', confused_matrix)
        for metric in tf.get_collection(tf.GraphKeys.METRIC_VARIABLES):
            tf.add_to_collection(self._metrics_reset_ops_collection,
                                 tf.assign(metric, tf.zeros(metric.get_shape(), metric.dtype)))
        with tf.control_dependencies(tf.get_collection(self._metrics_reset_ops_collection)):
            after_reset_loss = tf.identity(loss)
            after_reset_accuracy = tf.identity(accuracy)
            after_reset_mean_iou = tf.identity(mean_iou)
        final_loss, final_accuracy, final_mean_iou = tf.cond(self._ph_use_mean_metrics,
                                                             lambda: [loss,
                                                                      accuracy,
                                                                      mean_iou],
                                                             lambda: [after_reset_loss,
                                                                      after_reset_accuracy,
                                                                      after_reset_mean_iou])
        tf.summary.scalar('mean_iou', final_mean_iou)
        tf.summary.scalar('accuracy', final_accuracy)
        tf.summary.scalar('loss', final_loss)

        return [final_mean_iou, final_accuracy, final_loss]

    def _get_train_op(self, total_loss, optimizer):
        global_step = tf.train.get_or_create_global_step()
        return create_train_op(total_loss, optimizer, global_step,
                               update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    def _get_train_feed_fn(self):
        return {self._ph_is_training: True,
                self._ph_use_mean_metrics: self._use_mean_metrics}

    def _get_hooks(self):
        evaluate_feed_dict = {self._ph_is_training: False,
                              self._ph_use_mean_metrics: True}
        evaluate_fn = evaluate_on_single_scale(self._x, self._y,
                                               evaluate_feed_dict,
                                               tf.get_collection(self._metrics_reset_ops_collection),
                                               tf.get_collection(self._metrics_update_ops_collection),
                                               main_metric=self._main_metric)
        summary_feed_dict = {
            self._ph_is_training: False,
            self._ph_use_mean_metrics: True,
        }
        val_summary_writer = tf.summary.FileWriter(self._val_logs_dir, tf.get_default_graph())
        val_set_shrink_lr_flag = (self._learning_rate_type == 3)
        validation_evaluate_hook = ValidationDatasetEvaluationHook(self._val_dataset,
                                                                   self._evaluate_every_n_steps,
                                                                   summary_op=tf.summary.merge_all(),
                                                                   summary_writer=val_summary_writer,
                                                                   saver_file_prefix=os.path.join(
                                                                       self._best_val_ckpt_dir,
                                                                       'model.ckpt'),
                                                                   evaluate_fn=evaluate_fn,
                                                                   summary_feed_dict=summary_feed_dict,
                                                                   shrink_learning_rate=val_set_shrink_lr_flag,
                                                                   shrink_by_number=self._lr_shrink_by_number,
                                                                   shrink_epochs=self._lr_shrink_epochs)
        return [validation_evaluate_hook]


def compute_mean_iou_by_confusion_matrix(name, total_cm):
    sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
    sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
    cm_diag = tf.to_float(tf.diag_part(total_cm))
    denominator = sum_over_row + sum_over_col - cm_diag

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = tf.reduce_sum(
        tf.cast(
            tf.not_equal(denominator, 0), dtype=tf.float32))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = tf.where(
        tf.greater(denominator, 0), denominator,
        tf.ones_like(denominator))
    iou = tf.div(cm_diag, denominator)

    # If the number of valid entries is 0 (no classes) we return 0.
    result = tf.where(
        tf.greater(num_valid_entries, 0),
        tf.reduce_sum(iou, name=name) / num_valid_entries, 0)
    return result


def compute_mean_iou(name, predictions, labels):
    total_cm = tf.confusion_matrix(predictions=predictions, labels=labels)
    return compute_mean_iou_by_confusion_matrix(name, total_cm)
