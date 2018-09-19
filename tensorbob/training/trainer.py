import tensorflow as tf
import os
from tensorflow.python.platform import tf_logging as logging
from tensorbob.utils.variables import get_variables_to_restore, assign_from_checkpoint_fn
from tensorbob.training.training_utils import train, create_train_op, create_finetune_train_op, \
    ValidationDatasetEvaluationHook, InitFnHook
from tensorbob.training.trainer_utils import learning_rate_exponential_decay, learning_rate_val_evaluation, \
    learning_rate_steps_dict
from tensorbob.utils.metrics_utils import compute_mean_iou_by_confusion_matrix

__all__ = ['BaseTrainer', 'BaseClassificationTrainer', 'BaseSegmentationTrainer']


class BaseTrainer:
    def __init__(self,
                 # 基本参数
                 batch_size=32, weight_decay=0.0005, keep_prob=0.8,

                 # 学习率相关参数
                 learning_rate_type=0, learning_rate_start=0.0001,
                 lr_decay_rate=None, lr_decay_steps=None, lr_staircase=None,  # learning_rate_exponential_decay
                 steps_to_lr_dict=None, min_lr=None,  # learning_rate_steps_dict
                 lr_shrink_epochs=None, lr_shrink_by_number=None,  # 配合 ValidationDatasetEvaluationHook 衰减学习率

                 # 各种路径
                 base_logs_dir='./logs',
                 val_logs_dir='val',

                 # 性能指标相关参数
                 metrics_reset_ops_collection='reset_ops',

                 # 各种 step 限制
                 logging_every_n_steps=1000,
                 summary_every_n_steps=1000,
                 save_every_n_steps=1000,
                 evaluate_every_n_steps=10000,
                 max_steps=None,

                 # fine-tune 相关
                 fine_tune_steps=None,
                 fine_tune_file_path=None,
                 fine_tune_vars_include=None,
                 fine_tune_vars_exclude=None,
                 ):
        """
        设定基本训练流程

        参数举例如下
        {
            'batch_size': 32,
            'weight_decay': 0.0005,
            'keep_prob': 0.8,

            'learning_rate_type': 0,
            'learning_rate_start': 0.0001,
            'lr_decay_rate': None,
            'lr_decay_steps': None,
            'lr_staircase': None,
            'steps_to_lr_dict': None,
            'min_lr': None,
            'lr_shrink_epochs': None,
            'lr_shrink_by_number': None,

            'base_logs_dir': './logs'
            'val_logs_dir': 'val',

            'metrics_reset_ops_collection': 'reset_ops',

            'logging_every_n_steps': 1000,
            'summary_every_n_steps': 1000,
            'save_every_n_steps': 1000,
            'evaluate_every_n_steps': 10000,
            'max_steps': None,

            'fine_tune_steps': None,
            'fine_tune_file_path': None,
            'fine_tune_vars_include': None,
            'fine_tune_vars_exclude': None,

        }

        已实现的功能：
        1. `_get_learning_rate`: 获取学习率，根据 learning_rate_type 来获取不同类型学习率：
            0: 固定学习率为 learning_rate_start。
            1: tf.train.exponential_decay(learning_rate_start, global_step,
                                          lr_decay_steps, lr_decay_rate, lr_staircase)
            2: 根据 steps_to_lr_dict, min_lr 获取学习率。
               如 steps_to_lr_dict = {100: 0.001, 50000: 0.0001}, min_lr = 0.00001
               则 global_step 为[0, 100]时学习率为 0.001，(100, 50000]时学习率为 0.0001，(50000, inf]时学习率为 0.00001
            3：若连续 lr_shrink_epochs 次验证集测试最佳性能指标不提升，则将衰减学习率为之前的 1/lr_shrink_by_number
               具体实现方式如下：
               获取 tf.get_variable('learning_rate/learning_rate_start')，初始值为 learning_rate_start，数值不变。
               获取 tf.get_variable('learning_rate/learning_rate_shrink')，初始值为 1，
               通过 ValidationDatasetEvaluationHook 更新数值。
               学习率通过上述两值相除计算。
        2. `_get_hooks`：获取两个Hook。
            InitFnHook：获取数据集的 string_handle，初始化训练集。
            ValidationDatasetEvaluationHook：定期再验证机上评估性能指标。
        3. `_get_train_op`：获取训练模型所需的op。
        4. `_get_train_feed_fn`：获取训练所需的 feed_dict
                                 主要包括 self._ph_is_training 和数据集选择相关的 self._merged_dataset.ph_handle.
        5. `_get_scaffold`：获取 tf.train.Scaffold 对象。
                            默认配合 `_get_fine_tune_init_fn()` 和 `_get_fine_tune_var_dict()` ，导入 fine tune 模型。
                            在前者中使用 fine_tune_vars_include 和 fine_tune_vars_exclude 选择当前计算图中需要导入参数的变量
                            后者（未实现）返回map，key为 ckpt 文件中变量名（str），value为当前计算图中的变量对象（tf.Variable）




        需要实现的功能
        1. `_get_merged_dataset`：构建数据集，即创建 MergedDataset 对象（该数据集包括训练集与验证集）。
        2. `_get_optimizer`：获取优化器，一般会用到 `self._get_learning_rate()` 获取学习率。
        3. `_get_model`：获取模型，返回值包括 logits 和 end_points，一般会使用到 `self._x` 作为输入。
        4. `_get_loss`：获取损失函数，返回总损失函数，一般会用到 `self._y` 作为标签
        5. `_get_metrics`：返回一系列性能指标，其中第一个性能指标为主指标，要求数值越大性能最优。
                           由[summary_metrics_ops], [update_metrics_ops], [after_reset_update_metrics_ops]组成
                           summary_metrics_ops 由 tf.metrics 第一个返回值组成
                           update_metrics_ops 由 tf.metrics 的第二个返回值组成
                           after_reset_update_metrics_ops 需要先将 tf.get_collection(tf.GraphKeys.METRIC_VARIABLES) 清零，
                                再调用 update_metrics_ops


        训练流程：
        1. 获取训练集，`_get_merged_dataset`。
        2. 获取模型，`_get_model`。
        3. 计算损失函数，`_get_loss`。
        4. 获取优化器，`_get_optimizer`。
        5. 获取性能指标，`_get_metrics`。
        6. 构建train_op，`_get_train_op`。
        7. 构建hooks，`_get_hooks`，包括数据集初始化功能，和验证集评估功能。
        8. 构建scaffold，`_get_scaffold`，包括 fine-tune 模型获取。
        9. 开始训练：
            + 运行 scaffold 函数中的 init_op, init_fn（如果有的话）。
            + 判断 base_logs_dir 中是否存在ckpt文件，由的话先restore。
            + 正常训练（包括train_op，summary，save，logging，validation），具体参考 `training_utils.py` 中的 `train`函数。

        :param batch_size:
        :param weight_decay:
        :param keep_prob:

        学习率相关参数请参考 _get_learning_rate 函数注释
        :param learning_rate_type:
        :param learning_rate_start:
        :param lr_decay_rate:
        :param lr_decay_steps:
        :param lr_staircase:
        :param steps_to_lr_dict:
        :param min_lr:
        :param lr_shrink_epochs:
        :param lr_shrink_by_number:
        :param base_logs_dir:
        :param val_logs_dir:

        :param metrics_reset_ops_collection:

        # steps 相关参数请参考 train 函数注释
        :param logging_every_n_steps:
        :param summary_every_n_steps:
        :param save_every_n_steps:
        :param evaluate_every_n_steps:

        :param max_steps:                       最大steps
        :param fine_tune_steps:                 暂时没用
        :param fine_tune_file_path:             string类型，ckpt文件路径
        :param fine_tune_vars_include:          list类型，当前特征图中需要的 scope
        :param fine_tune_vars_exclude:          list类型，当前特征图中不需要的 scope
        """
        # 各种 steps 参数
        self._logging_every_n_steps = logging_every_n_steps
        self._summary_every_n_steps = summary_every_n_steps
        self._save_every_n_steps = save_every_n_steps
        self._evaluate_every_n_steps = evaluate_every_n_steps
        self._max_steps = max_steps

        # 日志路径参数
        self._base_logs_dir = base_logs_dir
        self._val_logs_dir = val_logs_dir

        # 性能指标相关参数
        self._metrics_reset_ops_collection = metrics_reset_ops_collection
        self._metrics_update_ops = None
        self._metrics_update_after_reset_ops = None
        self._metrics_summary_ops = None

        # 训练参数
        self._batch_size = batch_size
        self._weight_decay = weight_decay
        self._keep_prob = keep_prob

        # 学习率相关参数
        self._learning_rate_type = learning_rate_type
        self._learning_rate_start = learning_rate_start
        self._lr_decay_rate = lr_decay_rate
        self._lr_decay_steps = lr_decay_steps
        self._lr_staircase = lr_staircase
        self._steps_to_lr_dict = steps_to_lr_dict
        self._min_lr = min_lr
        self._lr_shrink_epochs = lr_shrink_epochs
        self._lr_shrink_by_number = lr_shrink_by_number

        # fine tune 相关参数
        self._fine_tune_steps = fine_tune_steps
        self._fine_tune_file_path = fine_tune_file_path
        self._fine_tune_vars_include = fine_tune_vars_include
        self._fine_tune_vars_exclude = fine_tune_vars_exclude

        # 其他需要初始化的参数
        self._ph_is_training = tf.placeholder(tf.bool, name='is_training')
        self._merged_dataset = None
        self._main_metric = None
        self._x = None
        self._y = None
        self._end_points = None

    def _get_merged_dataset(self):
        """
        获取数据集
        :return:    MergedDataset 对象
        """
        raise NotImplementedError

    def _get_optimizer(self):
        """
        获取优化器
        :return:
        """
        raise NotImplementedError

    def _get_model(self):
        """
        获取模型
        :return:    返回 logits 和 end_points
        """
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
        返回三组性能指标，分别时 [summary_metrics_ops], [update_metrics_ops], [after_reset_update_metrics_ops]
        summary_metrics_ops 由 tf.metrics 第一个返回值组成
        update_metrics_ops 由 tf.metrics 的第二个返回值组成
        after_reset_update_metrics_ops 需要先 reset 所有
        :param logits:          _get_model 返回结果
        :param total_loss:      _get_loss 返回结果
        :return:
        """
        raise NotImplementedError

    def _get_hooks(self):
        init_fn_hook = InitFnHook(self._merged_dataset.init)
        if self._evaluate_every_n_steps:
            val_feed_dict = {self._ph_is_training: False}
            validation_hook = ValidationDatasetEvaluationHook(self._merged_dataset,
                                                              evaluate_every_n_steps=self._evaluate_every_n_steps,
                                                              metrics_reset_ops=tf.get_collection(
                                                                  self._metrics_reset_ops_collection),
                                                              metrics_update_ops=self._metrics_update_ops,
                                                              evaluating_feed_dict=val_feed_dict,
                                                              summary_op=tf.summary.merge_all(),
                                                              summary_writer=tf.summary.FileWriter(os.path.join(
                                                                  self._base_logs_dir, self._val_logs_dir),
                                                                  tf.get_default_graph()),
                                                              saver_file_prefix=os.path.join(os.path.join(
                                                                  self._base_logs_dir, self._val_logs_dir),
                                                                  'model.ckpt'),
                                                              )
            return [init_fn_hook, validation_hook]
        return [init_fn_hook]

    def _get_train_op(self, total_loss, optimizer):
        # TODO: fine tune train ops
        return create_train_op(total_loss=total_loss,
                               optimizer=optimizer,
                               global_step=tf.train.get_or_create_global_step(),
                               update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        # if self._fine_tune_steps is None:
        #     return create_train_op(total_loss, optimizer, global_step,
        #                            update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        # train_op1 = create_train_op(total_loss, optimizer, global_step,
        #                             update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS),
        #                             variables_to_train=get_variables_to_restore(self._fine_tune_vars_include,
        #                                                                         self._fine_tune_vars_exclude))
        # train_op2 = create_train_op(total_loss, optimizer, global_step,
        #                             update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        # return create_finetune_train_op(train_op1, train_op2, self._fine_tune_steps, global_step)

    def _get_fine_tune_init_fn(self):
        if self._fine_tune_file_path is None:
            return None

        print(self._fine_tune_vars_include)
        print(self._fine_tune_vars_exclude)
        variables_to_restore = get_variables_to_restore(
            include=self._fine_tune_vars_include,
            exclude=self._fine_tune_vars_exclude,
        )
        var_dict = self._get_fine_tune_var_dict(variables_to_restore)

        logging.debug('restore %d variables' % len(var_dict))
        return assign_from_checkpoint_fn(self._fine_tune_file_path,
                                         var_dict,
                                         ignore_missing_vars=True,
                                         reshape_variables=True)

    def _get_fine_tune_var_dict(self, variables_to_restore):
        raise NotImplementedError

    def _get_scaffold(self):
        """
        主要用于导入 pre-trained model
        :return:        tf.train.Scaffold 对象，默认可以是None
        """
        fine_tune_init_fn = self._get_fine_tune_init_fn()
        if fine_tune_init_fn is None:
            return None

        def scaffold_init_fn(_, session):
            fine_tune_init_fn(session)

        return tf.train.Scaffold(init_fn=scaffold_init_fn)

    def _get_train_feed_fn(self):
        """
        返回一个函数，该函数的返回值就是每次训练时要用到的参数，如 ph_is_training
        该函数传递给 tf.train.FeedFnHook
        :return:
        """
        return {self._merged_dataset.ph_handle: self._merged_dataset.handle_strings[0],
                self._ph_is_training: True}

    def _get_learning_rate(self):
        """
        获取学习率，有三种模式
        0: 固定学习率，学习率为 learning_rate_start
        1：learning_rate_exponential_decay
        2：到规定 steps 修改 learning rate，参考 learning_rate_steps_dict 函数。
        3：根据验证集结果进行学习率衰减，参考 learning_rate_val_evaluation 函数。
        :return:    学习率
        """
        if self._learning_rate_type not in [0, 1, 2, 3]:
            raise ValueError('learning_rate_type must in [1, 2, 3]')
        if self._learning_rate_type == 0:
            if self._learning_rate_start is None:
                raise ValueError('learning rate vars error')
            self._learning_rate_tensor = self._learning_rate_start
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
        elif self._learning_rate_type == 3:
            if self._learning_rate_start is None \
                    or self._lr_shrink_epochs is None \
                    or self._lr_shrink_by_number is None:
                raise ValueError('learning rate vars error')
            lr = learning_rate_val_evaluation(self._learning_rate_start)
            self._learning_rate_tensor = lr
        else:
            raise ValueError('unknown learning_rate_type {}'.format(self._learning_rate_type))
        tf.summary.scalar('learning_rate', lr)
        return lr

    def train(self):
        # 获取数据集
        logging.debug('creating datasets')
        self._merged_dataset = self._get_merged_dataset()
        self._x, self._y = self._merged_dataset.next_batch
        logging.debug('successfully get dataset')

        # 建立模型，得到结果
        logits, self._end_points = self._get_model()
        logging.debug('successfully getting logits')

        # 获取损失函数
        total_loss = self._get_loss(logits)
        logging.debug('successfully getting total loss')

        # 构建优化器
        optimizer = self._get_optimizer()
        logging.debug('successfully getting optimizer')

        # 性能指标相关操作
        self._metrics_summary_ops, self._metrics_update_ops, \
            self._metrics_update_after_reset_ops = self._get_metrics(logits, total_loss)
        logging_tensors = self._metrics_update_after_reset_ops
        logging.debug('successfully getting metrics')

        # 构建train_op, hooks, scaffold
        train_op = self._get_train_op(total_loss, optimizer)
        logging.debug('successfully getting train_op')
        hooks = self._get_hooks()
        logging.debug('successfully getting hooks')
        scaffold = self._get_scaffold()
        logging.debug('successfully getting scaffold')

        logging.debug('training start')
        train(train_op,
              self._base_logs_dir,
              scaffold=scaffold,
              hooks=hooks,
              max_steps=self._max_steps,
              feed_fn=self._get_train_feed_fn,
              logging_tensors=logging_tensors, logging_every_n_steps=self._logging_every_n_steps,
              summary_writer=tf.summary.FileWriter(self._base_logs_dir, graph=tf.get_default_graph()),
              summary_every_n_steps=self._summary_every_n_steps,
              save_every_n_steps=self._save_every_n_steps,
              )


class BaseClassificationTrainer(BaseTrainer):
    def __init__(self, num_classes,
                 **kwargs):
        """
        分类任务的基本训练流程

        对 BaseTrainer 的改进：
        1. 添加了误差 tf.losses.sparse_softmax_cross_entropy
        2. 添加三个性能指标 mean_per_class_accuracy, accuracy, loss

        kwargs 的举例如下
        {
            'batch_size': 32,
            'weight_decay': 0.0005,
            'keep_prob': 0.8,

            'learning_rate_type': 0,
            'learning_rate_start': 0.0001,
            'lr_decay_rate': None,
            'lr_decay_steps': None,
            'lr_staircase': None,
            'steps_to_lr_dict': None,
            'min_lr': None,
            'lr_shrink_epochs': None,
            'lr_shrink_by_number': None,

            'base_logs_dir': './logs'
            'val_logs_dir': 'val',

            'metrics_reset_ops_collection': 'reset_ops',

            'logging_every_n_steps': 1000,
            'summary_every_n_steps': 1000,
            'save_every_n_steps': 1000,
            'evaluate_every_n_steps': 10000,
            'max_steps': None,

            'fine_tune_steps': None,
            'fine_tune_file_path': None,
            'fine_tune_vars_include': None,
            'fine_tune_vars_exclude': None,
        }
        :param num_classes:
        :param kwargs:
        """

        super().__init__(**kwargs)
        self._num_classes = num_classes

    def _get_merged_dataset(self):
        pass

    def _get_optimizer(self):
        pass

    def _get_model(self):
        pass

    def _get_fine_tune_var_dict(self, variables_to_restore):
        pass

    def _get_loss(self, logits):
        tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=self._y)
        return tf.losses.get_total_loss()

    def _get_metrics(self, logits, total_loss):
        predictions = tf.argmax(logits, axis=-1)
        summary_loss, loss = tf.metrics.mean(total_loss, name='loss')
        summary_accuracy, accuracy = tf.metrics.accuracy(self._y, predictions, name='accuracy')
        summary_mean_per_class_accuracy, mean_per_class_accuracy = tf.metrics.mean_per_class_accuracy(self._y,
                                                                                                      predictions,
                                                                                                      num_classes=self._num_classes)

        for metric in tf.get_collection(tf.GraphKeys.METRIC_VARIABLES):
            tf.add_to_collection(self._metrics_reset_ops_collection,
                                 tf.assign(metric, tf.zeros(metric.get_shape(), metric.dtype)))
        with tf.control_dependencies(tf.get_collection(self._metrics_reset_ops_collection)):
            after_reset_loss = tf.identity(loss)
            after_reset_accuracy = tf.identity(accuracy)
            after_reset_mean_per_class_accuracy = tf.identity(mean_per_class_accuracy)
        tf.summary.scalar('loss', summary_loss)
        tf.summary.scalar('accuracy', summary_accuracy)
        tf.summary.scalar('mean_per_class_accuracy', summary_mean_per_class_accuracy)

        return [summary_mean_per_class_accuracy, summary_accuracy, summary_loss], \
               [mean_per_class_accuracy, accuracy, loss], \
               [after_reset_mean_per_class_accuracy, after_reset_accuracy, after_reset_loss]


class BaseSegmentationTrainer(BaseTrainer):
    def __init__(self, num_classes=1000,
                 **kwargs):
        """
        图像分割任务的基本训练流程

        对 BaseTrainer 的改进：
        1. 添加了误差 tf.losses.sparse_softmax_cross_entropy
        2. 添加三个性能指标 mean_iou, accuracy, loss

        kwargs 举例如下：
        {
            'batch_size': 32,
            'weight_decay': 0.0005,
            'keep_prob': 0.8,

            'learning_rate_type': 0,
            'learning_rate_start': 0.0001,
            'lr_decay_rate': None,
            'lr_decay_steps': None,
            'lr_staircase': None,
            'steps_to_lr_dict': None,
            'min_lr': None,
            'lr_shrink_epochs': None,
            'lr_shrink_by_number': None,

            'base_logs_dir': './logs'
            'val_logs_dir': 'val',

            'metrics_reset_ops_collection': 'reset_ops',

            'logging_every_n_steps': 1000,
            'summary_every_n_steps': 1000,
            'save_every_n_steps': 1000,
            'evaluate_every_n_steps': 10000,
            'max_steps': None,

            'fine_tune_steps': None,
            'fine_tune_file_path': None,
            'fine_tune_vars_include': None,
            'fine_tune_vars_exclude': None,
        }
        :param num_classes:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self._num_classes = num_classes

    def _get_merged_dataset(self):
        raise NotImplementedError

    def _get_optimizer(self):
        raise NotImplementedError

    def _get_model(self):
        raise NotImplementedError

    def _get_fine_tune_var_dict(self, variables_to_restore):
        raise NotImplementedError

    def _get_loss(self, logits):
        tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=self._y)
        return tf.losses.get_total_loss()

    def _get_metrics(self, logits, total_loss):
        predictions = tf.argmax(logits, axis=-1)
        summary_loss, loss = tf.metrics.mean(total_loss, name='loss')
        summary_accuracy, accuracy = tf.metrics.accuracy(self._y, predictions, name='accuracy')
        summary_mean_iou, confused_matrix = tf.metrics.mean_iou(tf.reshape(self._y, [-1]),
                                                                tf.reshape(predictions, [-1]),
                                                                self._num_classes,
                                                                name='confused_matrix')
        mean_iou = compute_mean_iou_by_confusion_matrix('mean_iou', confused_matrix)

        for metric in tf.get_collection(tf.GraphKeys.METRIC_VARIABLES):
            tf.add_to_collection(self._metrics_reset_ops_collection,
                                 tf.assign(metric, tf.zeros(metric.get_shape(), metric.dtype)))
        with tf.control_dependencies(tf.get_collection(self._metrics_reset_ops_collection)):
            after_reset_loss = tf.identity(loss)
            after_reset_accuracy = tf.identity(accuracy)
            after_reset_mean_iou = tf.identity(mean_iou)
        tf.summary.scalar('loss', summary_loss)
        tf.summary.scalar('accuracy', summary_accuracy)
        tf.summary.scalar('mean_iou', summary_mean_iou)

        return [summary_mean_iou, summary_accuracy, summary_loss], \
               [mean_iou, accuracy, loss], \
               [after_reset_mean_iou, after_reset_accuracy, after_reset_loss]
