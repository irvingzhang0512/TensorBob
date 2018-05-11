import tensorflow as tf
import sys

sys.path.append('/home/ubuntu/bob/TensorBob/')
sys.path.append('/home/ubuntu/bob/models/research/')
import tensorbob as bob
import slim.nets.vgg as vgg
import tensorflow.contrib.slim as slim
import logging

logging.getLogger('tensorflow').setLevel(logging.DEBUG)

# 基本参数
BATCH_SIZE = 16
EPOCHS = 20
CROP_IMAGE_SIZE = 224
NUM_CLASSES = 20
WEIGHT_DECAY = 0.00005
KEEP_PROB = 0.5
LEARNING_RATE_START = 0.001
DECAY_RATE = 0.5
DECAY_STEPS = 500
LOGS_DIR = './logs/'
CKPT_FILE_PATH = './logs/model.ckpt'
TRAIN_LOGS_DIR = './logs/train/'
VAL_LOGS_DIR = './logs/val/'
PRE_TRAINED_MODEL_PATH = '/home/ubuntu/data/slim/vgg_19.ckpt'

# 训练过程参数
VAL_SINGLE_IMAGE_SIZE = 384  # 使用验证集时，图片的尺寸
VAL_EVERY_N_STEPS = 360  # 使用验证集评估模型的频率
LOGGING_EVERY_N_STEPS = 30  # 输出测试信息的频率
SUMMARY_EVERY_N_STEPS = 30  # 记录summary结果的频率
MAX_STEPS = 36000  # 最多训练的步数
FINE_TUNE_STAGE_ONE_STEPS = 1800  # finetune第一阶段的steps


def get_train_dataset():
    train_dataset_config = {
        'norm_fn': bob.data.norm_imagenet,
        'crop_width': CROP_IMAGE_SIZE,
        'crop_height': CROP_IMAGE_SIZE,
        'random_flip_horizontal_flag': True,
        'multi_scale_training_list': [256, 512],
    }
    train_dataset = bob.data.get_voc_classification_dataset('train', batch_size=BATCH_SIZE,
                                                            **train_dataset_config)
    tf.logging.debug('successfully created training dataset with size {}'.format(train_dataset.size))
    return train_dataset


def get_val_dataset():
    ph_val_image_size = tf.placeholder(tf.int32, [], 'val_image_size')
    val_dataset_config = {
        'norm_fn': bob.data.norm_imagenet,
        'image_width': ph_val_image_size,
        'image_height': ph_val_image_size
    }
    val_dataset = bob.data.get_voc_classification_dataset('val', BATCH_SIZE, **val_dataset_config)
    tf.logging.debug('successfully created val dataset with size {}'.format(val_dataset.size))
    return val_dataset, ph_val_image_size


def get_vgg_model(x, is_training):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME'):
            logits, _ = vgg.vgg_19(x,
                                   num_classes=NUM_CLASSES,
                                   is_training=is_training,
                                   dropout_keep_prob=KEEP_PROB,
                                   spatial_squeeze=True,
                                   scope='vgg_19',
                                   fc_conv_padding='VALID',
                                   global_pool=True)
    return logits


def get_train_op(total_loss, optimizer, global_step):
    fc8_variables = bob.variables.get_variables_to_restore(include=['vgg_19/fc8'])
    train_op_stage_one = bob.training.create_train_op(total_loss, optimizer, global_step,
                                                      variables_to_train=fc8_variables,
                                                      update_ops=tf.get_collection('val_metrics_update_ops'))
    train_op_stage_two = bob.training.create_train_op(total_loss, optimizer, global_step,
                                                      update_ops=tf.get_collection('val_metrics_update_ops'))
    train_op = bob.training.create_finetune_train_op(train_op_stage_one, train_op_stage_two,
                                                     FINE_TUNE_STAGE_ONE_STEPS, global_step)
    return train_op


def init_metrics(ph_y, predictions, total_loss):
    # metrics 相关
    mean_per_class_accuracy, _ = tf.metrics.mean_per_class_accuracy(ph_y, predictions, NUM_CLASSES, None,
                                                                    ['val_metrics'], ['val_metrics_update_ops'],
                                                                    'mean_accuracy_per_class')
    mean_loss, _ = tf.metrics.mean(total_loss, None, ['val_metrics'], ['val_metrics_update_ops'], 'mean_loss')
    mean_accuracy, _ = tf.metrics.accuracy(ph_y, predictions, None, ['val_metrics'], ['val_metrics_update_ops'],
                                           'mean_accuracy')
    for metric in tf.get_collection(tf.GraphKeys.METRIC_VARIABLES):
        tf.add_to_collection('val_metrics_reset_ops', tf.assign(metric, tf.zeros(metric.get_shape(), metric.dtype)))

    # summary 相关
    tf.summary.scalar('loss', mean_loss)
    tf.summary.scalar('accuracy', mean_accuracy)
    tf.summary.scalar('mean_per_class_accuracy', mean_per_class_accuracy)

    return mean_per_class_accuracy


def get_pre_trained_init_fn():
    variables_to_restore = bob.variables.get_variables_to_restore(include=['vgg_19'], exclude=['vgg_19/fc8'])
    init_fn = bob.variables.assign_from_checkpoint_fn(PRE_TRAINED_MODEL_PATH, variables_to_restore, True, True)

    def new_init_fn(scaffold, sess):
        init_fn(sess)

    return new_init_fn


def main(_):
    # 获取数据集
    train_dataset = get_train_dataset()
    val_dataset, ph_val_image_size = get_val_dataset()

    # 构建模型
    ph_image_size = tf.placeholder(tf.int32, name='image_size')
    ph_x = tf.placeholder(tf.float32, name='x')
    ph_y = tf.placeholder(tf.int32, [None], 'y')
    ph_is_training = tf.placeholder(tf.bool)
    logits = get_vgg_model(tf.cast(tf.reshape(ph_x, [-1, ph_image_size, ph_image_size, 3]), tf.float32), ph_is_training)
    predictions = tf.argmax(tf.nn.softmax(logits), axis=1)

    # 损失函数与优化器设置
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_START, global_step,
                                               decay_rate=DECAY_RATE, decay_steps=DECAY_STEPS)
    tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=ph_y)
    total_loss = tf.losses.get_total_loss()
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

    # train_op, metrics, summary, pre-trained model, scaffold
    main_metric = init_metrics(ph_y, predictions, total_loss)
    train_op = get_train_op(total_loss, optimizer, global_step)
    merged_summary = tf.summary.merge_all()
    train_summary_writer = tf.summary.FileWriter(TRAIN_LOGS_DIR)
    val_summary_writer = tf.summary.FileWriter(VAL_LOGS_DIR)
    scaffold = tf.train.Scaffold(init_fn=get_pre_trained_init_fn())

    # 创建hooks

    # 训练集数据feed_dict
    train_dataset_hook = bob.training_utils.TrainDatasetFeedDictHook(train_dataset, ph_x, ph_y)

    # 其他数据feed_dict
    def get_training_feed_dict():
        return {ph_is_training: True, ph_image_size: CROP_IMAGE_SIZE}

    train_feed_fn_hook = bob.training_utils.FeedFnHook(get_training_feed_dict)

    # 定期在验证集上调试模型，保存验证集上性能最好的模型
    evaluate_fn = bob.training_utils.evaluate_on_single_scale(VAL_SINGLE_IMAGE_SIZE,
                                                              ph_image_size, ph_x, ph_y,
                                                              ph_val_image_size,
                                                              ph_is_training,
                                                              tf.get_collection(
                                                                  'val_metrics_reset_ops'),
                                                              tf.get_collection(
                                                                  'val_metrics_update_ops'),
                                                              main_metric=main_metric
                                                              )
    validation_evaluate_hook = bob.training_utils.ValidationDatasetEvaluationHook(val_dataset,
                                                                                  VAL_EVERY_N_STEPS,
                                                                                  summary_op=merged_summary,
                                                                                  summary_writer=val_summary_writer,
                                                                                  saver_file_prefix=CKPT_FILE_PATH,
                                                                                  evaluate_fn=evaluate_fn
                                                                                  ),

    # 定期在console中输出性能指标
    logging_hook = tf.train.LoggingTensorHook(tf.get_collection('val_metrics'),
                                              LOGGING_EVERY_N_STEPS)

    # 定期summary
    train_summary_hook = tf.train.SummarySaverHook(scaffold=scaffold,
                                                   save_steps=SUMMARY_EVERY_N_STEPS,
                                                   summary_writer=train_summary_writer)

    # 设置训练的最大steps
    stop_hook = bob.training_utils.StopAtStepHook(MAX_STEPS)

    hooks = [
        logging_hook,
        train_dataset_hook,
        train_feed_fn_hook,
        validation_evaluate_hook,
        train_summary_hook,
        stop_hook,
    ]

    bob.training.train(train_op, LOGS_DIR, scaffold, hooks=hooks)
