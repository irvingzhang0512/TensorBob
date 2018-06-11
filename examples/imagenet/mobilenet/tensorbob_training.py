import tensorflow as tf
import sys
import argparse
import logging

sys.path.append('/home/tensorflow05/zyy/tensorbob/')
sys.path.append('/home/tensorflow05/zyy/models/research/slim/')
import tensorbob as bob
import numpy as np
from nets import nets_factory

logger = logging.getLogger('tensorflow')
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()

# 基本参数
parser.add_argument('--NUM_CLASSES', help='数据集类别数量', type=int, default=1000)
parser.add_argument('--BATCH_SIZE', help='batch size', type=int, default=32)
parser.add_argument('--WEIGHT_DECAY', help='L2范数常量', type=float, default=0.00005)
parser.add_argument('--KEEP_PROB', help='池化层保留神经元比例', type=float, default=0.5)

# 学习率
parser.add_argument('--LEARNING_RATE_START', help='学习率初始值', type=float, default=0.001)
parser.add_argument('--DECAY_RATE', help='学习率衰减比率', type=float, default=0.5)
parser.add_argument('--DECAY_STEPS', help='学习率衰减steps数量', type=int, default=40000 * 5)
parser.add_argument('--STAIRCASE', help='学习率衰减steps数量', type=bool, default=False)

# 路径
parser.add_argument('--LOGS_DIR', help='保存训练过程中，ckpt的路径', type=str, default='./logs/ckpt/')
parser.add_argument('--CKPT_FILE_PATH', help='验证集中性能最好的模型路径', type=str, default='./logs/best-val/model.ckpt')
parser.add_argument('--TRAIN_LOGS_DIR', help='训练过程中summary路径', type=str, default='./logs/train/')
parser.add_argument('--VAL_LOGS_DIR', help='验证过程中summary路径', type=str, default='./logs/val/')
parser.add_argument('--DATA_ROOT', help='', type=str, default='/home/tensorflow05/data/ILSVRC2012')

# 训练过程中过程中参数
parser.add_argument('--TRAIN_CROP_IMAGE_SIZE', help='训练阶段，图片切片尺寸', type=int, default=224)
parser.add_argument('--TRAIN_MIN_IMAGE_SIZE', help='训练阶段，图片缩放最小尺寸', type=int, default=256)
parser.add_argument('--TRAIN_MAX_IMAGE_SIZE', help='训练阶段，图片缩放最大尺寸', type=int, default=512)
parser.add_argument('--VAL_SINGLE_IMAGE_SIZE', help='验证阶段，图片尺寸', type=int, default=384)
parser.add_argument('--VAL_EVERY_N_STEPS', help='每经过多少steps，在验证集上评估一次模型', type=int, default=40000)
parser.add_argument('--LOGGING_EVERY_N_STEPS', help='每经过多少steps，在命令行中输出一次metrics', type=int, default=1000)
parser.add_argument('--SUMMARY_EVERY_N_STEPS', help='每经过多少steps，summary一次数据', type=int, default=1000)
parser.add_argument('--SAVE_EVERY_N_STEPS', help='每经过多少steps，summary一次数据save一次', type=int, default=5000)
parser.add_argument('--MAX_STEPS', help='训练最大步数', type=int, default=40000 * 80)

# metrics相关
parser.add_argument('--METRICS_COLLECTION', help='', type=str, default='val_metrics')
parser.add_argument('--METRICS_UPDATE_OPS_COLLECTION', help='', type=str, default='update_ops')
parser.add_argument('--METRICS_RESET_OPS_COLLECTION', help='', type=str, default='reset_ops')
parser.add_argument('--USING_MEAN_METRICS', help='', type=bool, default=False)

args = parser.parse_args()


def get_dataset():
    train_dataset_config = {
        'norm_fn_first': bob.data.norm_zero_to_one,
        'norm_fn_last': bob.data.norm_minus_one_to_one,
        'crop_type': bob.data.CropType.random_vgg,
        'crop_width': args.TRAIN_CROP_IMAGE_SIZE,
        'crop_height': args.TRAIN_CROP_IMAGE_SIZE,
        'vgg_image_size_min': args.TRAIN_MIN_IMAGE_SIZE,
        'vgg_image_size_max': args.TRAIN_MAX_IMAGE_SIZE,
        'random_flip_horizontal_flag': True,
    }
    train_dataset = bob.data.get_imagenet_classification_dataset('train',
                                                                 args.BATCH_SIZE,
                                                                 args.DATA_ROOT, 100, 100,
                                                                 **train_dataset_config)
    val_dataset_config = {
        'norm_fn_first': bob.data.norm_zero_to_one,
        'norm_fn_last': bob.data.norm_minus_one_to_one,
        'image_width': args.VAL_SINGLE_IMAGE_SIZE,
        'image_height': args.VAL_SINGLE_IMAGE_SIZE,
    }
    val_dataset = bob.data.get_imagenet_classification_dataset('val',
                                                               args.BATCH_SIZE,
                                                               args.DATA_ROOT,
                                                               **val_dataset_config)
    logger.debug('val set size is {}'.format(val_dataset.size))
    logger.debug('training set size is {}'.format(train_dataset.size))
    return train_dataset, val_dataset


def get_model(x, is_training):
    model_fn = nets_factory.get_network_fn("mobilenet_v2", args.NUM_CLASSES, args.WEIGHT_DECAY, is_training)
    logits, _ = model_fn(x, dropout_keep_prob=args.KEEP_PROB)
    return logits


def init_metrics(ph_y, predictions, total_loss, ph_using_mean_metrics):
    mean_loss, _ = tf.metrics.mean(total_loss,
                                   metrics_collections=[args.METRICS_COLLECTION],
                                   updates_collections=[args.METRICS_UPDATE_OPS_COLLECTION],
                                   name='mean_loss')
    non_mean_loss = total_loss
    loss = tf.case([(ph_using_mean_metrics, lambda: mean_loss)], default=lambda: non_mean_loss)
    mean_accuracy, _ = tf.metrics.accuracy(ph_y, predictions,
                                           metrics_collections=[args.METRICS_COLLECTION],
                                           updates_collections=[args.METRICS_UPDATE_OPS_COLLECTION],
                                           name='mean_accuracy')
    non_mean_accuracy = tf.reduce_mean(tf.cast(tf.equal(ph_y, predictions), tf.float32))
    accuracy = tf.case([(ph_using_mean_metrics, lambda: mean_accuracy)], default=lambda: non_mean_accuracy)

    for metric in tf.get_collection(tf.GraphKeys.METRIC_VARIABLES):
        tf.add_to_collection(args.METRICS_RESET_OPS_COLLECTION,
                             tf.assign(metric, tf.zeros(metric.get_shape(), metric.dtype)))

    # summary 相关
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    return [accuracy, loss]


def main():
    # 获取数据集
    train_dataset, val_dataset = get_dataset()

    # 构建模型，获取模型结果
    ph_image_size = tf.placeholder(tf.int32, name='image_size')
    ph_x = tf.placeholder(tf.float32, name='x')
    ph_y = tf.placeholder(tf.int64, [None], 'y')
    ph_is_training = tf.placeholder(tf.bool)
    ph_using_mean_metrics = tf.placeholder(tf.bool)
    logits = get_model(tf.cast(tf.reshape(ph_x, [-1, ph_image_size, ph_image_size, 3]), tf.float32), ph_is_training)
    predictions = tf.argmax(tf.nn.softmax(logits), axis=1)

    # 损失函数与优化器设置
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(args.LEARNING_RATE_START,
                                               global_step,
                                               decay_rate=args.DECAY_RATE,
                                               decay_steps=args.DECAY_STEPS,
                                               staircase=args.STAIRCASE)
    tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=ph_y)
    total_loss = tf.losses.get_total_loss()
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

    # metrics & train_op & summary writer
    cur_metrics = init_metrics(ph_y, predictions, total_loss, ph_using_mean_metrics)
    train_op = bob.training.create_train_op(total_loss, optimizer, global_step,
                                            update_ops=tf.get_collection(args.METRICS_UPDATE_OPS_COLLECTION))
    merged_summary = tf.summary.merge_all()
    train_summary_writer = tf.summary.FileWriter(args.TRAIN_LOGS_DIR, tf.get_default_graph())
    val_summary_writer = tf.summary.FileWriter(args.VAL_LOGS_DIR, tf.get_default_graph())

    # hooks
    train_dataset_hook = bob.training.TrainDatasetFeedDictHook(train_dataset, ph_x, ph_y)
    evaluate_fn = bob.training.evaluate_on_single_scale(args.VAL_SINGLE_IMAGE_SIZE,
                                                        ph_x, ph_y,
                                                        {ph_image_size: args.VAL_SINGLE_IMAGE_SIZE,
                                                         ph_is_training: False,
                                                         ph_using_mean_metrics: True},
                                                        None,
                                                        tf.get_collection(args.METRICS_RESET_OPS_COLLECTION),
                                                        tf.get_collection(args.METRICS_UPDATE_OPS_COLLECTION),
                                                        main_metric=cur_metrics[0])
    summary_feed_dict = {
        ph_image_size: args.VAL_SINGLE_IMAGE_SIZE,
        ph_is_training: False,
        ph_using_mean_metrics: True,
        ph_x: np.zeros([args.BATCH_SIZE, args.VAL_SINGLE_IMAGE_SIZE, args.VAL_SINGLE_IMAGE_SIZE, 3]),
        ph_y: np.zeros([args.BATCH_SIZE])
    }
    validation_evaluate_hook = bob.training.ValidationDatasetEvaluationHook(val_dataset,
                                                                            args.VAL_EVERY_N_STEPS,
                                                                            summary_op=merged_summary,
                                                                            summary_writer=val_summary_writer,
                                                                            saver_file_prefix=args.CKPT_FILE_PATH,
                                                                            evaluate_fn=evaluate_fn,
                                                                            summary_feed_dict=summary_feed_dict)
    hooks = [train_dataset_hook, validation_evaluate_hook]

    # 训练
    def get_training_feed_dict():
        return {ph_is_training: True,
                ph_image_size: args.TRAIN_CROP_IMAGE_SIZE,
                ph_using_mean_metrics: args.USING_MEAN_METRICS}

    bob.training.train(train_op, args.LOGS_DIR, hooks=hooks,
                       max_steps=args.MAX_STEPS,

                       logging_tensors=cur_metrics,
                       logging_every_n_steps=args.LOGGING_EVERY_N_STEPS,

                       feed_fn=get_training_feed_dict,

                       summary_writer=train_summary_writer, summary_op=merged_summary,
                       summary_every_n_steps=args.SUMMARY_EVERY_N_STEPS,

                       save_every_n_steps=args.SAVE_EVERY_N_STEPS)


if __name__ == '__main__':
    main()
