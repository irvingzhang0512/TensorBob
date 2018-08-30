import tensorbob as bob
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import os

logging.set_verbosity(logging.DEBUG)

LOGS_DIR = "./logs-resnet"
VAL_LOGS_DIR = os.path.join(LOGS_DIR, 'val')

LOGGING_AND_SUMMARY_EVERY_N_STEPS = 100
VALIDATION_EVERY_N_OPS = 1300
SAVE_EVERY_N_STEPS = 2000

PRE_TRAINED_MODEL_PATH = "/home/tensorflow05/data/pre-trained/slim/resnet_v2_50_2017_04_14/resnet_v2_50.ckpt"
# PRE_TRAINED_MODEL_PATH = "/home/tensorflow05/data/pre-trained/slim/vgg_16.ckpt"
# PRE_TRAINED_MODEL_PATH = None

IMAGE_SIZE = 224
VAL_SIZE = 200
BATCH_SIZE = 16
EPOCHS = 1000

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.00005
NUM_CLASSES = 151
KEEP_PROB = 0.8


def get_dataset():
    # 获取数据集
    train_configs = {
        #         'norm_fn_first': bob.preprocessing.norm_imagenet,
        'norm_fn_first': bob.preprocessing.norm_zero_to_one,
        'norm_fn_end': bob.preprocessing.norm_minus_one_to_one,
        'crop_type': bob.data.CropType.no_crop,
        'image_width': IMAGE_SIZE,
        'image_height': IMAGE_SIZE,
        'random_distort_color_flag': True,
    }
    val_configs = {
        #         'norm_fn_first': bob.preprocessing.norm_imagenet,
        'norm_fn_first': bob.preprocessing.norm_zero_to_one,
        'norm_fn_end': bob.preprocessing.norm_minus_one_to_one,
        'crop_type': bob.data.CropType.no_crop,
        'image_width': IMAGE_SIZE,
        'image_height': IMAGE_SIZE,
    }
    return bob.data.get_ade_segmentation_merged_dataset(train_configs, val_configs,
                                                        batch_size=BATCH_SIZE,
                                                        repeat=EPOCHS,
                                                        label_image_height=IMAGE_SIZE,
                                                        label_image_width=IMAGE_SIZE,
                                                        shuffle_buffer_size=100)


def get_model(x, is_training):
    return bob.segmentation.resnet50_fcn_8s(x,
                                            num_classes=NUM_CLASSES,
                                            is_training=is_training,
                                            weight_decay=WEIGHT_DECAY)


#     return bob.segmentation.vgg16_fcn_8s(images,
#                                          num_classes=NUM_CLASSES,
#                                          is_training=is_training,
#                                          keep_prob=KEEP_PROB,
#                                          weight_decay=WEIGHT_DECAY)


def get_metrics(logits, labels, total_loss):
    predictions = tf.argmax(logits, axis=-1)
    summary_loss, loss = tf.metrics.mean(total_loss,
                                         name='loss')
    summary_accuracy, accuracy = tf.metrics.accuracy(labels, predictions,
                                                     name='accuracy')
    summary_mean_iou, confused_matrix = tf.metrics.mean_iou(tf.reshape(labels, [-1]),
                                                            tf.reshape(predictions, [-1]),
                                                            NUM_CLASSES,
                                                            name='confused_matrix')
    mean_iou = bob.metrics_utils.compute_mean_iou_by_confusion_matrix('mean_iou', confused_matrix)

    for metric in tf.get_collection(tf.GraphKeys.METRIC_VARIABLES):
        tf.add_to_collection('RESET_OPS',
                             tf.assign(metric, tf.zeros(metric.get_shape(), metric.dtype)))
    with tf.control_dependencies(tf.get_collection('RESET_OPS')):
        after_reset_loss = tf.identity(loss)
        after_reset_accuracy = tf.identity(accuracy)
        after_reset_mean_iou = tf.identity(mean_iou)
    tf.summary.scalar('loss', summary_loss)
    tf.summary.scalar('accuracy', summary_accuracy)
    tf.summary.scalar('mean_iou', summary_mean_iou)

    return [summary_mean_iou, summary_accuracy, summary_loss], \
           [mean_iou, accuracy, loss], \
           [after_reset_mean_iou, after_reset_accuracy, after_reset_loss]


def get_pre_trained_init_fn(pre_trained_model_path):
    if pre_trained_model_path is None:
        return None

    variables_to_restore = bob.variables.get_variables_to_restore(
        include=['resnet50_fcn_8s/resnet_v2_50'],
        # exclude=['resnet50_fcn_8s/logtis', 'resnet50_fcn_8s/predictions'],
        # include=['vgg16_fcn_8s/vgg_16'],
        # exclude=['vgg16_fcn_8s/vgg_16/fc8'],
                                                                  )
    var_dict = {}
    for var in variables_to_restore:
        var_name = var.name[var.name.find('/') + 1:var.name.find(':')]
        var_dict[var_name] = var
        logging.debug(var_name, var)

    logging.debug('restore %d variables' % len(var_dict))
    return bob.variables.assign_from_checkpoint_fn(pre_trained_model_path,
                                                   var_dict,
                                                   ignore_missing_vars=True,
                                                   reshape_variables=True)


if __name__ == '__main__':
    # 各种参数
    ph_is_training = tf.placeholder(tf.bool, name='is_training')
    global_step = tf.train.get_or_create_global_step()

    # 获取数据集
    merged_dataset = get_dataset()

    # 搭建网络
    images, labels = merged_dataset.next_batch
    logits, _ = get_model(images, ph_is_training)
    pre_trained_init_fn = get_pre_trained_init_fn(PRE_TRAINED_MODEL_PATH)

    # 获取train_op
    tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    total_loss = tf.losses.get_total_loss()
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_op = bob.training.create_train_op(total_loss,
                                            optimizer,
                                            global_step=global_step,
                                            )

    # 获取性能指标
    summary_metrics, update_metrics, after_reset_metrics = get_metrics(logits, labels, total_loss)
    summary_op = tf.summary.merge_all()

    # 构建hooks
    val_feed_dict = {ph_is_training: False}
    validation_hook = bob.training.ValidationDatasetEvaluationHook(merged_dataset,
                                                                   evaluate_every_n_steps=VALIDATION_EVERY_N_OPS,

                                                                   metrics_reset_ops=tf.get_collection('RESET_OPS'),
                                                                   metrics_update_ops=update_metrics,
                                                                   evaluating_feed_dict=val_feed_dict,

                                                                   summary_op=summary_op,
                                                                   summary_writer=tf.summary.FileWriter(VAL_LOGS_DIR,
                                                                                                        tf.get_default_graph()),

                                                                   saver_file_prefix=os.path.join(VAL_LOGS_DIR, 'model.ckpt'),
                                                                   )
    init_fn_hook = bob.training.InitFnHook(merged_dataset.init)
    hooks = [validation_hook, init_fn_hook]


    def init_fn(scaffold, session):
        if pre_trained_init_fn:
            pre_trained_init_fn(session)
        logging.debug('init_fn successfully processed...')


    scaffold = tf.train.Scaffold(init_fn=init_fn)


    def feed_fn():
        return {merged_dataset.ph_handle: merged_dataset.handle_strings[0],
                ph_is_training: True}


    bob.training.train(train_op, LOGS_DIR,
                       scaffold=scaffold,
                       hooks=hooks,
                       logging_tensors=after_reset_metrics,
                       logging_every_n_steps=LOGGING_AND_SUMMARY_EVERY_N_STEPS,
                       feed_fn=feed_fn,
                       summary_every_n_steps=LOGGING_AND_SUMMARY_EVERY_N_STEPS,
                       summary_op=summary_op,
                       summary_writer=tf.summary.FileWriter(LOGS_DIR, tf.get_default_graph()),
                       save_every_n_steps=SAVE_EVERY_N_STEPS,
                       )
