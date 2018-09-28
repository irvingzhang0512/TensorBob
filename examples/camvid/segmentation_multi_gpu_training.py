import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import tensorflow as tf
import tensorbob as bob
from tensorflow.python.platform import tf_logging as logging

logging.set_verbosity(logging.DEBUG)

LOGS_DIR = "./logs-multi-gpu-fc-densenet"
VAL_LOGS_DIR = os.path.join(LOGS_DIR, 'val')

LOGGING_AND_SUMMARY_EVERY_N_STEPS = 20
SAVE_EVERY_N_STEPS = 1000
VALIDATION_EVERY_N_OPS = 100

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 960
CROP_HEIGHT = 512
CROP_WIDTH = 512
BATCH_SIZE_PER_GPU = 1
NUM_OF_GPU = 4
EPOCHS = 1000

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
NUM_CLASSES = 32
KEEP_PROB = 0.8


def get_dataset():
    # 获取数据集
    train_configs = {
        'norm_fn_first': bob.preprocessing.norm_zero_to_one,
        'norm_fn_end': bob.preprocessing.norm_minus_one_to_one,
        'crop_type': bob.data.CropType.random_normal,
        'image_width': IMAGE_HEIGHT,
        'image_height': IMAGE_WIDTH,
        'crop_height': CROP_HEIGHT,
        'crop_width': CROP_WIDTH,
        'random_distort_color_flag': True,
    }
    val_configs = {
        'norm_fn_first': bob.preprocessing.norm_zero_to_one,
        'norm_fn_end': bob.preprocessing.norm_minus_one_to_one,
        'crop_type': bob.data.CropType.random_normal,
        'image_width': IMAGE_HEIGHT,
        'image_height': IMAGE_WIDTH,
        'crop_height': CROP_HEIGHT,
        'crop_width': CROP_WIDTH,
    }
    return bob.data.get_camvid_segmentation_merged_dataset(train_configs, val_configs,
                                                           data_path='/ssd/zhangyiyang/CamVid',
                                                           batch_size=BATCH_SIZE_PER_GPU * NUM_OF_GPU,
                                                           repeat=EPOCHS,
                                                           shuffle_buffer_size=100)


def get_logits_and_loss(scope, cur_x, cur_y, is_training, ):
    cur_logits, _ = bob.segmentation.fc_densenet(cur_x,
                                                 num_classes=NUM_CLASSES,
                                                 is_training=is_training,
                                                 keep_prob=KEEP_PROB,
                                                 weight_decay=WEIGHT_DECAY,
                                                 mode="67", )
    tf.losses.sparse_softmax_cross_entropy(labels=cur_y, logits=cur_logits)
    losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
    return cur_logits, tf.add_n(losses, name='total_loss')


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        cur_grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)

            cur_grads.append(expanded_g)

        cur_grad = tf.concat(cur_grads, 0)
        cur_grad = tf.reduce_mean(cur_grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (cur_grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def get_total_loss():
    """
    softmax loss must div by NUM_OF_GPU
    :return:
    """
    return tf.add(tf.losses.get_regularization_loss(),
                  tf.div(tf.add_n(tf.losses.get_losses()), NUM_OF_GPU, name='softmax_loss'),
                  name='total_loss')


def get_metrics(logits, ground_truth, total_loss):
    predictions = tf.argmax(logits, axis=-1)
    summary_loss, loss = tf.metrics.mean(total_loss,
                                         name='loss')
    summary_accuracy, accuracy = tf.metrics.accuracy(ground_truth, predictions,
                                                     name='accuracy')
    summary_mean_iou, confused_matrix = tf.metrics.mean_iou(tf.reshape(ground_truth, [-1]),
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
    tf.summary.scalar('total_loss', summary_loss)
    tf.summary.scalar('accuracy', summary_accuracy)
    tf.summary.scalar('mean_iou', summary_mean_iou)

    return [summary_mean_iou, summary_accuracy, summary_loss], \
           [mean_iou, accuracy, loss], \
           [after_reset_mean_iou, after_reset_accuracy, after_reset_loss]


if __name__ == '__main__':
    # 构建计算图
    with tf.device('/cpu:0'):
        ph_is_training = tf.placeholder(tf.bool, name='is_training')
        global_step = tf.train.get_or_create_global_step()
        merged_dataset = get_dataset()

        # multi GPU
        images, labels = merged_dataset.next_batch
        split_images = tf.split(images, num_or_size_splits=NUM_OF_GPU, axis=0)
        split_labels = tf.split(labels, num_or_size_splits=NUM_OF_GPU, axis=0)
        grads_list = []
        logits_list = []
        optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, decay=0.995)
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(NUM_OF_GPU):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('part_%d' % i) as sc:
                        partial_logits, partial_loss = get_logits_and_loss(sc,
                                                                           split_images[i],
                                                                           split_labels[i],
                                                                           ph_is_training)

                        tf.get_variable_scope().reuse_variables()
                        grads = optimizer.compute_gradients(partial_loss)
                        grads_list.append(grads)
                        logits_list.append(partial_logits)

        # get grads for each GPU, cal mean grads, get train op
        grads = average_gradients(grads_list)
        train_op = optimizer.apply_gradients(grads, global_step=global_step)

        # metrics
        total_logits = tf.concat(logits_list, axis=0, name='logits')
        summary_metrics, update_metrics, after_reset_update_metrics = get_metrics(total_logits,
                                                                                  labels,
                                                                                  get_total_loss())

        # save & summary
        # tf.summary.image('images', images)
        # tf.summary.image('labels', tf.cast(labels, tf.uint8))
        summary_writer = tf.summary.FileWriter(LOGS_DIR, tf.get_default_graph(), max_queue=5, flush_secs=30)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()

    # 运行计算图
    # 获取 Session 参数配置
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    config.allow_soft_placement = True

    # 构建hooks
    # summary_hook_v1 = bob.training.SummarySaverHook(save_steps=LOGGING_AND_SUMMARY_EVERY_N_STEPS,
    #                                                 summary_writer=summary_writer,
    #                                                 summary_op=summary_op)
    val_feed_dict = {ph_is_training: False}
    val_summary_writer = tf.summary.FileWriter(VAL_LOGS_DIR, tf.get_default_graph(), max_queue=5, flush_secs=30)
    validation_hook = bob.training.ValidationDatasetEvaluationHook(merged_dataset,
                                                                   evaluate_every_n_steps=VALIDATION_EVERY_N_OPS,

                                                                   metrics_reset_ops=tf.get_collection('RESET_OPS'),
                                                                   metrics_update_ops=update_metrics,
                                                                   evaluating_feed_dict=val_feed_dict,

                                                                   summary_op=summary_op,
                                                                   summary_writer=val_summary_writer,

                                                                   saver_file_prefix=os.path.join(VAL_LOGS_DIR,
                                                                                                  'model.ckpt'),
                                                                   )
    init_fn_hook = bob.training.InitFnHook(merged_dataset.init)
    hooks = [validation_hook,
             init_fn_hook,
             # image_summary_op,
             ]

    # feed_fn
    def feed_fn():
        return {merged_dataset.ph_handle: merged_dataset.handle_strings[0],
                ph_is_training: True}

    # start training
    bob.training.train(train_op, LOGS_DIR,
                       session_config=config,
                       hooks=hooks,
                       logging_tensors=after_reset_update_metrics,
                       logging_every_n_steps=LOGGING_AND_SUMMARY_EVERY_N_STEPS,

                       feed_fn=feed_fn,

                       summary_every_n_steps=None,
                       summary_op=summary_op, summary_writer=summary_writer,

                       save_every_n_steps=SAVE_EVERY_N_STEPS, saver=saver,
                       )
