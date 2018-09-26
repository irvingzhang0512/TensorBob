import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import tensorflow as tf
import tensorbob as bob
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework.errors_impl import OutOfRangeError

logging.set_verbosity(logging.DEBUG)

LOGS_DIR = "./logs-multi-gpu-fc-densenet"
VAL_LOGS_DIR = os.path.join(LOGS_DIR, 'val')

LOGGING_AND_SUMMARY_EVERY_N_STEPS = 10
SAVE_EVERY_N_STEPS = 150

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 960
CROP_HEIGHT = 512
CROP_WIDTH = 512
BATCH_SIZE_PER_GPU = 1
NUM_OF_GPU = 3
EPOCHS = 1000

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
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


def get_loss(scope, cur_x, cur_y, is_training, ):
    cur_logits, _ = bob.segmentation.fc_densenet(cur_x,
                                                 num_classes=NUM_CLASSES,
                                                 is_training=is_training,
                                                 keep_prob=KEEP_PROB,
                                                 weight_decay=WEIGHT_DECAY,
                                                 mode="67", )
    tf.losses.sparse_softmax_cross_entropy(labels=cur_y, logits=cur_logits)
    losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
    return tf.add_n(losses, name='total_loss')


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


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        ph_is_training = tf.placeholder(tf.bool, name='is_training')
        global_step = tf.train.get_or_create_global_step()
        merged_dataset = get_dataset()

        # multi GPU
        images, labels = merged_dataset.next_batch
        split_images = tf.split(images, num_or_size_splits=NUM_OF_GPU, axis=0)
        split_labels = tf.split(labels, num_or_size_splits=NUM_OF_GPU, axis=0)
        grads_list = []
        optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, decay=0.995)
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(NUM_OF_GPU):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('part_%d' % i) as sc:
                        cur_loss = get_loss(sc, split_images[i], split_labels[i], ph_is_training)

                        tf.get_variable_scope().reuse_variables()
                        grads = optimizer.compute_gradients(cur_loss)
                        grads_list.append(grads)

        # get grads for each GPU, cal mean grads, get train op
        grads = average_gradients(grads_list)
        train_op = optimizer.apply_gradients(grads, global_step=global_step)
        total_loss = tf.losses.get_total_loss()

        # save & summary
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
        tf.summary.scalar('loss', total_loss)
        summary_writer = tf.summary.FileWriter(LOGS_DIR, tf.get_default_graph(), max_queue=5, flush_secs=30)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        merged_dataset.init(sess)
        print('init successfully...')

        try:
            cur_global_step = 0
            training_feed_dict = {merged_dataset.ph_handle: merged_dataset.handle_strings[0],
                                  ph_is_training: True}
            while True:
                if cur_global_step % LOGGING_AND_SUMMARY_EVERY_N_STEPS == 0 and cur_global_step != 0:
                    _, summary_str, cur_global_step = sess.run([train_op, summary_op, global_step],
                                                               feed_dict=training_feed_dict)
                    summary_writer.add_summary(summary_str, global_step=cur_global_step)
                    print('%d: add summary successfully...' % cur_global_step)
                else:
                    _, cur_global_step = sess.run([train_op, global_step], feed_dict=training_feed_dict)

                if cur_global_step % SAVE_EVERY_N_STEPS == 0 and cur_global_step != 0:
                    saver.save(sess, os.path.join(LOGS_DIR, 'model.ckpt'), global_step=cur_global_step)
        except OutOfRangeError:
            pass
