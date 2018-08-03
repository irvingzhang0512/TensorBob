import tensorflow as tf
import tensorflow.contrib.slim as slim
import nets.vgg as vgg
import math
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

# voc2012 分类信息，共有20类
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

# basic
tf.flags.DEFINE_integer('EPOCH_STAGE_1', 5, 'epochs to train only fc8')
tf.flags.DEFINE_integer('EPOCH_STAGE_2', 30, 'epochs to train all variables')
tf.flags.DEFINE_integer('BATCH_SIZE', 64, 'batch size')
tf.flags.DEFINE_float('WEIGHT_DECAY', 0.00005, 'l2 loss')
tf.flags.DEFINE_float('KEEP_PROB', 0.5, 'dropout layer')
tf.flags.DEFINE_float('MOMENTUM', 0.9, 'optimizer momentum')
tf.flags.DEFINE_string('VOC2012_ROOT', "/home/ubuntu/data/voc2012/train/voc2012", 'where to store logs')
tf.flags.DEFINE_integer('TRAIN_IMAGE_SIZE', 224, '')
tf.flags.DEFINE_integer('VAL_IMAGE_SIZE', 384, '')
tf.flags.DEFINE_integer('NUM_CLASSES', 20, '')

# logging
tf.flags.DEFINE_integer('LOGGING_EVERY_N_STEPS', 10, 'logging in console every n steps')
tf.flags.DEFINE_string('LOGS_DIR', './logs/', 'where to store logs')

# learning rate
tf.flags.DEFINE_float('LEARNING_RATE_START', 0.001, 'learning rate at epoch 0')
tf.flags.DEFINE_integer('DECAY_STEPS', 500, 'learning rate decay var')
tf.flags.DEFINE_float('DECAY_RATE', 0.5, 'learning rate decay var')

# pre-trained model
tf.flags.DEFINE_boolean('USE_PRE_TRAINED_MODEL', True, 'whether or not use slim pre-trained model.')
tf.flags.DEFINE_string('PRE_TRAINED_MODEL_PATH', '/home/ubuntu/data/slim/vgg_19.ckpt', 'pre-trained model path.')

FLAGS = tf.app.flags.FLAGS


def get_dataset(mode='train', resize_image_min=256, resize_image_max=512, image_size=224):
    """
    获取数据集
    :param image_size:
    :param resize_image_max:
    :param resize_image_min:
    :param mode: 可以是 train val trainval 三者之一
    :return: tf.data.Dataset实例，以及数据集大小
    """

    def get_image_paths_and_labels():
        if mode not in ['train', 'val', 'trainval']:
            raise ValueError('Unknown mode: {}'.format(mode))
        result_dict = {}
        for i, class_name in enumerate(CLASSES):
            file_name = class_name + "_" + mode + '.txt'
            for line in open(os.path.join(FLAGS.VOC2012_ROOT, 'ImageSets', 'Main', file_name), 'r'):
                line = line.replace('  ', ' ').replace('\n', '')
                parts = line.split(' ')
                if int(parts[1]) == 1:
                    result_dict[os.path.join(FLAGS.VOC2012_ROOT, 'JPEGImages', parts[0] + '.jpg')] = i
        keys, values = [], []
        for key, value in result_dict.items():
            keys.append(key)
            values.append(value)
        return keys, values

    def norm_imagenet(image):
        means = [103.939, 116.779, 123.68]
        channels = tf.split(axis=2, num_or_size_splits=3, value=image)
        for i in range(3):
            channels[i] -= means[i]
        return tf.concat(axis=2, values=channels)

    def random_crop(images, cur_image_size):
        image_height = tf.shape(images)[-3]
        image_width = tf.shape(images)[-2]
        offset_height = tf.random_uniform([], 0, (image_height - cur_image_size + 1), dtype=tf.int32)
        offset_width = tf.random_uniform([], 0, (image_width - cur_image_size + 1), dtype=tf.int32)
        return tf.image.crop_to_bounding_box(images, offset_height, offset_width, cur_image_size, cur_image_size)

    # VGG中的图像增强
    def parse_image_by_path_fn(image_path):
        img_file = tf.read_file(image_path)
        cur_image = tf.image.decode_jpeg(img_file)
        cur_image = tf.image.resize_images(cur_image, [random_image_size, random_image_size])
        cur_image = tf.image.random_flip_left_right(cur_image)
        cur_image = random_crop(cur_image, image_size)
        cur_image = norm_imagenet(cur_image)
        return cur_image

    random_image_size = tf.random_uniform([], resize_image_min, resize_image_max, tf.int32)
    paths, labels = get_image_paths_and_labels()
    images_dataset = tf.data.Dataset.from_tensor_slices(paths).map(parse_image_by_path_fn)
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((images_dataset, labels_dataset))
    if mode == 'train':
        dataset = dataset.shuffle(buffer_size=len(paths))
    return dataset.batch(batch_size=FLAGS.BATCH_SIZE), len(paths)


def get_vgg_model(x, is_training):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(FLAGS.WEIGHT_DECAY),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME'):
            logits, _ = vgg.vgg_19(x,
                                   num_classes=FLAGS.NUM_CLASSES,
                                   is_training=is_training,
                                   dropout_keep_prob=FLAGS.KEEP_PROB,
                                   spatial_squeeze=True,
                                   scope='vgg_19',
                                   fc_conv_padding='VALID',
                                   global_pool=True)
    return logits


def val_set_test(sess, global_step, train_step_kwargs):
    sess.run(train_step_kwargs['val_iter_initializer'])
    while True:
        try:
            cur_images, cur_labels = sess.run(train_step_kwargs['val_next_element'])
            feed_dict = {train_step_kwargs['ph_x']: cur_images,
                         train_step_kwargs['ph_y']: cur_labels,
                         train_step_kwargs['ph_is_training']: False,
                         train_step_kwargs['ph_image_size']: FLAGS.VAL_IMAGE_SIZE}
            sess.run([train_step_kwargs['mean_val_loss_update_op'], train_step_kwargs['val_accuracy_update_op']],
                     feed_dict=feed_dict)
        except:
            break
    val_loss, val_accuracy = sess.run([train_step_kwargs['mean_val_loss'], train_step_kwargs['val_accuracy']])
    logger.info('epoch val loss is %.4f, val accuracy is %.4f' % (val_loss, val_accuracy))
    if val_accuracy > train_step_kwargs['best_val_accuracy']:
        train_step_kwargs['best_val_accuracy'] = val_accuracy
        if val_accuracy > 0.8:
            train_step_kwargs['saver'].save(sess, os.path.join(FLAGS.LOGS_DIR, 'model.ckpt'), global_step=global_step)
    return val_loss, val_accuracy


def train_step_vgg(sess, train_op, global_step, train_step_kwargs):
    """
    slim.learning中要调用的函数，代表一次梯度下降
    :param sess:
    :param train_op:
    :param global_step:
    :param train_step_kwargs:
    :return:
    """

    cur_global_step = sess.run(global_step)

    # 获取输入数据
    train_next_element = train_step_kwargs['train_next_element']
    try:
        cur_images, cur_labels = sess.run(train_next_element)
    except:
        if int(cur_global_step) != 0:
            val_set_test(sess, global_step, train_step_kwargs)

        train_iter_initializer = train_step_kwargs['train_iter_initializer']
        sess.run(train_iter_initializer)
        cur_images, cur_labels = sess.run(train_next_element)
        sess.run(train_step_kwargs['reset_metrics_ops'])

    # 进行训练
    feed_dict = {train_step_kwargs['ph_x']: cur_images,
                 train_step_kwargs['ph_y']: cur_labels,
                 train_step_kwargs['ph_is_training']: True,
                 train_step_kwargs['ph_image_size']: FLAGS.TRAIN_IMAGE_SIZE}
    train_op = train_op if cur_global_step < train_step_kwargs['stage_1_steps'] else train_step_kwargs['train_op2']
    cur_total_loss, cur_accuracy = sess.run(
        [train_op, train_step_kwargs['train_accuracy']],
        feed_dict=feed_dict)

    # logging
    if cur_global_step % train_step_kwargs['logging_every_n_steps'] == 0:
        logger.info('step %d: loss is %.4f, accuracy is %.3f.' % (cur_global_step, cur_total_loss, cur_accuracy))

    if cur_global_step + 1 >= train_step_kwargs['max_steps']:
        val_set_test(sess, global_step, train_step_kwargs)

    # 输出两个值
    return cur_total_loss, cur_global_step + 1 >= train_step_kwargs['max_steps']


def main(_):
    # 获取训练集
    train_set, train_set_size = get_dataset('train')
    train_iter = train_set.make_initializable_iterator()
    logger.info('train set created successfully with {} items.'.format(train_set_size))

    # 验证集
    val_set, val_set_size = get_dataset('val',
                                        image_size=FLAGS.VAL_IMAGE_SIZE,
                                        resize_image_min=FLAGS.VAL_IMAGE_SIZE,
                                        resize_image_max=FLAGS.VAL_IMAGE_SIZE+1)
    val_iter = val_set.make_initializable_iterator()
    logger.info('val set created successfully with {} items.'.format(val_set_size))

    # 构建tf.placeholder
    ph_image_size = tf.placeholder(tf.int32)
    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.int32, [None])
    ph_is_training = tf.placeholder(tf.bool)

    global_step = tf.train.get_or_create_global_step()

    # 学习率
    learning_rate = tf.train.exponential_decay(FLAGS.LEARNING_RATE_START, global_step,
                                               decay_rate=FLAGS.DECAY_RATE, decay_steps=FLAGS.DECAY_STEPS)

    # 模型相关
    logits = get_vgg_model(tf.reshape(ph_x, [-1, ph_image_size, ph_image_size, 3]), ph_is_training)
    tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=ph_y)
    total_loss = tf.losses.get_total_loss()
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=FLAGS.MOMENTUM)
    accuracy, _ = tf.metrics.accuracy(ph_y, tf.argmax(tf.nn.softmax(logits), axis=1),
                                      updates_collections=tf.GraphKeys.UPDATE_OPS)
    val_accuracy, val_accuracy_update_op = tf.metrics.accuracy(ph_y, tf.argmax(tf.nn.softmax(logits), axis=1))
    mean_val_loss, mean_val_loss_update_op = tf.metrics.mean(total_loss, name='mean_val_loss')
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    if FLAGS.USE_PRE_TRAINED_MODEL:
        variables_to_restore = slim.get_variables_to_restore(include=['vgg_19'], exclude=['vgg_19/fc8'])
        init_fn = slim.assign_from_checkpoint_fn(FLAGS.PRE_TRAINED_MODEL_PATH, variables_to_restore, True, True)
        logger.info('use pre-trained model with %d variables' % len(variables_to_restore))
    else:
        init_fn = None

    def get_train_step_kwargs():
        """
        获取 train_step 函数所需要的参数
        """
        metrics = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)
        logger.info('this graph has %d metrics' % len(metrics))
        reset_metrics_ops = []
        for metric in metrics:
            reset_metrics_ops.append(tf.assign(metric, 0))
        train_step_kwargs = {'max_steps': int(math.ceil(train_set_size / FLAGS.BATCH_SIZE)) * FLAGS.EPOCH_STAGE_2,
                             'logging_every_n_steps': FLAGS.LOGGING_EVERY_N_STEPS,
                             'train_next_element': train_iter.get_next(),
                             'train_iter_initializer': train_iter.initializer,
                             'train_accuracy': accuracy,
                             'ph_x': ph_x,
                             'ph_y': ph_y,
                             'ph_is_training': ph_is_training,
                             'ph_image_size': ph_image_size,
                             'reset_metrics_ops': reset_metrics_ops,
                             'val_next_element': val_iter.get_next(),
                             'val_iter_initializer': val_iter.initializer,
                             'mean_val_loss': mean_val_loss,
                             'mean_val_loss_update_op': mean_val_loss_update_op,
                             'val_accuracy': val_accuracy,
                             'val_accuracy_update_op': val_accuracy_update_op,
                             'saver': saver,
                             'best_val_accuracy': .0,
                             'train_op2': train_op2,
                             'stage_1_steps': int(math.ceil(train_set_size / FLAGS.BATCH_SIZE)) * FLAGS.EPOCH_STAGE_1
                             }
        return train_step_kwargs

    logger.debug('there are %d model variables, %d trainable variables, %d local variables and %d global variables' % (
        len(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)),
        len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)),
        len(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)),
        len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    ))

    train_op1 = slim.learning.create_train_op(total_loss, optimizer,
                                              variables_to_train=slim.get_variables_to_restore(include=['vgg_19/fc8']),
                                              global_step=global_step)
    train_op2 = slim.learning.create_train_op(total_loss, optimizer, global_step=global_step)
    slim.learning.train(train_op1,
                        logdir=FLAGS.LOGS_DIR,
                        train_step_fn=train_step_vgg,
                        train_step_kwargs=get_train_step_kwargs(),
                        init_fn=init_fn,
                        global_step=global_step,
                        save_interval_secs=None,
                        save_summaries_secs=None)


if __name__ == '__main__':
    tf.app.run()
