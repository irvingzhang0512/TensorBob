import tensorbob as bob
import tensorflow as tf
import cv2
import os
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework.errors_impl import OutOfRangeError
logging.set_verbosity(logging.DEBUG)

PRE_TRAINED_MODEL_PATH = "/home/tensorflow05/zyy/tensorbob/examples/ade/logs-resnet/val/model.ckpt-33800"
TEST_FILE_PATH = "/home/tensorflow05/data/ade/ADEChallengeData2016/list.txt"
OUTPUT_DIR = '/home/tensorflow05/data/ade/ADEChallengeData2016/annotations/testing/'
WEIGHT_DECAY = 0.00005
NUM_CLASSES = 151


if __name__ == '__main__':
    configs = {
        'norm_fn_first': bob.preprocessing.norm_zero_to_one,
        'norm_fn_end': bob.preprocessing.norm_minus_one_to_one,
        'crop_type': bob.data.CropType.no_crop,
    }
    test_dataset = bob.data.get_ade_segmentation_dataset(mode='test', batch_size=1, **configs)

    # 搭建网络
    images = test_dataset.next_batch
    logits, _ = bob.segmentation.resnet50_fcn_8s(images[0],
                                                 num_classes=NUM_CLASSES,
                                                 is_training=False,
                                                 weight_decay=WEIGHT_DECAY)
    predictions = tf.cast(tf.argmax(logits, axis=-1), tf.uint8)
    predictions = tf.squeeze(predictions, [0])
    predictions = tf.expand_dims(predictions, -1)

    with open(os.path.join(TEST_FILE_PATH)) as f:
        test_file_names = f.readlines()
    test_file_names = [image_file_name.replace('\n', '')[:image_file_name.find('.')] + '.png' for image_file_name in
                       test_file_names]
    test_file_names = [
        os.path.join(OUTPUT_DIR, image_file_name) for
        image_file_name in test_file_names]

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, PRE_TRAINED_MODEL_PATH)
        logging.debug('restore successfully...')

        sess.run(test_dataset.iterator.initializer)
        logging.debug('init test dataset successfully...')

        i = 0
        while True:
            try:
                cur_res = sess.run(predictions)
                cv2.imwrite(test_file_names[i], cur_res)
                i += 1
                if i % 100 == 0:
                    logging.debug('successfully predicting %d images' % i)
            except OutOfRangeError:
                break
        logging.debug('predicting successfully...')
