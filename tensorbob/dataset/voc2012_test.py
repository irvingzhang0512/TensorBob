import unittest
import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError

from .preprocessing import norm_imagenet
from .voc2012 import get_voc_classification_dataset


class Voc2012Test(unittest.TestCase):
    @unittest.skip
    def test_vgg_train(self):
        dataset_config = {
            'norm_fn': norm_imagenet,
            'crop_width': 224,
            'crop_height': 224,
            'central_crop_flag': True,
            'random_flip_horizontal_flag': True,
            'random_flip_vertical_flag': True,
            'multi_scale_training_list': [256, 512],
        }
        dataset = get_voc_classification_dataset('train', 32, **dataset_config)
        with tf.Session() as sess:
            total_cnt = 0
            while True:
                try:
                    images, labels = dataset.get_next_batch(sess)
                    total_cnt += images.shape[0]
                    self.assertEqual(images.shape[1], 224)
                    self.assertEqual(images.shape[2], 224)
                    self.assertEqual(images.shape[3], 3)
                except OutOfRangeError:
                    break
        self.assertEqual(total_cnt, dataset.size)

    @unittest.skip
    def test_val_multi_scale(self):
        multi_scales = [256, 384, 512]
        ph_image_size = tf.placeholder(tf.int32, [])
        dataset_config = {
            'norm_fn': norm_imagenet,
            'image_width': ph_image_size,
            'image_height': ph_image_size
        }
        dataset = get_voc_classification_dataset('val', 32, **dataset_config)
        for multi_scale in multi_scales:
            with tf.Session() as sess:
                total_cnt = 0
                dataset.reset(sess, feed_dict={ph_image_size: multi_scale})
                while True:
                    try:
                        images, labels = dataset.get_next_batch(sess)
                        total_cnt += images.shape[0]
                        self.assertEqual(images.shape[1], multi_scale)
                        self.assertEqual(images.shape[2], multi_scale)
                        self.assertEqual(images.shape[3], 3)
                    except OutOfRangeError:
                        break
            self.assertEqual(total_cnt, dataset.size)
