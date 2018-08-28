import unittest
import tensorflow as tf

from tensorbob.utils.preprocessing import norm_zero_to_one, norm_minus_one_to_one
from tensorbob.dataset.imagenet import get_imagenet_classification_dataset, get_imagenet_classification_merged_dataset
from tensorbob.dataset.dataset_utils import CropType


class ImageNetTest(unittest.TestCase):
    def test_imagenet_classification_dataset(self):
        train_configs = {
            'norm_fn_first': norm_zero_to_one,
            'norm_fn_end': norm_minus_one_to_one,
            'random_flip_horizontal_flag': True,
            'random_flip_vertical_flag': True,
            'random_distort_color_flag': True,
            'distort_color_fast_mode_flag': False,

            'crop_type': CropType.random_vgg,
            'crop_width': 224,
            'crop_height': 224,
            'vgg_image_size_min': 256,
            'vgg_image_size_max': 512,
        }
        d = get_imagenet_classification_dataset('val',
                                                repeat=1,
                                                shuffle_buffer_size=10000,
                                                prefetch_buffer_size=10000,
                                                labels_offset=0,
                                                **train_configs)
        with tf.Session() as sess:
            sess.run(d.iterator.initializer)
            print('dataset size is', d.size)
            for _ in range(10):
                images, labels = sess.run(d.next_batch)
                print(images.shape, labels)

        self.assertTrue(True, 'Always True')

    def test_imagenet_classification_merged_dataset(self):
        train_args = {
            'norm_fn_first': norm_zero_to_one,
            'norm_fn_end': norm_minus_one_to_one,
            'random_flip_horizontal_flag': True,
            'random_flip_vertical_flag': True,
            'random_distort_color_flag': True,
            'distort_color_fast_mode_flag': False,

            # inception随机切片
            'crop_type': CropType.random_inception,
            'crop_width': 224,
            'crop_height': 224,
            'inception_bbox': None,
        }
        val_args = {
            'norm_fn_first': norm_zero_to_one,
            'norm_fn_end': norm_minus_one_to_one,
            'crop_type': CropType.no_crop,
            'image_width': 224,
            'image_height': 224,
        }

        d = get_imagenet_classification_merged_dataset(train_args, val_args)
        with tf.Session() as sess:
            d.init(sess)
            sess.run(d.tf_dataset_2_iterator.initializer)
            for i in range(5):
                images, labels = d.get_next_batch(sess, 0)
                print(images.shape, labels.shape)

            for i in range(5):
                images, lables = d.get_next_batch(sess, 1)
                print(images.shape, lables.shape)
        self.assertTrue(True, 'Always True')
