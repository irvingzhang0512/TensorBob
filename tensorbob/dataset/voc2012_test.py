import unittest
import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError

from tensorbob.utils.preprocessing import norm_zero_to_one, norm_minus_one_to_one
from tensorbob.dataset.voc2012 import get_voc_classification_dataset, get_voc_segmentation_merged_dataset
from tensorbob.dataset.dataset_utils import CropType


class Voc2012Test(unittest.TestCase):
    def test_classification(self):
        dataset_config = {
            'norm_fn_first': norm_zero_to_one,
            'norm_fn_end': norm_minus_one_to_one,
            'random_flip_horizontal_flag': False,
            'random_flip_vertical_flag': False,
            'random_distort_color_flag': True,
            'distort_color_fast_mode_flag': False,

            'crop_type': CropType.no_crop,
            'image_width': 224,
            'image_height': 224,
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

    def test_segmentation_dataset(self):
        train_configs = {
            'norm_fn_first': norm_zero_to_one,
            'norm_fn_end': norm_minus_one_to_one,
            'random_flip_horizontal_flag': False,
            'random_flip_vertical_flag': False,
            'random_distort_color_flag': True,
            'distort_color_fast_mode_flag': False,

            'crop_type': CropType.no_crop,
            'image_width': 224,
            'image_height': 224,
        }
        val_configs = {
            'norm_fn_first': norm_zero_to_one,
            'norm_fn_end': norm_minus_one_to_one,
            'crop_type': CropType.no_crop,
            'image_width': 224,
            'image_height': 224,
        }
        dataset = get_voc_segmentation_merged_dataset(
            train_args=train_configs,
            val_args=val_configs,
            val_set_size=300,
            batch_size=32,
            shuffle_buffer_size=10000,
            prefetch_buffer_size=10000,
            repeat=10,
            label_image_height=224, label_image_width=224,
        )
        with tf.Session() as sess:
            dataset.init(sess)

            for _ in range(10):
                images, labels = dataset.get_next_batch(sess, 0)
                print(images.shape, labels.shape)

            sess.run(dataset.tf_dataset_2_iterator.initializer)
            for _ in range(10):
                images, labels = dataset.get_next_batch(sess, 1)
                print(images.shape, labels.shape)

        self.assertTrue(True, 'Always True')
