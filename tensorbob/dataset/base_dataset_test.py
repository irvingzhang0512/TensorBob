from tensorbob.dataset.base_dataset import *
from tensorbob.dataset.dataset_utils import *
from tensorbob.utils.preprocessing import *
from tensorflow.python.framework.errors_impl import OutOfRangeError
import tensorflow as tf
import unittest


class BaseDatasetTest(unittest.TestCase):
    @unittest.skip
    def test_merged_dataset(self):
        d1 = tf.data.Dataset.range(100)
        d2 = tf.data.Dataset.range(5)
        d = MergedDataset(d1, d2)
        with tf.Session() as session:
            d.init(session)
            for i in range(10):
                self.assertEqual(d.get_next_batch(session, 0), i)
            session.run(d.tf_dataset_2_iterator.initializer)
            for i in range(5):
                self.assertEqual(d.get_next_batch(session, 1), i)
            for i in range(10):
                self.assertEqual(d.get_next_batch(session, 0), i + 10)
            session.run(d.tf_dataset_2_iterator.initializer)
            for i in range(5):
                self.assertEqual(d.get_next_batch(session, 1), i)

    @unittest.skip
    def test_base_dataset(self):
        file_paths = ['../examples/images/1.jpg', '../examples/images/2.jpg']
        labels = [0, 1]
        args = {
            'norm_fn_first': norm_zero_to_one,
            'norm_fn_end': norm_minus_one_to_one,
            'random_flip_horizontal_flag': True,
            'random_flip_vertical_flag': True,
            'random_distort_color_flag': True,
            'distort_color_fast_mode_flag': False,

            # 无切片(默认配置)
            'crop_type': CropType.no_crop,
            'image_width': 256,
            'image_height': 256,

            # #             中心切片
            # 'crop_type': CropType.central_crop,
            # 'crop_width': 300,
            # 'crop_height': 200,
            # 'image_width': 500,
            # 'image_height': 500,
            #
            # # 普通随机切片
            # 'crop_type': CropType.random_normal,
            # 'crop_width': None,
            # 'crop_height': None,
            # 'image_width': None,
            # 'image_height': None,
            #
            # # vgg随机切片
            # 'crop_type': CropType.random_vgg,
            # 'crop_width': None,
            # 'crop_height': None,
            # 'vgg_image_size_min': None,
            # 'vgg_image_size_max': None,
            #
            # # inception随机切片
            # 'crop_type': CropType.random_inception,
            # 'crop_width': None,
            # 'crop_height': None,
            # 'inception_bbox': None,
        }
        dataset_configs = [get_images_dataset_by_paths_config(file_paths, **args),
                           get_classification_labels_dataset_config(labels)]
        dataset = BaseDataset(dataset_configs,
                              batch_size=32,
                              shuffle=True,
                              shuffle_buffer_size=None,
                              repeat=10,
                              prefetch_buffer_size=None)
        with tf.Session() as sess:
            sess.run(dataset.iterator.initializer)
            try:
                while True:
                    images, labels = sess.run(dataset.next_batch)
                    print(images.shape)
                    print(labels)
            except OutOfRangeError:
                pass
        self.assertTrue(True, 'always true')

