import tensorflow as tf
import random

from .preprocessing import *


def _get_images_by_path_dataset(dataset_config, batch_size):
    """
    :param dataset_config: 参考本文件中 get_image_by_path_dataset_config 方法
    :param batch_size:
    :return: dataset
    """
    file_paths = dataset_config.get('src')  # 原始数据
    image_width = dataset_config.get('image_width')  # 图片宽度
    image_height = dataset_config.get('image_height')  # 图片高度
    norm_fn = dataset_config.get('norm_fn')  # 归一化函数
    crop_width = dataset_config.get('crop_width')  # 切片宽
    crop_height = dataset_config.get('crop_height')  # 切片高
    central_crop_flag = dataset_config.get('central_crop_flag')  # 是否使用中心切片（默认随机切片）
    random_flip_horizontal_flag = dataset_config.get('random_flip_horizontal_flag')  # 随机水平镜像
    random_flip_vertical_flag = dataset_config.get('random_flip_vertical_flag')  # 随机垂直镜像

    # 实现vgg中的 multi-scale 数据增强
    # 输入的multi_scale_list，必须长度为2或4，代表图片长宽的最小、大尺寸， 如(256, 384, 256, 384)
    multi_scale_training_list = dataset_config['multi_scale_training_list']
    if multi_scale_training_list:
        if len(multi_scale_training_list) == 2:
            image_height = random.randint(multi_scale_training_list[0], multi_scale_training_list[1])
            image_width = image_height
        elif len(multi_scale_training_list) == 4:
            image_height = random.randint(multi_scale_training_list[0], multi_scale_training_list[1])
            image_width = random.randint(multi_scale_training_list[2], multi_scale_training_list[3])
        else:
            raise ValueError(
                'multi_scale_training_list has 2 or 4 elements but get %d' % (len(multi_scale_training_list)))

    def _cur_parse_image_fn(image_path):
        img_file = tf.read_file(image_path)
        cur_image = tf.image.decode_jpeg(img_file)
        if image_width is not None and image_height is not None:
            cur_image = tf.image.resize_images(cur_image, [image_height, image_width])
        if random_flip_horizontal_flag:
            cur_image = random_flip_horizontal(cur_image)
        if random_flip_vertical_flag:
            cur_image = random_flip_vertical(cur_image)
        return cur_image

    def _cur_preprocess_fn(images):
        if norm_fn:
            images = norm_fn(images)

        # 如果crop_width 和 crop_height 有一个为None，则不进行切片
        if crop_width is not None and crop_height is not None:
            if central_crop_flag:
                # 通过配置，可以使用中心切片
                images = central_crop(images, crop_height, crop_width)
            else:
                # 默认使用随机切片
                images = random_crop(images, crop_height, crop_width)
        return images

    cur_dataset = tf.data.Dataset.from_tensor_slices(file_paths) \
        .map(_cur_parse_image_fn) \
        .batch(batch_size=batch_size) \
        .map(_cur_preprocess_fn)
    return cur_dataset


def _get_labels_dataset(dataset_config, batch_size):
    """
    {
        'type': 1,
        'src': []
    }
    :param dataset_config:
    :param batch_size:
    :return:
    """
    src = dataset_config['src']
    return tf.data.Dataset.from_tensor_slices(tf.constant(src)).batch(batch_size)


def get_dataset_by_config(dataset_config, batch_size):
    """
    通过config获取dataset以及dataset_size
    :param dataset_config:
    :param batch_size:
    :return: dataset 和 dataset_size
    """

    dataset_type = dataset_config['type']

    if dataset_type == 0:
        # 通过 file_path 获取 image
        return _get_images_by_path_dataset(dataset_config, batch_size), len(dataset_config['src'])
    elif dataset_type == 1:
        return _get_labels_dataset(dataset_config, batch_size), len(dataset_config['src'])
    else:
        raise ValueError('unknown dataset type {}'.format(dataset_type))


def get_image_by_path_dataset_config(file_paths, **kwargs):
    """
    通过file_paths获取图片
    :param file_paths: 图片path
    :param kwargs: 举例如下
    {
        'image_width': 500,
        'image_height': 500,
        'norm_fn': None,
        'crop_width': 300,
        'crop_height': 200,
        'central_crop_flag': True,
        'random_flip_horizontal_flag': True,
        'random_flip_vertical_flag': True,
        'multi_scale_training_list': [256, 384, 256, 384],
    }
    1. image_width/image_height 和 multi_scale_training_list的作用相同，相当于对图片进行resize。
        上述两个参数必须存在其中一个，同时存在则 multi_scale_training_list 起作用。
    2. crop_width/crop_height 有一个为None时，不切片，返回整张图。
    :return:
    """
    dataset_config = {
        'type': 0,
        'src': file_paths,
        'image_width': kwargs.get('image_width') or 224,
        'image_height': kwargs.get('image_height') or 224,
        'norm_fn': kwargs.get('norm_fn'),
        'crop_width': kwargs.get('crop_width'),
        'crop_height': kwargs.get('crop_height'),
        'central_crop_flag': kwargs.get('central_crop_flag') or False,
        'random_flip_horizontal_flag': kwargs.get('random_flip_horizontal_flag') or False,
        'random_flip_vertical_flag': kwargs.get('random_flip_vertical_flag') or False,
        'multi_scale_training_list': kwargs.get('multi_scale_training_list'),
    }
    return dataset_config


def get_labels_dataset_config(labels):
    """
    直接返回 labels
    :param labels: 
    :return: 
    """
    dataset_config = {
        'type': 1,
        'src': labels
    }
    return dataset_config
