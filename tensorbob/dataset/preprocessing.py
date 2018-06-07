import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


__all__ = ['norm_imagenet',
           'norm_zero_to_one',
           'norm_minus_one_to_one',
           'resize_smallest_size',
           'central_crop',
           'random_crop',
           'random_crop_vgg',
           'random_crop_inception',
           'distort_color',
           'random_distort_color',
           ]


def norm_imagenet(image):
    """
    减去imagenet的平均数
    :param image:
    :return: 
    """
    image = tf.cast(image, tf.float32)
    means = [103.939, 116.779, 123.68]
    channels = tf.split(axis=2, num_or_size_splits=3, value=image)
    for i in range(3):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)


def norm_zero_to_one(image):
    """
    将原始图片像素数据数据从tf.uint8转换为tf.float32，数据范围是[0, 1]
    PS：如果输入tensor原本就是tf.float32，则默认处于[0, 1]之间
    :param image:
    :return:
    """
    return tf.image.convert_image_dtype(image, tf.float32)


def norm_minus_one_to_one(image):
    """
    将原始图片像素数据数据从tf.uint8转换为tf.float32，数据范围是[-1, 1]
    PS：如果输入tensor原本就是tf.float32，则默认进行剪裁到[0, 1]之间
    :param image:
    :return:
    """
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.clip_by_value(image, .0, 1.0)
    image = tf.multiply(image, 2.0)
    image = tf.add(image, -1.0)
    return image


def central_crop(images, crop_height, crop_width, keep_aspect_ratio=True):
    """
    中心点切片
    切片前通过 resize_smallest_size 确认短边大小
    如果输入图片有多张，那要求所有图片的shape相同
    :param images: 
    :param crop_height: 
    :param crop_width: 
    :param keep_aspect_ratio: 
    :return: 
    """
    images = resize_smallest_size(images, crop_height, crop_width, keep_aspect_ratio)
    image_height = tf.shape(images)[-3]
    image_width = tf.shape(images)[-2]
    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2
    return tf.image.crop_to_bounding_box(images,
                                         tf.to_int32(offset_height), tf.to_int32(offset_width),
                                         crop_height, crop_width)


def random_crop(images, crop_height, crop_width, keep_aspect_ratio=True):
    """
    随机切片
    切片前通过 resize_smallest_size 确认短边大小
    如果输入图片有多张，则要求所有图片shape相同
    :param keep_aspect_ratio: 
    :param images: 
    :param crop_height: 
    :param crop_width: 
    :return: 
    """
    images = resize_smallest_size(images, crop_height, crop_width, keep_aspect_ratio)
    image_height = tf.shape(images)[-3]
    image_width = tf.shape(images)[-2]
    offset_height = tf.random_uniform([], 0, (image_height - crop_height + 1), dtype=tf.int32)
    offset_width = tf.random_uniform([], 0, (image_width - crop_width + 1), dtype=tf.int32)
    return tf.image.crop_to_bounding_box(images, offset_height, offset_width, crop_height, crop_width)


def random_crop_vgg(image,
                    resize_min, resize_max,
                    crop_height, crop_width,
                    scope=None):
    """
    使用vgg的随机切片
    先讲图片resize，再进行切片
    :param image:
    :param resize_min:
    :param resize_max:
    :param crop_height:
    :param crop_width:
    :param scope:
    :return:
    """
    with tf.name_scope(scope, 'random_crop_vgg', [image, resize_min, resize_max, crop_height, crop_width]):
        resize_length = tf.random_uniform(
            [], minval=resize_min, maxval=resize_max + 1, dtype=tf.int32)
        image = resize_smallest_size(image, resize_length, resize_length, True)
        original_shape = tf.shape(image)
        size_assertion = tf.Assert(
            tf.logical_and(
                tf.greater_equal(original_shape[0], crop_height),
                tf.greater_equal(original_shape[1], crop_width)),
            ['Crop size greater than the image size.'])
        with tf.control_dependencies([size_assertion]):
            image_height = tf.shape(image)[-3]
            image_width = tf.shape(image)[-2]
            offset_height = tf.random_uniform([], 0, (image_height - crop_height + 1), dtype=tf.int32)
            offset_width = tf.random_uniform([], 0, (image_width - crop_width + 1), dtype=tf.int32)
            return tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_height, crop_width)


def random_crop_inception(image, crop_height, crop_width,
                          bbox=None,
                          min_object_covered=0.1,
                          aspect_ratio_range=(0.75, 1.33),
                          area_range=(0.05, 1.0),
                          max_attempts=100,
                          scope=None):
    """
    使用inception的随机切片
    先通过条件获取bbox，再将bbox中的图片resize到固定尺寸
    :param image:
    :param bbox:
    :param crop_height:
    :param crop_width:
    :param min_object_covered:
    :param aspect_ratio_range:
    :param area_range:
    :param max_attempts:
    :param scope:
    :return:
    """
    with tf.name_scope(scope, 'random_crop_inception', [image, bbox]):
        if bbox is None:
            bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                               dtype=tf.float32,
                               shape=[1, 1, 4])
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
        image = tf.slice(image, bbox_begin, bbox_size)
        image.set_shape([None, None, 3])
        return tf.image.resize_images(image, [crop_height, crop_width])


def resize_smallest_size(images,
                         smallest_height, smallest_width,
                         keep_aspect_ratio=True):
    """
    要求图片的height width不能小于指定值
    如果输入多张图片，则默认图片shape相同
    当小于指定值，需要resize时，通过 keep_aspect_ratio 确定是否进行等比例resize
    :param images: 
    :param smallest_height: 
    :param smallest_width: 
    :param keep_aspect_ratio: 
    :return: 
    """
    image_height = tf.to_float(tf.shape(images)[-3])
    image_width = tf.to_float(tf.shape(images)[-2])
    smallest_height = tf.to_float(smallest_height)
    smallest_width = tf.to_float(smallest_width)
    height_scale = smallest_height / image_height
    width_scale = smallest_width / image_width
    if keep_aspect_ratio:
        scale = tf.cond(height_scale > width_scale, lambda: height_scale, lambda: width_scale)
        scale = tf.cond(scale < 1, lambda: 1.0, lambda: scale)
        new_height = tf.to_int32(image_height * scale)
        new_width = tf.to_int32(image_width * scale)
    else:
        new_width = tf.cond(width_scale < 1, lambda: image_width, lambda: image_width * width_scale)
        new_height = tf.cond(height_scale < 1, lambda: image_height, lambda: image_height * height_scale)
    return tf.image.resize_images(images, [new_height, new_width])


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """
    这段代码完全复制了slim中inception_preprocessing的内容
    https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py
    :param image:
    :param color_ordering:
    :param fast_mode:
    :param scope:
    :return:
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')

        # 变换后，可能导致像素值脱离[0, 1]的范围，所以要进行数据切割
        return tf.clip_by_value(image, 0.0, 1.0)


def random_distort_color(image, fast_mode=True, scope=None):
    """
    随机选择 distort_color 四种变换中的一种，进行图片变换
    :param image:
    :param fast_mode:
    :param scope:
    :return:
    """
    with tf.name_scope('random_distort_color', scope, [image, fast_mode]):
        color_ordering_id = tf.random_uniform([], maxval=4, dtype=tf.int32)
        return control_flow_ops.merge(
            [distort_color(
                control_flow_ops.switch(image, tf.equal(color_ordering_id, case))[1],
                case,
                fast_mode
            ) for case in range(4)]
        )[0]
