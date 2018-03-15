import tensorflow as tf


def norm_inception_v3(x):
    """
    将图片数值归一化到[-1, 1]之间
    :param x: 
    :return: 
    """
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def norm_imagenet(x):
    """
    减去imagenet的平均数
    :param x: 
    :return: 
    """
    x = x - [103.939, 116.779, 123.68]
    # x[..., 2] -= 103.939
    # x[..., 1] -= 116.779
    # x[..., 0] -= 123.68
    return x


def crop(images, offset_height, offset_width, target_height, target_width):
    """
    切片
    :param images: 
    :param offset_height: 
    :param offset_width: 
    :param target_height: 
    :param target_width: 
    :return: 
    """
    return tf.image.crop_to_bounding_box(images,
                                         tf.to_int32(offset_height), tf.to_int32(offset_width),
                                         target_height, target_width)


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
    return crop(images, offset_height, offset_width, crop_height, crop_width)


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


def resize_smallest_size(images, smallest_height, smallest_width, keep_aspect_ratio=True):
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


def random_flip_vertical(image):
    return tf.image.random_flip_up_down(image)


def random_flip_horizontal(image):
    return tf.image.random_flip_left_right(image)


def random_brightness(image, max_delta):
    return tf.image.random_brightness(image, max_delta)


def random_contrast(image, lower, upper):
    return tf.image.random_contrast(image, lower, upper)


def random_hue(image, max_delta):
    return tf.image.random_hue(image, max_delta)


def random_saturation(image, lower, upper):
    return tf.image.random_saturation(image, lower, upper)
