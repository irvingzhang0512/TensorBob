from .base_dataset import *

if __name__ == '__main__':
    IMAGES = ["D:\\1.jpg", "D:\\2.jpg"]
    LABELS = [1, 2]
    dataset_configs = [{
        'type': 0,
        'src': IMAGES,
        'image_width': 500,
        'image_height': 500,
        'norm_fn': None,
        'crop_width': 300,
        'crop_height': 200,
        'central_crop_flag': True,
        'random_flip_horizontal_flag': True,
        'random_flip_vertical_flag': True,
    }, {
        'type': 1,
        'src': LABELS
    }]

    base_dataset = BaseDataset(dataset_configs, 32, repeat=True)

    with tf.Session() as sess:
        images, labels = base_dataset.get_next_batch(sess)
        images, labels = base_dataset.get_next_batch(sess)
        print(images.shape, labels)