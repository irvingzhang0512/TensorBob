import tensorflow as tf
import tensorbob as bob
from nets import nets_factory
import logging

logger = logging.getLogger('tensorflow')
logger.setLevel(logging.DEBUG)


class ImageNetMobileNetTrainer(bob.basic_trainers.BaseClassificationTrainer):
    def __init__(self, **kwargs):
        super().__init__(val_crop_size=224, **kwargs)

    def _get_training_dataset(self):
        train_configs = {
            'norm_fn_first': bob.data.norm_zero_to_one,
            'norm_fn_end': bob.data.norm_minus_one_to_one,
            'random_flip_horizontal_flag': True,
            'random_distort_color_flag': True,
            'distort_color_fast_mode_flag': False,

            # inception随机切片
            'crop_type': bob.data.CropType.random_inception,
            'crop_width': self._training_crop_size,
            'crop_height': self._training_crop_size,
            'inception_bbox': None,

        }
        return bob.data.get_imagenet_classification_dataset('train',
                                                            self._batch_size,
                                                            '/home/ubuntu/data/ILSVRC2012',
                                                            **train_configs)

    def _get_val_dataset(self):
        val_configs = {
            'norm_fn_first': bob.data.norm_zero_to_one,
            'norm_fn_end': bob.data.norm_minus_one_to_one,
            'crop_type': bob.data.CropType.no_crop,
            'image_width': self._val_crop_size,
            'image_height': self._val_crop_size,
        }
        return bob.data.get_imagenet_classification_dataset('val',
                                                            self._batch_size,
                                                            '/home/ubuntu/data/ILSVRC2012',
                                                            **val_configs)

    def _get_model(self):
        network_fn = nets_factory.get_network_fn('mobilenet_v2',
                                                 num_classes=self._num_classes,
                                                 weight_decay=self._weight_decay,
                                                 is_training=self._ph_is_training,
                                                 )
        return network_fn(images=tf.reshape(self._ph_x, [-1, self._ph_image_size, self._ph_image_size, 3]))

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self._get_learning_rate())

    def _get_scaffold(self):
        return None


if __name__ == '__main__':
    whale = ImageNetMobileNetTrainer(
        training_crop_size=224,
        best_val_ckpt_dir='./logs/best_val/model.ckpt',
        logging_every_n_steps=500,
        summary_every_n_steps=500,
        save_every_n_steps=5000,
        evaluate_every_n_steps=20000,
        lr_decay_steps=20000*5,
        lr_staircase=False,
        learning_rate_start=0.0005
    )
    whale.train()
