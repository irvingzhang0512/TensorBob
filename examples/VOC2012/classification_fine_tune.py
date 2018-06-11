import tensorbob as bob
import tensorflow as tf
from nets import nets_factory

tf.logging.set_verbosity(tf.logging.DEBUG)


class VocClassificationFineTuneTrainer(bob.trainer.BaseClassificationTrainer):
    def __init__(self, data_path, pre_trained_model_path=None, **kwargs):
        super().__init__(num_classes=20, **kwargs)
        self._data_path = data_path
        self._pre_trained_model_path = pre_trained_model_path

    def _get_training_dataset(self):
        train_configs = {
            # 'norm_fn_first': bob.data.norm_zero_to_one,
            # 'norm_fn_end': bob.data.norm_minus_one_to_one,
            # 'random_flip_horizontal_flag': True,
            # 'random_distort_color_flag': True,
            # 'crop_type': bob.data.CropType.random_inception,
            # 'crop_width': self._training_crop_size,
            # 'crop_height': self._training_crop_size,
            # 'inception_bbox': None,

            'norm_fn_first': bob.data.norm_imagenet,
            'random_flip_horizontal_flag': True,
            'crop_type': bob.data.CropType.random_vgg,
            'vgg_image_size_min': 256,
            'vgg_image_size_max': 512,
            'crop_width': self._training_crop_size,
            'crop_height': self._training_crop_size,
        }
        return bob.data.get_voc_classification_dataset(data=self._data_path,
                                                       batch_size=self._batch_size,
                                                       **train_configs)

    def _get_val_dataset(self):
        val_configs = {
            # 'norm_fn_first': bob.data.norm_zero_to_one,
            # 'norm_fn_end': bob.data.norm_minus_one_to_one,
            'norm_fn_first': bob.data.norm_imagenet,
            'image_width': self._val_crop_size,
            'image_height': self._val_crop_size,
        }
        return bob.data.get_voc_classification_dataset(mode='val',
                                                       data=self._data_path,
                                                       batch_size=self._batch_size,
                                                       **val_configs)

    def _get_model(self):
        # network_fn = nets_factory.get_network_fn('inception_v3',
        #                                          num_classes=self._num_classes,
        #                                          weight_decay=self._weight_decay,
        #                                          is_training=self._ph_is_training,
        #                                          )
        # return network_fn(images=tf.reshape(self._ph_x, [-1, self._ph_image_size, self._ph_image_size, 3]),
        #                   dropout_keep_prob=self._keep_prob,
        #                   global_pool=False,
        #                   create_aux_logits=False)
        network_fn = nets_factory.get_network_fn('vgg_16',
                                                 num_classes=self._num_classes,
                                                 weight_decay=self._weight_decay,
                                                 is_training=self._ph_is_training,
                                                 )
        return network_fn(images=tf.reshape(self._ph_x, [-1, self._ph_image_size, self._ph_image_size, 3]),
                          dropout_keep_prob=self._keep_prob,
                          global_pool=True,)

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(self._get_learning_rate())

    def _get_scaffold(self):
        if self._pre_trained_model_path is None:
            return None

        # variables_to_restore = bob.variables.get_variables_to_restore(include=['InceptionV3'],
        #                                                               exclude=['InceptionV3/Logits'])
        variables_to_restore = bob.variables.get_variables_to_restore(include=['vgg_16'],
                                                                      exclude=['vgg_16/fc8'])
        tf.logging.debug('restore %d variables' % len(variables_to_restore))
        init_fn = bob.variables.assign_from_checkpoint_fn(self._pre_trained_model_path,
                                                          variables_to_restore,
                                                          ignore_missing_vars=True,
                                                          reshape_variables=True)

        def new_init_fn(scaffold, session):
            init_fn(session)

        return tf.train.Scaffold(init_fn=new_init_fn)


if __name__ == '__main__':
    t = VocClassificationFineTuneTrainer('/home/tensorflow05/data/VOC2012',
                                         pre_trained_model_path='/home/tensorflow05/data/pre-trained/slim'
                                                                '/vgg_16.ckpt',

                                         training_crop_size=224,
                                         val_crop_size=384,

                                         logging_every_n_steps=50,
                                         summary_every_n_steps=50,
                                         save_every_n_steps=500,
                                         evaluate_every_n_steps=200,

                                         learning_rate_type=3,
                                         learning_rate_start=0.001,
                                         lr_shrink_epochs=3,
                                         lr_shrink_by_number=10.0,
                                         )
    t.train()

