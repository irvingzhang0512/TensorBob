import tensorbob as bob
import tensorflow as tf
from nets import nets_factory

tf.logging.set_verbosity(tf.logging.DEBUG)


DATA_PATH = '/home/ubuntu/data/voc2012/train/voc2012'

# PRE_TRAINED_MODEL_PATH = '/home/ubuntu/data/slim/vgg_16.ckpt'
# FINE_TUNE_VAR_INCLUDE = ['vgg_16/fc8']
# VARS_INCLUDE = ['vgg_16']
# VARS_EXCLUDE = ['vgg_16/fc8']
# NORM_FN_FIRST = bob.data.norm_imagenet
# NORM_FN_END = None
# MODEL_NAME = 'vgg_16'
# NET_KWARGS = {'dropout_keep_prob': 0.8,
#               'global_pool': True, }

PRE_TRAINED_MODEL_PATH = '/home/ubuntu/data/slim/inception_v3.ckpt'
FINE_TUNE_VAR_INCLUDE = ['InceptionV3/Logits']
VARS_INCLUDE = ['InceptionV3']
VARS_EXCLUDE = ['InceptionV3/Logits']
NORM_FN_FIRST = bob.preprocessing.norm_zero_to_one
NORM_FN_END = bob.preprocessing.norm_minus_one_to_one
MODEL_NAME = 'inception_v3'
NET_KWARGS = {'create_aux_logits': False,
              'global_pool': True,
              'dropout_keep_prob': 0.8, }


class VocClassificationFineTuneTrainer(bob.training.BaseClassificationTrainer):
    def __init__(self, data_path, pre_trained_model_path=None, **kwargs):
        super().__init__(num_classes=20, **kwargs)
        self._data_path = data_path
        self._pre_trained_model_path = pre_trained_model_path

    def _get_training_dataset(self):
        train_configs = {
            'norm_fn_first': NORM_FN_FIRST,
            'norm_fn_end': NORM_FN_END,
            'random_flip_horizontal_flag': True,
            'random_distort_color_flag': True,
            'crop_width': self._training_crop_size,
            'crop_height': self._training_crop_size,

            'crop_type': bob.data.CropType.random_inception,
            'inception_bbox': None,

            # 'crop_type': bob.data.CropType.random_vgg,
            # 'vgg_image_size_min': 256,
            # 'vgg_image_size_max': 512,
        }
        return bob.data.get_voc_classification_dataset(data_path=self._data_path,
                                                       batch_size=self._batch_size,
                                                       **train_configs)

    def _get_val_dataset(self):
        val_configs = {
            'norm_fn_first': NORM_FN_FIRST,
            'norm_fn_end': NORM_FN_END,
            'image_width': self._val_crop_size,
            'image_height': self._val_crop_size,
        }
        return bob.data.get_voc_classification_dataset(mode='val',
                                                       data_path=self._data_path,
                                                       batch_size=self._batch_size,
                                                       **val_configs)

    def _get_model(self):
        network_fn = nets_factory.get_network_fn(MODEL_NAME,
                                                 num_classes=self._num_classes,
                                                 weight_decay=self._weight_decay,
                                                 is_training=self._ph_is_training,
                                                 )
        return network_fn(images=self._x, **NET_KWARGS)

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(self._get_learning_rate())

    def _get_scaffold(self):
        if self._pre_trained_model_path is None:
            return None

        variables_to_restore = bob.variables.get_variables_to_restore(include=VARS_INCLUDE,
                                                                      exclude=VARS_EXCLUDE)
        tf.logging.debug('restore %d variables' % len(variables_to_restore))
        init_fn = bob.variables.assign_from_checkpoint_fn(self._pre_trained_model_path,
                                                          variables_to_restore,
                                                          ignore_missing_vars=True,
                                                          reshape_variables=True)

        def new_init_fn(scaffold, session):
            init_fn(session)

        return tf.train.Scaffold(init_fn=new_init_fn)


if __name__ == '__main__':
    t = VocClassificationFineTuneTrainer(DATA_PATH,
                                         pre_trained_model_path=PRE_TRAINED_MODEL_PATH,

                                         fine_tune_steps=1000,
                                         fine_tune_var_include=FINE_TUNE_VAR_INCLUDE,
                                         training_crop_size=299,
                                         val_crop_size=299,

                                         logging_every_n_steps=50,
                                         summary_every_n_steps=50,
                                         save_every_n_steps=500,
                                         evaluate_every_n_steps=200,

                                         learning_rate_type=3,
                                         learning_rate_start=0.005,
                                         lr_shrink_epochs=5,
                                         lr_shrink_by_number=10.0,
                                         )
    t.train()
