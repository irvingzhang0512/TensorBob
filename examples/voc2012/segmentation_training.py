import tensorbob as bob
import tensorflow as tf
import logging

logger = logging.getLogger('tensorflow')
logger.setLevel(logging.DEBUG)


class VocSegmentationTrainer(bob.training.trainer.BaseSegmentationTrainer):
    def __init__(self,
                 data_path,
                 pre_trained_model_path=None,
                 **kwargs):
        # {
        #     'training_crop_size': 224,
        #     'val_crop_size': 224,
        #
        #     'batch_size': 32,
        #     'weight_decay': 0.00005,
        #     'keep_prob': 0.5,
        #     'learning_rate_start': 0.001,
        #     'lr_decay_rate': 0.5,
        #     'lr_decay_steps': 40000 * 10,
        #     'lr_staircase': False,
        #     'step_ckpt_dir': './logs/ckpt/',
        #     'train_logs_dir': './logs/train/',
        #     'val_logs_dir': './logs/val/',
        #     'best_val_ckpt_dir': './logs/best_val/',
        #     'metrics_collection': 'val_metrics',
        #     'metrics_update_ops_collection': 'update_ops',
        #     'metrics_reset_ops_collection': 'reset_ops',
        #     'use_mean_metrics': False,
        #     'logging_every_n_steps': 1000,
        #     'summary_every_n_steps': 1000,
        #     'save_every_n_steps': 1000,
        #     'evaluate_every_n_steps': 10000,
        #     'max_steps': None
        # }
        super().__init__(num_classes=21, **kwargs)
        self._data_path = data_path
        self._pre_trained_model_path = pre_trained_model_path

    def _get_training_dataset(self):
        train_configs = {
            'norm_fn_first': bob.preprocessing.norm_imagenet,
            'image_width': 224,
            'image_height': 224,
        }
        return bob.data.get_voc_segmentation_dataset(self._data_path,
                                                     'train',
                                                     batch_size=self._batch_size,
                                                     label_image_height=train_configs['image_height'],
                                                     label_image_width=train_configs['image_height'],
                                                     **train_configs
                                                     )

    def _get_val_dataset(self):
        val_configs = {
            'norm_fn_first': bob.preprocessing.norm_imagenet,
            'image_width': 224,
            'image_height': 224,
        }
        return bob.data.get_voc_segmentation_dataset(self._data_path,
                                                     'val',
                                                     batch_size=self._batch_size,
                                                     label_image_height=val_configs['image_height'],
                                                     label_image_width=val_configs['image_height'],
                                                     **val_configs
                                                     )

    def _get_model(self):
        logits, _ = bob.segmentation.vgg16_fcn_8s(
            self._x,
            self._num_classes,
            self._ph_is_training,
            self._keep_prob,
            self._weight_decay)
        return logits, None

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self._get_learning_rate())

    def _get_scaffold(self):
        if self._pre_trained_model_path is None:
            return None

        variables_to_restore = bob.variables.get_variables_to_restore(include=['vgg16_fcn_8s/vgg_16'],
                                                                      # exclude=['vgg16_fcn_8s/vgg_16/fc8'],
                                                                      )
        var_dict = {}
        for var in variables_to_restore:
            var_name = var.name[var.name.find('/') + 1:var.name.find(':')]
            var_dict[var_name] = var
            print(var_name, var)

        logger.debug('restore %d variables' % len(var_dict))
        init_fn = bob.variables.assign_from_checkpoint_fn(self._pre_trained_model_path,
                                                          var_dict,
                                                          ignore_missing_vars=True,
                                                          reshape_variables=True)

        def new_init_fn(scaffold, session):
            init_fn(session)

        return tf.train.Scaffold(init_fn=new_init_fn)


if __name__ == '__main__':
    logs_dir_name = 'logs_new'
    t = VocSegmentationTrainer(data_path="/home/tensorflow05/data/VOCdevkit/VOC2012",
                               pre_trained_model_path='/home/tensorflow05/data/pre-trained/slim/vgg_16.ckpt',
                               logging_every_n_steps=10,
                               summary_every_n_steps=10,
                               save_every_n_steps=500,
                               learning_rate_start=0.0001,
                               step_ckpt_dir='./'+logs_dir_name+'/ckpt/',
                               train_logs_dir='./'+logs_dir_name+'/train/',
                               val_logs_dir='./'+logs_dir_name+'/val/',
                               best_val_ckpt_dir='./'+logs_dir_name+'/best_val/',
                               )
    t.train()
