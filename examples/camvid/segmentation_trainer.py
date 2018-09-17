import tensorbob as bob
import argparse
import sys
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

logging.set_verbosity(logging.DEBUG)


class CamVidTrainer(bob.training.BaseSegmentationTrainer):
    def __init__(self, epochs, train_configs, val_configs, **kwargs):
        # {
        #     'batch_size': 32,
        #     'weight_decay': 0.0005,
        #     'keep_prob': 0.8,
        #
        #     'learning_rate_type': 1,
        #     'learning_rate_start': 0.0001,
        #     'lr_decay_rate': 0.5,
        #     'lr_decay_steps': 40000,
        #     'lr_staircase': False,
        #     'steps_to_lr_dict': None,
        #     'min_lr': 0.000001,
        #     'lr_shrink_epochs': 3,
        #     'lr_shrink_by_number': 10.0,
        #
        #     'base_logs_dir': './logs'
        #     'val_logs_dir': 'val',
        #
        #     'metrics_reset_ops_collection': 'reset_ops',
        #
        #     'logging_every_n_steps': 1000,
        #     'summary_every_n_steps': 1000,
        #     'save_every_n_steps': 1000,
        #     'evaluate_every_n_steps': 10000,
        #     'max_steps': None,
        #
        #     'fine_tune_steps': None,
        #     'fine_tune_file_path': None,
        #     'fine_tune_vars_include': None,
        #     'fine_tune_vars_exclude': None,
        #
        # }
        super().__init__(num_classes=32, **kwargs)
        self._epochs = epochs
        self._train_configs = train_configs
        self._val_configs = val_configs

    def _get_merged_dataset(self):
        return bob.data.get_camvid_segmentation_merged_dataset(self._train_configs,
                                                               self._val_configs,
                                                               batch_size=self._batch_size,
                                                               repeat=self._epochs,
                                                               label_image_height=self._val_configs['image_height'],
                                                               label_image_width=self._val_configs['image_width'],
                                                               shuffle_buffer_size=100)

    def _get_optimizer(self):
        return tf.train.MomentumOptimizer(learning_rate=self._get_learning_rate(), momentum=0.9)

    def _get_model(self):
        return bob.segmentation.fcn_8s_vgg16(self._x,
                                             num_classes=self._num_classes,
                                             is_training=self._ph_is_training,
                                             keep_prob=self._keep_prob,
                                             weight_decay=self._weight_decay)

    def _get_fine_tune_var_dict(self, variables_to_restore):
        print(variables_to_restore)
        var_dict = {}
        for var in variables_to_restore:
            var_name = var.name[var.name.find('/') + 1:var.name.find(':')]
            var_dict[var_name] = var
            logging.debug(var_name, var)
        return var_dict


def main(args):
    train_configs = {
        'norm_fn_first': bob.preprocessing.norm_imagenet,
        # 'norm_fn_first': bob.preprocessing.norm_zero_to_one,
        # 'norm_fn_end': bob.preprocessing.norm_minus_one_to_one,
        'crop_type': bob.data.CropType.no_crop,
        'image_width': args.image_size,
        'image_height': args.image_size,
        # 'random_distort_color_flag': True,
    }
    val_configs = {
        'norm_fn_first': bob.preprocessing.norm_imagenet,
        # 'norm_fn_first': bob.preprocessing.norm_zero_to_one,
        # 'norm_fn_end': bob.preprocessing.norm_minus_one_to_one,
        'crop_type': bob.data.CropType.no_crop,
        'image_width': args.image_size,
        'image_height': args.image_size,
    }
    trainer_configs = {
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'keep_prob': args.keep_prob,

        'learning_rate_type': 1,
        'learning_rate_start': args.learning_rate_start,
        'lr_decay_rate': args.learning_rate_decay_rate,
        'lr_decay_steps': args.learning_rate_decay_steps,
        'lr_staircase': args.learning_rate_staircase,

        'base_logs_dir': args.base_logs_dir,
        'val_logs_dir': 'val',

        'logging_every_n_steps': args.logging_every_n_steps,
        'summary_every_n_steps': args.summary_every_n_steps,
        'save_every_n_steps': args.save_every_n_steps,
        'evaluate_every_n_steps': args.evaluate_every_n_steps,

        'fine_tune_file_path': args.fine_tune_file_path,
        'fine_tune_vars_include': args.fine_tune_vars_include,
        'fine_tune_vars_exclude': args.fine_tune_vars_exclude,
    }
    trainer = CamVidTrainer(args.epochs,
                            train_configs,
                            val_configs,
                            **trainer_configs)
    trainer.train()


def _parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # local input file
    parser.add_argument('--data_path', type=str,
                        default="E:\\PycharmProjects\\data\\CamVid")

    # training configs
    parser.add_argument('--batch_size', type=int, default=16)  # 必须是3的倍数
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--keep_prob', type=float, default=0.8)

    # learning rate
    parser.add_argument('--learning_rate_start', type=float, default=0.00001)
    parser.add_argument('--learning_rate_decay_steps', type=int, default=10000)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=1)
    parser.add_argument('--learning_rate_staircase', type=bool, default=True)

    # model
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--fine_tune_file_path', type=str,
                        default='E:\\PycharmProjects\\data\\slim\\vgg_16.ckpt')
    parser.add_argument('--fine_tune_vars_include', type=list, default=['vgg16_fcn_8s/vgg_16'])
    parser.add_argument('--fine_tune_vars_exclude', type=list, default=['vgg16_fcn_8s/vgg_16/fc8'])

    # logs
    parser.add_argument('--base_logs_dir', type=str, default="./logs", help='')

    # steps
    parser.add_argument('--logging_every_n_steps', type=int, default=10)
    parser.add_argument('--summary_every_n_steps', type=int, default=10)
    parser.add_argument('--save_every_n_steps', type=int, default=100)
    parser.add_argument('--evaluate_every_n_steps', type=int, default=100)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(_parse_arguments(sys.argv[1:]))
