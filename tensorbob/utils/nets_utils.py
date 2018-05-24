import tensorflow as tf
import tensorflow.contrib.slim as slim


__all__ = ['vgg_model', 'mobilenet_v2_model']


def vgg_model(vgg_type=16, **kwargs):
    # weight_decay
    # inputs,
    # num_classes=1000,
    # is_training=True,
    # dropout_keep_prob=0.5,
    # spatial_squeeze=True,
    # scope='vgg_16',
    # fc_conv_padding='VALID',
    # global_pool=False
    import nets.vgg as vgg
    vgg_type_dict = {16: vgg.vgg_16,
                     19: vgg.vgg_19}
    if vgg_type not in vgg_type_dict.keys():
        raise ValueError('vgg type must in {}'.format(vgg_type_dict.keys()))
    weight_decay = kwargs.pop('weight_decay')
    if weight_decay is None:
        weight_decay = 0.0005
    with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
        return vgg_type_dict[vgg_type](**kwargs)


def mobilenet_v2_model(training_scope_dict, model_dict):
    # training_scope_dict = {
    #     'is_training': True,
    #     'weight_decay': 0.00004,
    #     'stddev': 0.09,
    #     'dropout_keep_prob': 0.8,
    #     'bn_decay': 0.997
    # }

    # model_dict = {
    #     'input_tensor': None,
    #     'num_classes': 1001,
    #     'depth_multiplier': 1.0,
    #     'scope': 'MobilenetV2',
    #     'conv_defs': None,
    #     'finegrain_classification_mode': False,
    #     'min_depth': None,
    #     'divisible_by': None,
    #
    #     'prediction_fn': slim.softmax,
    #     'reuse': None,
    #     'base_only': False,
    #
    #     'multiplier': 1.0,
    #     'final_endpoint': None,
    #     'output_stride': None,
    #     'use_explicit_padding': False,
    #     'is_training': False,
    # }
    if training_scope_dict is None:
        training_scope_dict = {}
    if model_dict is None:
        model_dict = {}
    from nets.mobilenet import mobilenet_v2
    with slim.arg_scope(mobilenet_v2.training_scope(**training_scope_dict)):
        return mobilenet_v2.mobilenet(**model_dict)

