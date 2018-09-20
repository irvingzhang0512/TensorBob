# 基本工具

## 1. metrics_utils.py
+ use to compute iou. code modified by `tf.metrics.mean_iou`。
```python
__all__ = ['compute_mean_iou',
           'compute_mean_iou_by_confusion_matrix',
           ]
```

## 2. preprocessing.py
+ 数据归一化（一般用于原始数据处理的第一步）
+ 图像增强方法，包括镜像、切片、色彩变换。
```python
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
```

## 3. regularizers.py
+ regularizer functions.
```python
__all__ = ['l1_regularizer',
           'l2_regularizer',
           'l1_l2_regularizer',
           'sum_regularizer',
           'apply_regularization']
```


## 4. variables.py
+ create new var.
+ get vars by condition.
+ add var to collection list.
+ set value to vars.
```python
__all__ = ['add_model_variable',
           'assert_global_step',
           'assert_or_get_global_step',
           'assign_from_checkpoint',
           'assign_from_checkpoint_fn',
           'assign_from_values',
           'assign_from_values_fn',
           'create_global_step',
           'filter_variables',
           'get_global_step',
           'get_or_create_global_step',
           'get_local_variables',
           'get_model_variables',
           'get_trainable_variables',
           'get_unique_variable',
           'get_variables_by_name',
           'get_variables_by_suffix',
           'get_variable_full_name',
           'get_variables_to_restore',
           'get_variables',
           'global_variable',
           'local_variable',
           'model_variable',
           'variable',
           'VariableDeviceChooser',
           'zero_initializer']
```
