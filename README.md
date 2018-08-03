# TensorBob
TensorFlow tools

## 0. dependency
+ `slim`: [tensorflow/models/research/slim][1].
    + add `/path/to/models/research/slim` in `PYTHONPATH`, so that we can use `from nets import vgg` to use build models.

## 1. tools

### 1.1. dataset
+ target:
    + create `tf.data.Dataset` objects with Data Argument.
    + create `tf.data.Dataset` object for Open Data Source, such as ImageNet, VOC, etc.
    + config by python dicts.
+ Module Architecture:
    + `base_dataset.py`: `tf.data.Dataset` wrapper.
    + `dataset_utils`: utils to create `tf.data.Dataset` objects configurated by python dicts.
    + `preprocessing.py`: utils for image preprocessing.
    + `imagenet.py`: create `BaseDataset` objects for ImageNet(classification).
    + `voc2012.py`: create `BaseDataset` objects for VOC2012(classification & sementic segmentation).

### 1.2. Training
+ training utils:
    + all kinds of hooks.
    + learning rate.
    + finetune train op.
+ train function using `SingularMonitoredSession` & hooks.
+ trainer: use all the tools to tackle specific tasks(classification, semantic segmentation).

### 1.3. Evaluating & Predicting
+ evaluator: evaluate given metrics with existing models on val set.
+ predictor: predict with exsiting models on test set.
+ PS: use model fusion.

### 1.4. Models
+ creating reusable models for different tasks.
+ classification models:
    + xception(todo).
    + densenet(todo).
+ sementic segmentation:
    + fcn(finished).

### 1.5. Utils
+ regularizers.
+ losses.
+ variables.
+ metrics.



## 2. Examples
+ target: use tensorbob tools to train/test models.
+ modules:
    + ImageNet：classification models training/testing.
    + VOC2012: classification & image segmentation training/testing.
    + kaggle:
        + whale: more information about this solution, please click [here][2].


  [1]: https://github.com/tensorflow/models/tree/master/research/slim
  [2]: https://zhuanlan.zhihu.com/p/39440686