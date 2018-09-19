# TensorBob
TensorFlow tools

## 0. dependency
+ `slim`: [tensorflow/models/research/slim][1].
    + add `/path/to/models/research/slim` in `PYTHONPATH`, so that we can use `from nets import vgg` to build models.

## 1. tools

### 1.1. dataset
+ target:
    + create `tf.data.Dataset` objects with Data Argument.
    + create `tf.data.Dataset` object for Open Data Source, such as ImageNet, VOC, etc.
    + config by python dicts.
+ Module Architecture:
    + `base_dataset.py`: `tf.data.Dataset` wrapper, including `BaseDataset` and `MergedDataset`.
    + `dataset_utils`: utils to create `tf.data.Dataset` objects configured by python dicts.
    + `segmentation_dataset_utils.py`: create `BaseDataset` and `MergedDataset` object for segmentation task.
    + Open Database:
        + `imagenet.py`: ImageNet(classification).
        + `voc2012.py`: VOC2012(classification & segmentation).
        + `ade2016.py`: Scene Parsing Challenge 2016(segmentation).
        + `camvid.py`: Cambridge-driving Labeled Video Database(segmentation).
+ For more information about dataset, please check [here](dataset/README.md).

### 1.2. training
+ target:
    + train models by `SingularMonitoredSession` & hooks.
    + build basic training procedure.
+ Module Architecture:
    + `training_utils.py`: produce hooks, creating train_op function and training function.
    + `trainer_utils.py`: learning rate utils, slim model utils and scaffold utils.
    + `trainer.py`:
        + produce basic training procedure by `Trainer`.
        + trainer for classification task `BaseClassificationTrainer`.
        + trainer for segmentation task `BaseSegmentationTrainer`.

### 1.3. evaluating
+ evaluator: evaluate given metrics with existing models on val set.

### 1.4. Models
+ creating reusable models for different tasks.
+ semantic segmentation:
    + fcn(`fcn_8s_vgg16`, `fcn_8s_resnet_v2_50`)
    + segnet(`segnet_vgg16`)
    + fc_densenet(`fc_densenet`)

### 1.5. Utils
+ `initializers.py`
+ `regularizers.py`
+ `metrics_utils.py`: utils to compute iou by confusion matrix.
+ `variables.py`: very important and useful.
+ `preprocessing.py`: utils for image preprocess.

## 2. Examples
+ target: use tensorbob tools to train/test models.
+ modules:
    + imagenet：classification models training/testing.
    + voc2012: classification & image segmentation training/testing.
    + kaggle:
        + whale: more information about this solution, please click [here][2].
    + ade: segmentation training and predicting.
    + camvid: segmentation training and evaluating.


  [1]: https://github.com/tensorflow/models/tree/master/research/slim
  [2]: https://zhuanlan.zhihu.com/p/39440686