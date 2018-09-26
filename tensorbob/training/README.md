# training utils
+ 详细信息请参考对应方法的注释文档。
+ 下面仅仅罗列一下提供的功能。

## 1. training_utils.py
+ all kinds of hooks.
+ train_op creating function.
+ train function.
    + the most significant function of training.
    + use `SingularMonitoredSession` and hooks to train.
```python
__all__ = ['SecondOrStepTimer',
    # hooks
    'LoggingTensorHook',
    'StopAtStepHook',
    'CheckpointSaverHook',
    'StepCounterHook',
    'NanLossDuringTrainingError',
    'NanTensorHook',
    'SummarySaverHook',
    'SummarySaverHookV2',
    'InitFnHook',
    'GlobalStepWaiterHook',
    'ProfilerHook',
    'FinalOpsHook',
    'FeedFnHook',
    'ValidationDatasetEvaluationHook',
    
    # train_op creating function
    'create_train_op',
    'create_train_op_v2',
    'create_finetune_train_op',
    
    # train function
    'train',
]
```

## 2. trainer_utils.py
+ 学习率相关
+ 通过 slim models 获取图像分类模型
+ 获取 scaffold，导入 fine-tune 模型
```python
__all__ = [
    # 学习率相关
    'learning_rate_exponential_decay',
    'learning_rate_steps_dict',
    'learning_rate_val_evaluation',
    
    # 获取 scaffold，导入 fine-tune 模型
    'scaffold_pre_trained_model',
    
    # 通过 slim models 获取图像分类模型
    'model_from_slim_nets_factory',
]
```

## 3. trainer.py
+ 基本训练器
+ 分类任务训练器
+ 图像分割任务训练器
```python
__all__ = [
    'BaseTrainer', 
    'BaseClassificationTrainer', 
    'BaseSegmentationTrainer'
]
```
