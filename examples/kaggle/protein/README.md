# 简单记录

## 2018-11-20
+ 下载比赛文件。
+ 想把原始数据转换为 tfrecords，完成脚本`create_tf_record.py`。
    + 发现生成的文件体积太大……（1000张图片1G，一共有30000张图片，而原始文件只有15G左右，相当于体积增加了一倍……，所以放弃了）

## 2018-11-21
+ 跑通流程：
    + 使用tensorbob构建数据集。
    + 使用slim的pre-trained model搭建模型。
    + 借鉴kaggle kernel，实现f1。
    + 使用tensorbob训练。
    + 完成测试函数，生成csv文件。
+ TODO:
    + `BaseDataset`和`MergedDataset`中可能需要init函数，不然老需要手动处理……
    + multi-gpu training.
    + 模型的改进思路：
        + data argument.
        + model ensemble.
        + f1 loss function.