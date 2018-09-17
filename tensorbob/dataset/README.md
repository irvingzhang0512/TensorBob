# 数据集相关

## 1. 主要功能
+ `BaseDataset`：封装单个`tf.data.Dataset`对象。
+ `MergedDataset`：封装两个`tf.data.Dataset`对象，使用 feedable iterator 来处理这两个实例，用于训练集/验证集。
+ 提供`get_single_dataset_by_config`方法，根据属性参数（以python dict表示）获取`tf.data.Dataset`实例。
    + 数据预处理
        + 提供了数据增强方法（中心切片、vgg切片、inception切片、水平镜像、垂直镜像、色彩转换）。
        + 提供了数据预处理方法（减去ImageNet平均数，归一化到[0, 1]或[-1, 1]）。
        + 这些具体实现都在`tensorbob.utils.preprocessing`中。
    + 获取`tf.data.Dataset`包括三种：
        + 根据一系列图片的file_path来获取数据集。
        + 根据一系列图片的分类标签（数字）来获取数据集。
        + 根据一系列图片（图像分割标签）的file_path来获取数据集。

## 2. Datasets

### 2.1. ImageNet
+ [参考资料](https://github.com/dontfollowmeimcrazy/imagenet)
+ META File：
    + ImageNet中包括low-level synsets（编号从1-1000）和high-level synsets（编号1000以上）
    + meta数据中包括：
        + ILSVRC2012_ID：编号，数据范围从1开始编号。
        + WNID：Word Net ID的简写，ImageNet或WordNet中的唯一标识。
        + num_children：自分类数量。high-level synsets中肯定不为0，low-level synsets中肯定为0。
        + children：自分类的ILSVRC2012_ID。
        + wordnet_height：整体是个树形图，其中1001为根节点。
+ 文件结构介绍：
    + ILSVRC2012_img_train：train set（图片）
        + 本文件夹中一共有1000个子文件夹，每个文件夹代表一类数据。
        + 每个子文件夹下有若干图片文件。
    + ILSVRC2012_bbox_train：train set（图片定位数据，包括标签）
        + 本文件夹包括1000个子文件夹，每个子文件夹中有若干xml文件，每个xml文件对应一张训练图片。
        + 训练数据（jpeg文件）与标签（xml文件）名称相同，仅文件后缀不同。
    + ILSVRC2012_img_val：val set（图片）
        + 直接保存各类图片。
    + ILSVRC2012_bbox_val: val set（不包括分类标签，图片定位数据）
        + 每张图片对应一个xml文件，其中包括图像定位数据。
    + ILSVRC2012_devkit_t12\data\ILSVRC2012_validation_ground_truth.txt：val set（图片分类标签）
+ `imagenet.py`

### 2.2. VOC2012
+ [参考资料](http://blog.csdn.net/gzhermit/article/details/75729885)
+ 文件结构：
    + Annotation：保存的是`xml`文件，文件名与JPEGImages文件夹中图片对应（仅扩展名不同）。有物体检测、图像基本信息等数据。
    + ImageSets：存放的是每一种类型的challenge对应的图像数据。
        + Action为人的动作，Layout为人体部位，Main为图像识别数据（一共20类），Segmentation存放可用于图像分割的图片编号。
        + _train是训练集图片编号，_val是验证集图片编号，_trainval是两者合并集合，训练集验证集没有交集。
    + JPEGImages：原始图像数据。
    + SegmentationClass：图像分割标签，文件名与JPEGImages中对应。
    + SegmentationObject：物体分隔标签，文件名与JPEGImages中对应。
+ `voc2012.py`

### 2.3. ADEChallenge

### 2.4. CamVid