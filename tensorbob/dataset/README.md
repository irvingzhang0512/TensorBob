# 数据集介绍
+ [参考](https://github.com/dontfollowmeimcrazy/imagenet)

## 1. Base Class Info
+ BaseDataSet:

## 2. Datasets

### 2.1. ImageNet
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

### 2.2. VOC2012
http://blog.csdn.net/gzhermit/article/details/75729885

# 图像增强

## 1. 图片切片

## 2. 图像增强
1. 水平翻转
2. 垂直翻转
3. 亮度
4. 对比度
5. 饱和度
6. 色彩