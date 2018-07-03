python train.py \
    --logtostderr \
    --train_dir=./logs_whale_resnet50_train/ \
    --pipeline_config_path=/home/tensorflow05/zyy/tensorbob/examples/whale/od/faster_rcnn_resnet50_whale.config

python eval.py \
    --logtostderr \
    --checkpoint_dir=./logs_whale_resnet50/ \
    --eval_dir=./logs_whale_resnet50_eval/ \
    --pipeline_config_path=/home/tensorflow05/zyy/tensorbob/examples/whale/od/faster_rcnn_resnet50_whale.config


python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path=/home/tensorflow05/zyy/tensorbob/examples/whale/od/faster_rcnn_resnet50_whale.config \
    --trained_checkpoint_prefix ./logs_whale_resnet50_train/model.ckpt-3569 \
    --output_directory ./logs_whale_resnet50_export/


python train.py \
    --logtostderr \
    --train_dir=./logs_whale_resnet101_train/ \
    --pipeline_config_path=/home/ubuntu/bob/TensorBob/examples/whale/od/faster_rcnn_resnet101_whale.config


python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path=/home/ubuntu/bob/TensorBob/examples/whale/od/faster_rcnn_resnet101_whale.config \
    --trained_checkpoint_prefix ./logs_whale_resnet101_train/model.ckpt-10918 \
    --output_directory ./logs_whale_resnet101_export/

