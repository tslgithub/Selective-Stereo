#!/bin/bash

#To evaluate SceneFlow, run
# python evaluate_stereo.py \
#     --restore_ckpt ./weighs/sceneflow/sceneflow.pth 


# To predict Middlebury, run
# python demo_imgs.py --restore_ckpt ./weighs/middlebury/middlebury_finetune.pth  --max_disp 768  --left_imgs '/mnt/Data2/depth/depth20250102/left1/*.png' --right_imgs '/mnt/Data2/depth/depth20250102/right1/*.png' --output_directory output/middlebury
output_path=output/middlebury
mkdir -p ${output_path}

python demo_imgs.py \
    --restore_ckpt ./weighs/middlebury/middlebury_finetune.pth \
    --left_imgs '/mnt/Data2/depth/depth20250102/left1/*.png'  \
    --right_imgs '/mnt/Data2/depth/depth20250102/right1/*.png' \
    --output_directory ${output_path} \
    --precision_dtype float16

