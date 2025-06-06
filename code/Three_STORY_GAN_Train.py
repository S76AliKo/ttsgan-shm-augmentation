#!/usr/bin/env bash

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=str, default="0")
    parser.add_argument('--node', type=str, default="0015")
    opt = parser.parse_args()

    return opt
args = parse_args()

os.system(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_GAN.py \
-gen_bs 16 \
-dis_bs 16 \
--dist-url 'tcp://localhost:4321' \
--dist-backend 'nccl' \
--world-size 1 \
--rank {args.rank} \
--dataset UniMiB \
--bottom_width 8 \
--max_iter 2000 \
--img_size 32 \
--gen_model my_gen \
--dis_model my_dis \
--df_dim 512 \
--d_heads 3 \
--d_depth 3 \
--g_depth 7,7,7 \
--dropout 0.2 \
--latent_dim 256 \
--gf_dim 2048 \
--num_workers 16 \
--g_lr 0.0001 \
--d_lr 0.0002 \
--optimizer adam \
--loss hinge  \
--wd 1e-3 \
--beta1 0.9 \
--beta2 0.999 \
--phi 1 \
--batch_size 16 \
--num_eval_imgs 50000 \
--init_type xavier_uniform \
--n_critic 1 \
--val_freq 20 \
--print_freq 50 \
--grow_steps 0 0 \
--fade_in 0 \
--patch_size 4 \
--ema_kimg 500 \
--ema_warmup 0.1 \
--ema 0.995 \
--diff_aug translation,cutout,color \
--class_name 3_STORY \
--lambda_freq 0.5 \
--exp_name 3_STORY")