#!/usr/bin/env bash


python train_triplet.py \
  --dataroot='data/train'  \
  --lfw-dir='data/val' \
  --lfw-pairs-path='./lfw_pairs.txt'  \
  --log-dir='./log_test' \
  --epochs=3 \
  --embedding-size=32 \
  --batch-size=64 \
  --n-triplets=1000 \

