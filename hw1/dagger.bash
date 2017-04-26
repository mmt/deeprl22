#!/bin/bash
set -eux
for e in Hopper-v1 Ant-v1 HalfCheetah-v1 Humanoid-v1 Walker2d-v1 Reacher-v1
do
    mkdir -p $e
    python dagger.py experts/$e.pkl $e --num_rollouts=10 --train_file=train/$e.h5 \
      --validation_file=validation/$e.h5 --save_name=$e/dagger.ckpt
done
