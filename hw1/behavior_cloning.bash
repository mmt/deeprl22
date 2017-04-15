#!/bin/bash
set -eux
for e in Hopper-v1 Ant-v1 HalfCheetah-v1 Humanoid-v1 Walker2d-v1
do
    python behavior_cloning.py $e --num_rollouts=10 --train_file=train/$e.h5 \
      --validation_file=validation/$e.h5 --output_file=behavior_cloning_test/$e.h5
done
