#!/bin/bash
set -eux
for e in Hopper-v1 Ant-v1 HalfCheetah-v1 Humanoid-v1 Reacher-v1 Walker2d-v1
do
    python run_expert.py experts/$e.pkl $e --render --num_rollouts=10 --output_file=train/$e.h5
    python run_expert.py experts/$e.pkl $e --num_rollouts=1 --output_file=test/$e.h5
done
