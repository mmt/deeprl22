#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import h5py
import load_policy
import numpy as np
import tensorflow as tf
import tools


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Filename to output to.')
    parser.add_argument('--render', action='store_true')

    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.ERROR)

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    env = tools.Environment(args.envname)
    observations, actions, returns = env.simulate(
        args.max_timesteps, args.num_rollouts, 
        policy_fn, render=args.render)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    f = h5py.File(args.output_file, 'w')
    f.create_dataset('returns', data=returns)    
    f.create_dataset('observations', data=observations)
    f.create_dataset('actions', data=np.array(actions).squeeze())
    f.close()


if __name__ == '__main__':
    main()
