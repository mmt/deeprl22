#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import gym
import h5py
import load_policy
import numpy as np
import pickle
import tensorflow as tf
import tf_util

def simulate(envname, max_timesteps, num_rollouts, policy_fn, render=False):
    with tf.Session():
        tf_util.initialize()
    
        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps % 500 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)
        return observations, actions, returns


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Filename to output to.')
    
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    observations, actions, returns = simulate(
        args.envname, args.max_timesteps, args.num_rollouts, 
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
