#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""
import os

# Reduce TensorFlow logging.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import h5py
import load_policy
from matplotlib import pyplot
import numpy as np
import os
import progressbar
import tensorflow as tf
import tools

def train(train_observations, train_actions,
          validation_observations, validation_actions,
          train_policy, session, policy_fn, env, max_timesteps, save_name=None):
    num_epochs = 25000
    progress = progressbar.ProgressBar()
    losses = np.zeros((num_epochs,))
    # Now we normalize the inputs and outputs.

    saver = tf.train.Saver()
    for epoch in progress(range(num_epochs)):

        N = train_observations.shape[0]
        m = 1 # int(np.max((10, N / 1e4)))
        batch_size = 2000
        sel = np.random.choice(range(N), batch_size)
        loss = 0.0
        for _ in range(m):
            loss_i, _ = train_policy.run(
                session, 0.001,
                train_observations[sel, :],
                train_actions[sel, :],
                optimize=True)
            loss += loss_i

        losses[epoch] = loss / m
        if save_name:
            saver.save(session, save_name, global_step=0)
        
        if epoch > 0 and epoch % 1000 == 999:
            if save_name:
                saver.save(session, save_name, global_step=epoch)
            pyplot.figure(22)
            pyplot.cla()
            pyplot.semilogy(losses)
            pyplot.xlim([0, epoch + 1])
            pyplot.show()
            pyplot.pause(0.001)
            validation_loss, _ = train_policy.run(
                session, 0.001,
                validation_observations, validation_actions)
            print 'train loss: %f' % losses[epoch]
            print 'validation loss: %f' % validation_loss

        if epoch % 500 == 499:
            observations, _, returns = env.simulate(
                max_timesteps, 1, train_policy.get_policy(session))
            
            train_observations = np.vstack((
                train_observations, observations
            ))
            no = train_observations.shape[1]
            actions = np.array([policy_fn(obs.reshape((1, no))) for obs in observations]).squeeze()
            train_actions = np.vstack((
                train_actions, actions
            ))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)    
    parser.add_argument("--max_timesteps", type=int)    
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of test rollouts.')
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--load_session_file", type=str, default=None)
    parser.add_argument("--save_name", type=str, default=None)
    args = parser.parse_args()

    env = tools.Environment(args.envname)

    f = h5py.File(args.train_file, 'r')
    train_observations = np.array(f['observations'])
    train_actions = np.array(f['actions'])
    train_returns = np.array(f['returns'])
    f.close()

    f = h5py.File(args.validation_file, 'r')
    validation_observations = np.array(f['observations'])
    validation_actions = np.array(f['actions'])
    f.close()

    tf.logging.set_verbosity(tf.logging.ERROR)
    
    graph = tf.Graph()
    hidden_layer_widths = [500]
    train_policy = tools.TrainPolicy(graph, hidden_layer_widths, train_observations, train_actions)

    pyplot.ion()
    with tf.Session(graph=graph) as session:
        policy_fn = load_policy.load_policy(args.expert_policy_file)
        tf.global_variables_initializer().run()

        if args.load_session_file:
            saver = tf.train.Saver()
            saver.restore(session, args.load_session_file)
        else:
            train(train_observations, train_actions,
                  validation_observations, validation_actions,
                  train_policy, session, policy_fn, env, args.max_timesteps,
                  save_name=args.save_name)
        
        observations, _, returns = env.simulate(
            args.max_timesteps, args.num_rollouts, train_policy.get_policy(session), render=args.render)
              
        print('expert mean return', np.mean(train_returns))
        print('expert std of return', np.std(train_returns))
        
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))


if __name__ == '__main__':
    main()
