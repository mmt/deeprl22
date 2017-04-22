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

import gym
import h5py
from matplotlib import pyplot
import numpy as np
import os
import progressbar
import tensorflow as tf
import tools


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of test rollouts.')
    parser.add_argument("--output_file", type=str, default=None)    
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--validation_file", type=str, default=None)
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

    N = train_observations.shape[0]

    tf.logging.set_verbosity(tf.logging.ERROR)

    graph = tf.Graph()
    hidden_layer_widths = [500]
    train_policy = tools.TrainPolicy(graph, hidden_layer_widths, train_observations, train_actions)

    pyplot.ion()
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        num_epochs = 2000
        batch_size = 1000
        m = N / batch_size
        losses = np.zeros((m * num_epochs,))
        progress = progressbar.ProgressBar()
        for epoch in progress(range(num_epochs)):
          _loss = 0.0
          for i in range(m):
              loss = train_policy.run(
                  session, 0.001,
                  train_observations[batch_size * i:batch_size * (i+1)],
                  train_actions[batch_size * i:batch_size * (i+1)],
                  optimize=True)
              losses[epoch * m + i] = loss

          if epoch % 100 == 99:
              pyplot.figure(22)
              pyplot.cla()
              pyplot.semilogy(losses[:(epoch + 1) * m - 1])
              pyplot.show()
              pyplot.pause(0.001)
        print 'train loss: %f' % losses[(epoch + 1) * m - 1]
        validation_loss = train_policy.run(
            session, 0.001,
            validation_observations, validation_actions)
        print 'validation loss: %f' % validation_loss

        _, _, returns = env.simulate(
            args.max_timesteps, args.num_rollouts, train_policy.get_policy(session),
            render=args.render)

        #print('expert returns', train_returns)
        print('expert mean return', np.mean(train_returns))
        print('expert std of return', np.std(train_returns))
        
        #print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        f=h5py.File(args.output_file, 'w')
        f.create_dataset('losses', data=losses)
        f.create_dataset('returns', data=returns)
        f.close()
        

if __name__ == '__main__':
    main()
