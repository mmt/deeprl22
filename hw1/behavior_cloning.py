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
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)    
    parser.add_argument("--max_timesteps", type=int)    
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of test rollouts.')
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

    no_r, input_preprocess, input_mean, input_scale, input_unscale = tools.whiten(train_observations)
    nu_r, output_preprocess, output_mean, output_scale, output_unscale = tools.whiten(train_actions)
    no = len(train_observations[0])
    nu = len(train_actions[0])
    print '%d observations reduced to %d' % (no, no_r)
    print '%d actions reduced to %d' % (nu, nu_r)

    graph = tf.Graph()
    connection_widths = [no_r, 500, nu_r]
    apply_nl = [False, True]
    (train_inputs, train_outputs, eval_layer, loss, eval_loss, learning_rate, optimizer
     ) = tools.build_network(graph, connection_widths, apply_nl)
    
    pyplot.ion()
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        num_epochs = 5000
        batch_size = 1000
        m = N / batch_size
        progress = progressbar.ProgressBar()
        losses = []
        for epoch in progress(range(num_epochs)):
          _loss = 0.0
          for i in range(m):
            feed_dict = {
                learning_rate: 0.005,
                train_inputs: input_preprocess[batch_size * i:batch_size * (i+1)],
                train_outputs: output_preprocess[batch_size * i:batch_size * (i+1)]
            }
            _, _loss, _eval_loss, _eval_layer = session.run([optimizer, loss, eval_loss, eval_layer], feed_dict=feed_dict)
          if epoch % 100 == 99:
              losses.append(_eval_loss / batch_size)
              if len(losses) > 1:
                pyplot.figure(22)
                pyplot.cla()
                pyplot.semilogy(losses)
                pyplot.show()
                pyplot.pause(0.001)
        print 'train loss: %f' % losses[-1]
        feed_dict = {
            train_inputs: (validation_observations - input_mean).dot(input_scale),
            train_outputs: (validation_actions - output_mean).dot(output_scale),
        }
        validation_loss, _eval_layer, = session.run([eval_loss, eval_layer], feed_dict=feed_dict)
        print 'validation loss: %f' % (_eval_loss / len(validation_observations.shape))

        policy_fn = tools.get_policy(session, train_inputs, train_outputs, eval_layer,
                                     input_mean, input_scale,
                                     output_mean, output_unscale)

        _, _, returns = env.simulate(
            args.max_timesteps, args.num_rollouts, policy_fn)

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
