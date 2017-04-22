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

    # Remove the mean from both inputs and outputs, then use PCA to:
    #
    # 1) Reduce the dimensionality of the data based on a threshold on
    #    the ration with the largest singular value.
    #
    # 2) Make input and output directions uncorrelated and unit variance.
    #
    no_r, input_preprocess, input_mean, input_scale, input_unscale = tools.whiten(train_observations)
    nu_r, output_preprocess, output_mean, output_scale, output_unscale = tools.whiten(train_actions)

    no = len(train_observations[0])
    nu = len(train_actions[0])
    print '%d observations reduced to %d' % (no, no_r)
    print '%d actions reduced to %d' % (nu, nu_r)

    tf.logging.set_verbosity(tf.logging.ERROR)
    
    N = len(train_observations)
    graph = tf.Graph()
    connection_widths = [no_r, 200, 200, nu_r]
    apply_nl = [False, True, True]

    (train_inputs, train_outputs, eval_layer, loss, eval_loss, learning_rate, optimizer
     ) = tools.build_network(graph, connection_widths, apply_nl)

    pyplot.ion()
    with tf.Session(graph=graph) as session:
        policy_fn = load_policy.load_policy(args.expert_policy_file)
        
        tf.global_variables_initializer().run()
        num_epochs = 10000
        progress = progressbar.ProgressBar()
        losses = np.zeros((num_epochs,))
        # Now we normalize the inputs and outputs.

        for epoch in progress(range(num_epochs)):
          _loss = 0.0
          N = output_preprocess.shape[0]
          m = 10 # int(np.max((10, N / 1e4)))
          batch_size = 1000
          sel = np.random.choice(range(N), batch_size)
          for _ in range(m):
              feed_dict = {
                  learning_rate: 0.001,
                  #learning_rate: 0.05 if epoch < 200 else (0.001 if epoch > 1000 else 0.005),
                  train_inputs: input_preprocess[sel, :],
                  train_outputs: output_preprocess[sel, :],
              }
              _, _loss, _eval_loss, _eval_layer = session.run([optimizer, loss, eval_loss, eval_layer], feed_dict=feed_dict)
          losses[epoch] = _eval_loss / len(sel)
          if epoch > 0 and epoch % 500 == 499:
              pyplot.figure(22)
              pyplot.cla()
              pyplot.semilogy(losses)
              pyplot.xlim([0, epoch + 1])
              pyplot.show()
              pyplot.pause(0.001)
              feed_dict = {
                  train_inputs: (validation_observations - input_mean).dot(input_scale),
                  train_outputs: (validation_actions - output_mean).dot(output_scale),
              }
              validation_loss, = session.run([eval_loss], feed_dict=feed_dict)
              print 'train loss: %f' % losses[epoch]
              print 'validation loss: %f' % (validation_loss / len(validation_observations))

          if epoch % 100 == 99:
          #if epoch > 1000 and epoch % 100 == 99:
              policy_fn = tools.get_policy(session, eval_layer,
                                           input_mean, input_scale,
                                           output_mean, output_unscale)
              observations, _, returns = env.simulate(
                  args.max_timesteps, 1, policy_fn)

              input_preprocess = np.vstack((
                  input_preprocess, (observations - input_mean).dot(input_scale)
              ))

              actions = np.array([policy_fn(obs.reshape((1, no))) for obs in observations]).squeeze()
              output_preprocess = np.vstack((
                  output_preprocess, (actions - output_mean).dot(output_scale)
              ))
        policy_fn = tools.get_policy(session, eval_layer,
                                     input_mean, input_scale,
                                     output_mean, output_unscale)

        observations, _, returns = env.simulate(
            args.max_timesteps, args.num_rollouts, trained_policy_fn, render=True)
              
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
