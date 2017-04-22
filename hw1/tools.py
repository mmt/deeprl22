import gym
import numpy as np
import progressbar
import tensorflow as tf
import tf_util

def whiten(D):
    mean = np.mean(D, axis=0)
    N = D.shape[0]
    preprocess = D - mean
    u, s, v = np.linalg.svd(preprocess)
    discard = np.argwhere(s < s[0] * 1e-8)
    nr = discard[0][0] if discard.size > 0 else len(s)
    v = v[:nr, :]
    s = s[:nr]
    scale = v.T.dot(np.diag(np.sqrt(N) / s))
    unscale = np.diag(s / np.sqrt(N)).dot(v)
    preprocess = preprocess.dot(scale)

    return nr, preprocess, mean, scale, unscale

def build_network(graph, connection_widths, apply_nl):
    no = connection_widths[0]
    nu = connection_widths[-1]
    with graph.as_default():
        train_inputs = tf.placeholder(tf.float32, shape=(None, no))
        train_outputs = tf.placeholder(tf.float32, shape=(None, nu))

        A = tf.Variable(tf.truncated_normal([no, nu]))
        b = tf.Variable(tf.truncated_normal([1, nu]))
        linear_layer = tf.matmul(train_inputs, A) + b          

        connection_widths = [no, 500, nu]
        apply_nl = [False, True]
        train_layer = train_inputs
        eval_layer = train_inputs
        for i in range(len(connection_widths) - 1):
            A = tf.Variable(tf.truncated_normal([
                connection_widths[i],
                connection_widths[i + 1]
            ],  stddev=connection_widths[i]**-0.5))
            b = tf.Variable(tf.truncated_normal([
                1, connection_widths[i + 1],
            ], stddev=connection_widths[i]**-0.5))

            if apply_nl[i]:
                train_layer = 1.7159 * tf.nn.tanh((2.0 / 3.0) * train_layer)
                eval_layer = 1.7159 * tf.nn.tanh((2.0 / 3.0) * eval_layer)

            train_layer = tf.matmul(tf.nn.dropout(train_layer, 0.8), A) + b
            eval_layer = tf.matmul(eval_layer, A) + b          

        train_layer += linear_layer
        eval_layer += linear_layer
        loss = tf.nn.l2_loss(train_layer - train_outputs)
        eval_loss = tf.nn.l2_loss(eval_layer - train_outputs)      
        learning_rate = tf.placeholder(tf.float32)            
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    return train_inputs, train_outputs, eval_layer, loss, eval_loss, learning_rate, optimizer

def get_policy(session, train_inputs, train_outputs, eval_layer, input_mean, input_scale, output_mean, output_unscale):
    def trained_policy_fn(obs):
        preprocess_obs = (obs - input_mean).dot(input_scale)
        action = session.run([eval_layer], feed_dict={
            train_inputs: preprocess_obs,
        })
        return np.array(action).dot(output_unscale) + output_mean

    return trained_policy_fn

class Environment():
  def __init__(self, envname):
      self._envname = envname
      self._env = gym.make(envname)      
      
  def simulate(self, max_timesteps, num_rollouts, policy_fn, render=False):
    with tf.Session():
        tf_util.initialize()        
        max_steps = max_timesteps or self._env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        progress = progressbar.ProgressBar()
        for i in progress(range(num_rollouts)):
            obs = self._env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = self._env.step(action)
                totalr += r
                steps += 1
                if render:
                    self._env.render()
                if steps >= max_steps:
                    break
            returns.append(totalr)

        return observations, actions, returns
