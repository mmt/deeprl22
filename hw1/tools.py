import gym
import numpy as np
import progressbar
import tensorflow as tf
import tf_util

def _whiten(D):
    mean = np.mean(D, axis=0)
    N = D.shape[0]
    u, s, v = np.linalg.svd(D - mean)
    discard = np.argwhere(s < s[0] * 1e-10)
    nr = discard[0][0] if discard.size > 0 else len(s)
    v = v[:nr, :]
    s = s[:nr]
    scale = v.T.dot(np.diag(np.sqrt(N) / s))
    unscale = np.diag(s / np.sqrt(N)).dot(v)

    return nr, mean, scale, unscale

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


class TrainPolicy():
    def __init__(self, graph, hidden_layer_widths, initial_inputs, initial_outputs):
        (self._no, self._input_mean,
         self._input_scale, self._input_unscale) = _whiten(initial_inputs)
        (self._nu, self._output_mean,
         self._output_scale, self._output_unscale) = _whiten(initial_outputs)

        connection_widths = [self._no] + hidden_layer_widths + [self._nu]
        no = connection_widths[0]
        nu = connection_widths[-1]
        with graph.as_default():
            self._train_inputs = tf.placeholder(tf.float32, shape=(None, self._no))
            self._train_outputs = tf.placeholder(tf.float32, shape=(None, self._nu))

            A = tf.Variable(tf.zeros([self._no, self._nu]))
            b = tf.Variable(tf.zeros([1, self._nu]))
            linear_layer = tf.matmul(self._train_inputs, A) + b

            train_layer = self._train_inputs
            self._eval_layer = self._train_inputs
            for i in range(len(connection_widths) - 1):
                A = tf.Variable(tf.truncated_normal([
                    connection_widths[i],
                    connection_widths[i + 1]
                ],  stddev=connection_widths[i]**-0.5))
                b = tf.Variable(tf.truncated_normal([
                    1, connection_widths[i + 1],
                ], stddev=connection_widths[i]**-0.5))

                if i > 0:
                    train_layer = 1.7159 * tf.nn.tanh((2.0 / 3.0) * train_layer)
                    self._eval_layer = 1.7159 * tf.nn.tanh((2.0 / 3.0) * self._eval_layer)

                train_layer = tf.matmul(tf.nn.dropout(train_layer, 0.8), A) + b
                self._eval_layer = tf.matmul(self._eval_layer, A) + b          

            train_layer += linear_layer
            self._eval_layer += linear_layer
            self._loss = tf.nn.l2_loss(tf.matmul(
                train_layer - self._train_outputs, self._output_unscale))
            self._eval_loss = tf.nn.l2_loss(tf.matmul(
                self._eval_layer - self._train_outputs, self._output_unscale))
            self._learning_rate = tf.placeholder(tf.float32)            
            self._optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self._loss)

    def run(self, session, learning_rate, inputs, outputs, optimize=False):
        feed_dict = {
            self._learning_rate: learning_rate,
            self._train_inputs: (inputs - self._input_mean).dot(self._input_scale),
            self._train_outputs: (outputs - self._output_mean).dot(self._output_scale),
        }
        if optimize:
            _, loss = session.run([self._optimizer, self._eval_loss], feed_dict=feed_dict)
        else:
            loss, = session.run([self._eval_loss], feed_dict=feed_dict)
        return loss / inputs.shape[0]

    def get_policy(self, session):
        def trained_policy_fn(obs):
            preprocess_obs = (obs - self._input_mean).dot(self._input_scale)
            action = session.run([self._eval_layer], feed_dict={
                self._train_inputs: preprocess_obs,
            })
            return np.array(action).dot(self._output_unscale) + self._output_mean

        return trained_policy_fn
