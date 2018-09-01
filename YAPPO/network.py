import tensorflow as tf
import gym
import numpy as np
import random
import baselines.common.tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.distributions import DiagGaussianPdType
from baselines.common import zipsame


class MlpPolicy(object):
    """A multilayer perceptron to map state observations into actions.

    Note:
        The last layer of this network parameterises a diagonal gaussian distribution
        so the output can be stochstic by sampling from the distribution, or deterministic
        by taking the mean.

    Args:
        name: Name of the scope under which to delare all the network's tf variables
        observation_shape: Shape of the observation space
        action_shape: Shape of the action space
        hid_size: Number of neurons per hidden layer
        num_hid_layers: Number of hidden layers
        stochastic: Whether to sample the output distribution or take its mean when generating actions

    """
    def __init__(self, name, observation_shape, action_shape, hid_size, num_hid_layers, stochastic=True):
        with tf.variable_scope(name):
            self.stochastic = stochastic
            self.hid_size, self.num_hid_layers = hid_size, num_hid_layers
            self.action_shape, self.observation_shape = action_shape, observation_shape
            self.scope = tf.get_variable_scope().name
            self.pdtype = DiagGaussianPdType(action_shape[0])

            observations_ph = U.get_placeholder(name='ob', dtype=tf.float32, shape=[None] + list(observation_shape))
            stochastic_ph = tf.placeholder(dtype=tf.bool, shape=())

            with tf.variable_scope('obfilter'):
                self.ob_rms = RunningMeanStd(shape=observation_shape)

            with tf.variable_scope('pol'):
                last_out = tf.clip_by_value((observations_ph - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
                for i in range(num_hid_layers):
                    last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))

                mean = tf.layers.dense(last_out, self.pdtype.param_shape()[0]//2, name='final', kernel_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(name='logstd', shape=[1, self.pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)

            self.pd = self.pdtype.pdfromflat(pdparam)

            action_op = U.switch(stochastic_ph, self.pd.sample(), self.pd.mode())
            self._act = U.function([stochastic_ph, observations_ph], action_op)

    def act(self, observation):
        """Convenience function for generating a single action given an observation

        Args:
            observation: A state observation

        """
        return self._act(self.stochastic, np.array(observation)[None])[0]

    def get_variables(self):
        """Gets all the tf variables associated with this network."""
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        """Gets all the trainable tf variables associated with this network."""
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def make_target_network(self, name):
        """Creates a network which periodically updates its weights by copying them from this network.

        Args:
            name: Name of the scope under which to delare all the target network's tf variables

        """
        return TargetMlpPolicy(name, self)


class TargetMlpPolicy(MlpPolicy):
    """A target network for an MlpPolicy.

    This class has the same network structure and behaviour as an MlpPolicy but can also copy
    the weights from its target on calls to the update method. PPO doesn't technically call for target
    networks, but it just so happens that the logic required to maintain the policy from timestep t-1
    is fairly similar so I've named it as such.

    Args:
        name: Name of the scope under which to delare all the network's tf variables
        target: An instance of MlpPolicy to copy weights from

    """
    def __init__(self, name, target):
        super(TargetMlpPolicy, self).__init__(name=name, action_shape=target.action_shape, observation_shape=target.observation_shape, hid_size=target.hid_size, num_hid_layers=target.num_hid_layers)
        self.update = U.function([], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in zipsame(self.get_variables(), target.get_variables())])


class MlpCritic(object):
    """A multilayer perceptron to map state observations into expected cumulative reward.

    Args:
        name: Name of the scope under which to delare all the network's tf variables
        observation_shape: Shape of the observation space
        hid_size: Number of neurons per hidden layer
        num_hid_layers: Number of hidden layers

    """
    def __init__(self, name, observation_shape, hid_size, num_hid_layers):
        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name

            observations_ph = U.get_placeholder(name='ob', dtype=tf.float32, shape=[None] + list(observation_shape))

            with tf.variable_scope('obfilter'):
                self.ob_rms = RunningMeanStd(shape=observation_shape)

            with tf.variable_scope('vf'):
                last_out = tf.clip_by_value((observations_ph - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
                for i in range(num_hid_layers):
                    last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))
                self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

            self.predict = U.function([observations_ph], self.vpred)

    def get_trainable_variables(self):
        """Gets all the trainable tf variables associated with this network."""
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_variables(self):
        """Gets all the tf variables associated with this network."""
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
