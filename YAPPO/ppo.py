import tensorflow as tf
import numpy as np
import gym
from mpi4py import MPI
from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from .rollout import TrajectoryGenerator


class PPO(object):
    """Yet Another Proximal Policy Optimisation Implementation!

    This class is based on ppo1 from the openai baselines but removes
    a bit of bloat and a few unnecessarily complicated algorithmic designs.

    Args:
        env: The gym environment in which to train
        pi: The network representing the policy
        critic: A network representing the value function
        mu: A behavioral policy used to generate the trajectories to train pi
        clip_param: The range within the old/new pi KL divergence term of the PPO loss function should be clipped
        entropy_coefficient: Amount of penalty to apply for policy entropy
        adam_epsilon: A really small number used by the Adam optimiser to prevent divzero errors
        timesteps_per_actorbatch: The number of timesteps per optimiser update

    """
    def __init__(self, env, pi, critic, clip_param=0.2, entropy_coefficient=0.0, adam_epsilon=1e-5, timesteps_per_actorbatch=2048):
        self.env = env
        self.pi = pi
        self.critic = critic

        # Set up a segment generator for rollouts
        self.segment_generator = TrajectoryGenerator(self.pi, self.env, timesteps_per_actorbatch)

        # Clone the policy so we can calculate the relative likelihood of a given action pre and post update
        self.old_pi = pi.make_target_network(name='oldpi')

        # Set up placeholders for the observations, actions and target advantages and returns
        observation_ph = U.get_placeholder_cached(name='ob')
        action_ph = self.pi.pdtype.sample_placeholder([None])
        advantage_target_ph = tf.placeholder(dtype=tf.float32, shape=[None])
        return_ph = tf.placeholder(dtype=tf.float32, shape=[None])

        # Set up PPO's clipped surrogate objective (L^CLIP)
        ratio = tf.exp(self.pi.pd.logp(action_ph) - self.old_pi.pd.logp(action_ph))
        surr1 = ratio * advantage_target_ph
        surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage_target_ph
        clipped_surrogate_objective = -tf.reduce_mean(tf.minimum(surr1, surr2))

        # Set up the value function loss
        value_function_loss = tf.reduce_mean(tf.square(self.critic.vpred - return_ph))

        # Set up the policy entropy loss
        entropy = self.pi.pd.entropy()
        mean_entropy = tf.reduce_mean(entropy)
        policy_entropy_penalty = (-entropy_coefficient) * mean_entropy

        # Tally up the three loss components
        total_loss = clipped_surrogate_objective + policy_entropy_penalty + value_function_loss

        losses = [clipped_surrogate_objective, value_function_loss, mean_entropy, policy_entropy_penalty]
        self.loss_names = ['L^CLIP', 'Value Function Loss', 'Policy Entropy', 'Policy Entropy Penalty']

        var_list = self.pi.get_trainable_variables() + self.critic.get_trainable_variables()
        self.lossandgrad = U.function([observation_ph, action_ph, advantage_target_ph, return_ph], losses + [U.flatgrad(total_loss, var_list)])

        self.adam = MpiAdam(var_list, epsilon=adam_epsilon)

        U.initialize()
        self.adam.sync()

    def estimate_advantage(self, segment, value_predictions, gamma, lam):
        """Estimates the advantage for a given set of state observations using GAE(lambda)

        Note:
            The lambda in this function has nothing to do with Python lambdas, it's just a greek symbol

        Args:
            segment: A segment containing trajectories of (S, A, R, done) tuples
            gamma: Future discounting coefficient
            lam: The amount of credit assigned past states for a present state reward

        """
        gaelam = np.empty(len(segment)-1, 'float32')
        lastgaelam = 0

        for t in reversed(range(len(segment)-1)):
            nonterminal = 1 - segment.dones[t]
            delta = segment.rewards[t] + gamma * value_predictions[t + 1] * nonterminal - value_predictions[t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

        return gaelam

    def train(self, max_timesteps, optimizer_stepsize=3e-4, optimizer_epochs=10, optimizer_batchsize=64, gamma=0.99, lam=0.95, user_callback=None):
        """Trains the policy pi using PPO

        Args:
            max_timesteps: The number of timesteps over which to train
            optimizer_stepsize: The Adam learning rate
            optimizer_epochs: The number of optimisation epochs per optimiser update
            optimizer_batchsize: The batch size for each optimiser epoch
            gamma: Future discounting coefficient
            lam: The amount of credit assigned past states for a present state reward
            user_callback: User defined hook for saving models etc
            (it's called every loop so it's a good idea to throttle this callback)

        """
        episodes_so_far = timesteps_so_far = iterations_so_far = 0

        while timesteps_so_far < max_timesteps:

            segment = self.segment_generator.run()

            value_predictions = self.critic.predict(segment.observations)
            advantage_targets = self.estimate_advantage(segment, value_predictions, gamma, lam)

            td_lambda_return = advantage_targets + value_predictions[:-1]
            observations, actions = segment.observations[:-1], segment.actions[:-1]

            # predicted value function before udpate
            value_predictions_before = value_predictions[:-1]

            # standardized advantage function estimate
            advantage_targets = (advantage_targets - advantage_targets.mean()) / advantage_targets.std()
            d = Dataset(dict(ob=observations, ac=actions, atarg=advantage_targets, vtarg=td_lambda_return), shuffle=True)
            optimizer_batchsize = optimizer_batchsize or observations.shape[0]

            # update running mean/std for policy and critic
            self.pi.ob_rms.update(observations)
            self.critic.ob_rms.update(observations)

            # set old parameter values to new parameter values
            self.old_pi.update()

            # Here we do a bunch of optimization epochs over the data
            for _ in range(optimizer_epochs):
                # list of tuples, each of which gives the loss for a minibatch
                losses = []
                for batch in d.iterate_once(optimizer_batchsize):
                    *newlosses, grads = self.lossandgrad(batch['ob'], batch['ac'], batch['atarg'], batch['vtarg'])
                    self.adam.update(grads, optimizer_stepsize)
                    losses.append(newlosses)

            timesteps = len(segment)
            episodes = np.count_nonzero(segment.dones) + 1
            returns = np.sum(segment.rewards)

            total_timesteps, total_episodes, total_returns = map(sum, zip(*MPI.COMM_WORLD.allgather([timesteps, episodes, returns])))
            timesteps_so_far += total_timesteps

            episodes_so_far += total_episodes
            iterations_so_far += 1

            mean_losses, _, _ = mpi_moments(losses, axis=0)

            logger.log('********** Iteration %i ************' % iterations_so_far)

            logger.record_tabular('Episodes So Far', episodes_so_far)
            logger.record_tabular('Timesteps So Far', timesteps_so_far)
            logger.record_tabular('Percent Complete', timesteps_so_far * 100. / max_timesteps)

            logger.record_tabular('Episode Length Mean', total_timesteps / total_episodes)
            logger.record_tabular('Episode Reward Mean', total_returns / total_episodes)
            logger.record_tabular('Episodes This Iteration', total_episodes)

            logger.record_tabular('Explained Variance TD(lam)', explained_variance(value_predictions_before, td_lambda_return))

            for (lossval, name) in zipsame(mean_losses, self.loss_names):
                logger.record_tabular(name, lossval)

            if user_callback is not None:
                user_callback(locals(), globals())

            if MPI.COMM_WORLD.Get_rank() == 0:
                logger.dump_tabular()
