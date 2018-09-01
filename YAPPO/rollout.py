import numpy as np


class Segment(object):
    """A container for a series of trajectories generated from an gym environment.

    Args:
        observation_shape: Shape of the observation space
        action_shape: Shape of the action space
        nsteps: The number of transitions to be stored in this segment

    """
    def __init__(self, action_shape, observation_shape, nsteps):
        ac = np.zeros(shape=action_shape)
        ob = np.zeros(shape=observation_shape)

        self.observations = np.array([ob for _ in range(nsteps)])
        self.actions = np.array([ac for _ in range(nsteps)])
        self.rewards = np.zeros(nsteps, 'float32')
        self.dones = np.zeros(nsteps, 'int32')
        self.infos = []
        self.t = 0

    def push_transition(self, observation, action, done, reward, info):
        """Pushes a transition onto the end of the segment

        Args:
            observation: A state observation
            action: The action made while making the observation (i.e the observation before the update caused by the action)
            done: Whether or not the new state is terminal
            reward: The reward bestowed for making the action while making the observation

        """
        self.observations[self.t] = observation
        self.actions[self.t] = action
        self.rewards[self.t] = reward
        self.dones[self.t] = done
        self.infos.append(info)

        self.t += 1

    def reset(self):
        """Resets the head index back to the beginning of the segment.

        """
        self.t = 0
        self.infos = []

    def __len__(self):
        """Gets the current length of the segment

        Returns:
            len: The number of transitions in the segment

        """
        return self.t


class TrajectoryGenerator(object):
    """Steps through a gym environment to generate a segment of trajectories.

    Args:
        pi: A policy which maps state observations to actions
        env: A gym style environment
        nsteps: number of transition tuples to generate

    """
    def __init__(self, pi, env, nsteps):
        self.nsteps = nsteps
        self.pi = pi
        self.env = env

        self.segment = Segment(env.action_space.shape, env.observation_space.shape, nsteps+1)

    def run(self):
        """Runs the generator

        Returns:
            segment: A Segment object containing nsteps worth of transitions

        """
        self.segment.reset()
        done = True

        for _ in range(self.nsteps + 1):
            if done:
                observation = self.env.reset()

            previous_observation = observation

            action = self.pi.act(observation)
            observation, reward, done, info = self.env.step(action)

            self.segment.push_transition(previous_observation, action, done, reward, info)

        return self.segment
