import tensorflow as tf
import gym
import roboschool
import argparse
from mpi4py import MPI
from distutils.util import strtobool
from datetime import datetime
from baselines import logger
from ppo import PPO
from network import MlpPolicy, MlpCritic
from util import throttle, visualise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PPO on your selected environment')
    parser.add_argument('--ntimesteps', default=10000000, type=int, help='number of timesteps to run on the environment')
    parser.add_argument('--env', type=str, default='RoboschoolHumanoid-v1', help='the id of the environment in which to learn')
    parser.add_argument('--train', type=lambda x:bool(strtobool(x)), default=True, help='whether to train the agent or visualise its behavoir')
    parser.add_argument('--save', type=lambda x:bool(strtobool(x)), default=True, help='whether to save the session before exiting')
    parser.add_argument('--comment', type=str, default='', help='comment explaining experiment conditions (added to logdir)')
    args = parser.parse_args()

    log_dir = 'runs/' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + args.comment
    logger.configure(dir=log_dir, format_strs=['tensorboard', 'stdout'])

    env = gym.make(args.env)
    env.seed(0)

    with tf.Session() as sess:
        pi = MlpPolicy(name='pi', action_shape=env.action_space.shape, observation_shape=env.observation_space.shape, hid_size=64, num_hid_layers=3)
        critic = MlpCritic(name='critic', observation_shape=env.observation_space.shape, hid_size=64, num_hid_layers=3)

        ppo = PPO(pi=pi, critic=critic, env=env)

        saver = tf.train.Saver()

        if args.save and MPI.COMM_WORLD.Get_rank() == 0:
            @throttle(minutes=5)
            def save_model(*args):
                saver.save(sess, '../models/test/model.ckpt')
            callback = save_model
        else:
            callback = None

        if args.train:
            ppo.train(max_timesteps=args.ntimesteps, optimizer_stepsize=5e-5, user_callback=callback)
        else:
            visualise(pi=pi, env=env)

        env.close()
