from datetime import datetime, timedelta
from functools import wraps
import tensorflow as tf
import os

class throttle(object):
    """Decorator that prevents a function from being called more than once every time period.

    To create a function that cannot be called more than once a minute:
        @throttle(minutes=1)
        def my_fun():
            pass

    Args:
        minutes: The minimum number of minutes between function calls

    """
    def __init__(self, minutes=0, seconds=0):
        self.throttle_period = timedelta(minutes=minutes, seconds=seconds)
        self.time_of_last_call = datetime.min

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            now = datetime.now()
            time_since_last_call = now - self.time_of_last_call

            if time_since_last_call > self.throttle_period:
                self.time_of_last_call = now
                return fn(*args, **kwargs)

        return wrapper


class Saver(object):
    """Saves and restores a model to/from a checkpoint file"""

    def __init__(self, model_dir, sess):
        self.saver = tf.train.Saver()
        self.model_dir = model_dir
        self.sess = sess

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def try_restore(self):
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    @throttle(minutes=5)
    def save(self, *args, **kwargs):
        self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'))


def visualise(pi, env):
    """Renders a pretrained policy behaving in an environment.

    Args:
        pi: A policy which implements an act method
        env: An environment which implements an openai gym interface

    """
    while 1:
        frame = score = 0
        obs = env.reset()

        while 1:
            a = pi.act(obs)
            obs, r, done, _ = env.step(a)
            score += r
            frame += 1

            if not env.render("human"): return
            if not done: continue

            print("score=%0.2f in %i frames" % (score, frame))
            break

