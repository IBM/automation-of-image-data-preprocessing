"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import numpy as np
from itertools import compress

from sentana.app.ril.env_sim import EnvSim


class BatchSimEnv(object):
    """
    This class is a wrapper to create and control a list of environments for
    batch processing.
    """
    def __init__(self, image_batch=[], label_batch=[]):
        """
        Initialize a list of environments.
        :param image_batch:
        :param label_batch:
        """
        self._envs = [EnvSim(i, l) for (i, l) in zip(image_batch, label_batch)]

    def step(self, action_batch):
        """
        Step one step for each environment.
        :param action_batch:
        :return:
        """
        states, rewards, dones, trues, ages = [], [], [], [], []
        for (idx, env) in enumerate(self._envs):
            state, reward, done = env.step(action_batch[idx])
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            trues.append(env.get_label)
            ages.append(env.get_age)

        self._envs = list(compress(self._envs, np.logical_not(dones)))

        return states, rewards, dones, trues, ages

    def add(self, image_batch, label_batch):
        """
        Add more environments.
        :param image_batch:
        :param label_batch:
        :return:
        """
        self._envs += [EnvSim(i, l) for (i, l) in zip(image_batch, label_batch)]
