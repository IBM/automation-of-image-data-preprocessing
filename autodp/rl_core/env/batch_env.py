"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import numpy as np
from itertools import compress

from autodp.rl_core.env.sim_env import SimEnv


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
        self._envs = [SimEnv(i, l) for (i, l) in zip(image_batch, label_batch)]

    def step(self, action_batch, qout=None):
        """
        Step one step for each environment.
        :param action_batch:
        :param qout:
        :return:
        """
        if qout is not None:
            for (idx, env) in enumerate(self._envs):
                env.step(action_batch[idx], qout[idx])

        else:
            for (idx, env) in enumerate(self._envs):
                env.step(action_batch[idx])

    def add(self, image_batch, label_batch):
        """
        Add more environments.
        :param image_batch:
        :param label_batch:
        :return:
        """
        self._envs += [SimEnv(i, l) for (i, l) in zip(image_batch, label_batch)]

    def update_done(self, dones):
        """
        Update done images.
        :param dones:
        :return:
        """
        self._envs = list(compress(self._envs, np.logical_not(dones)))

    def get_paths(self):
        """
        Return a list of paths.
        :return:
        """
        paths = [env.get_path for env in self._envs]

        return paths

    def get_path(self, idx):
        """
        Return the path of a specific environment.
        :param idx:
        :return:
        """
        path = self._envs[idx].get_path

        return path

    def get_labels(self):
        """
        Return the list of true labels.
        :return:
        """
        trues = [env.get_label for env in self._envs]

        return trues















































