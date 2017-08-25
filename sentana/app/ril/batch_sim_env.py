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
        :param qouts:
        :return:
        """
        states, rewards, dones, trues = [], [], [], []
        for (idx, env) in enumerate(self._envs):
            state, reward, done = env.step(action_batch[idx])
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            trues.append(env.get_label)

        self._envs = list(compress(self._envs, np.logical_not(dones)))

        return states, rewards, dones, trues

    def step_valid(self, action_batch, qouts):
        """
        Step one step for each environment.
        :param action_batch:
        :param qouts:
        :return:
        """
        states, rewards, dones, trues, actions = [], [], [], [], []
        for (idx, env) in enumerate(self._envs):
            state, reward, done, action = env.step_valid(action_batch[idx], qouts[idx])
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            trues.append(env.get_label)
            actions.append(action)

        self._envs = list(compress(self._envs, np.logical_not(dones)))

        return states, rewards, dones, trues, actions

    def add(self, image_batch, label_batch):
        """
        Add more environments.
        :param image_batch:
        :param label_batch:
        :return:
        """
        self._envs += [EnvSim(i, l) for (i, l) in zip(image_batch, label_batch)]

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

    def step_analysis(self, action_batch, qouts):
        """
        Step one step for each environment.
        :param action_batch:
        :param qouts:
        :return:
        """
        for (idx, env) in enumerate(self._envs):
            env.step_analysis(action_batch[idx], qouts[idx])
