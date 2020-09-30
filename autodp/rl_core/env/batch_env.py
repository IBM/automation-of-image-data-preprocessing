import numpy as np
from itertools import compress

from autodp.rl_core.env.sim_env import SimEnv


class BatchSimEnv(object):
    """This class is a wrapper to create and control a list of environments for batch processing."""
    def __init__(self, image_batch=[], label_batch=[]):
        self._envs = [SimEnv(i, l) for (i, l) in zip(image_batch, label_batch)]

    def step(self, action_batch, qout=None):
        """Step one step for each environment."""
        if qout is not None:
            for (idx, env) in enumerate(self._envs):
                env.step(action_batch[idx], qout[idx])
        else:
            for (idx, env) in enumerate(self._envs):
                env.step(action_batch[idx])

    def add(self, image_batch, label_batch):
        """Add more environments."""
        self._envs += [SimEnv(i, l) for (i, l) in zip(image_batch, label_batch)]

    def update_done(self, dones):
        """Update done images."""
        self._envs = list(compress(self._envs, np.logical_not(dones)))

    def get_paths(self):
        """Return a list of paths."""
        return [env.get_path for env in self._envs]

    def get_path(self, idx):
        """Return the path of a specific environment."""
        return self._envs[idx].get_path

    def get_labels(self):
        """Return the list of true labels."""
        return [env.get_label for env in self._envs]
