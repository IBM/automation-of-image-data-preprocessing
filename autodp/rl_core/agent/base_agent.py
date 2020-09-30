import abc

import pickle

from autodp import cf


class BaseAgent(metaclass=abc.ABCMeta):
    """This abstract class defines methods exposed by a reinforcement learning agent."""
    def __init__(self):
        # Setup neural network functions
        self._setup_policy()

    @staticmethod
    def _add_extra_example(env):
        """Strategy to add extra examples to the training set."""
        extra = []
        paths = env.get_paths()
        for path in paths:
            if path[-1][1] < cf.num_class:
                extra.extend(path)
        return extra

    @staticmethod
    def _compute_done(env):
        """Search for done images."""
        rewards, dones, states, actions = [], [], [], []
        trues = env.get_labels()
        paths = env.get_paths()
        for path in paths:
            rewards.append(sum([ex[3] for ex in path]))
            dones.append(path[-1][4])
            states.append(path[-1][2])
            actions.append(path[-1][1])
        return rewards, dones, states, actions, trues

    @staticmethod
    def _store_prep_images(fh, images, labels):
        """Store preprocessed images."""
        for (image, label) in zip(images, labels):
            line = {"i": image, "l": label}
            pickle.dump(line, fh, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _get_current_step(env):
        """Get examples from the latest step in the batch."""
        extra = []
        paths = env.get_paths()
        for path in paths:
            extra.append(path[-1])
        return extra

    @abc.abstractmethod
    def _setup_policy(self):
        """Build one or more networks to approximate value, action-value and policy functions."""

    @abc.abstractmethod
    def train_policy(self, sess, train_reader, valid_reader, verbose):
        """Policy improvement and evaluation."""

    @abc.abstractmethod
    def predict(self, sess, reader):
        """Apply the policy to predict image classification."""
