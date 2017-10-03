"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import numpy as np

from autodp.config.cf_container import Config as cf
from autodp.rl_core.env.action.simple_action import SimpleAction


class SimEnv(object):
    """
    This class simulates a simple environment.
    """
    def __init__(self, state, label):
        """
        Initialize by saving the label.
        :param label:
        """
        self._state = state
        self._label = label

        # For internal use
        self._age = 0
        self._path = []
        self._origin_state = state
        self._actor = SimpleAction()

    @property
    def get_label(self):
        """
        Get the true label of the current environment.
        :return:
        """
        return self._label

    @property
    def get_age(self):
        """
        Get the age of the current environment.
        :return:
        """
        return self._age

    def reset(self, state, label):
        """
        Reset the environment with new state and label.
        :param state:
        :param label:
        :return:
        """
        self._state = state
        self._label = label
        self._origin_state = state
        self._age = 0
        self._path = []

    def restart(self):
        """
        Restart the environment with original state and label.
        :return:
        """
        self._state = self._origin_state
        self._age = 0
        self._path = []

    def step(self, action, qout=None):
        """
        Step one step in the environment.
        :param action:
        :param qout:
        :return:
        """
        # Recover image when it is overaged
        self._age += 1
        if self._age > cf.max_age:
            if qout is None:
                self.restart()

            else:
                action = np.argmax(qout)

        state, reward, done = self._actor.apply_action(action, self._state,
                                                       self._label)

        # Store example
        ex = [self._state, action, state, reward, done]
        self._path.append(ex)

        # Update the state to the new state
        self._state = state

    @property
    def get_path(self):
        """
        Get the list of actions of the current environment.
        :return:
        """
        return self._path






























































