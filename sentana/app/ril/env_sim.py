"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import cv2 as cv
import numpy as np

from sentana.config.cf_container import Config as cf


class EnvSim(object):
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

    def step(self, action):
        """
        Step one step in the environment.
        :param action:
        :param qout:
        :return:
        """
        # Recover image when it is overaged
        self._age += 1
        if self._age > cf.max_age:
            self.restart()
            reward = None
            done = False
            return self._state, reward, done

        if action <= 9:
            state = self._state
            if self._label == action: reward = 1
            else: reward = -1
            done = True

        elif action == 10:
            state = self._flip(-1)
            reward = 0
            done = False

        elif action == 11:
            state = self._flip(0)
            reward = 0
            done = False

        elif action == 12:
            state = self._flip(1)
            reward = 0
            done = False

        elif action == 13:
            state = self._rotate(90)
            reward = 0
            done = False

        elif action == 14:
            state = self._rotate(-90)
            reward = 0
            done = False

        # Update the state to the new state
        self._state = state

        return state, reward, done

    def step_valid(self, action, qout):
        """
        Step one step in the environment.
        :param action:
        :param qout:
        :return:
        """
        # Recover image when it is overaged
        self._age += 1
        if self._age > cf.max_age:
            action = np.argmax(qout)

        if action <= 9:
            state = self._state
            if self._label == action:
                reward = 1
            else:
                reward = -1
            done = True

        elif action == 10:
            state = self._flip(-1)
            reward = 0
            done = False

        elif action == 11:
            state = self._flip(0)
            reward = 0
            done = False

        elif action == 12:
            state = self._flip(1)
            reward = 0
            done = False

        elif action == 13:
            state = self._rotate(90)
            reward = 0
            done = False

        elif action == 14:
            state = self._rotate(-90)
            reward = 0
            done = False

        # Update the state to the new state
        self._state = state

        return state, reward, done, action

    def _flip(self, flip_code):
        """
        Flip action.
        :param flip_code:
        :return:
        """
        state = cv.flip(self._state, flip_code)

        return np.reshape(state, [28, 28, 1])

    def _crop(self, ratio):
        """
        Crop action.
        :param ratio:
        :return:
        """
        crop_size0 = int(self._state.shape[0] * ratio)
        crop_size1 = int(self._state.shape[1] * ratio)
        d0 = int((self._state.shape[0] - crop_size0) / 2)
        d1 = int((self._state.shape[1] - crop_size1) / 2)

        crop_im = self._state[d0: d0 + crop_size0, d1: d1 + crop_size1, :]
        state = np.zeros(self._state.shape)
        state[d0: d0 + crop_size0, d1: d1 + crop_size1, :] = crop_im

        return state

    def _scale(self, ratio):
        """
        Scale action.
        :param ratio:
        :return:
        """
        height, width = self._state.shape[:2]
        res_im = cv.resize(self._state, (int(width*ratio), int(height*ratio)),
                           interpolation=cv.INTER_CUBIC)

        if ratio > 1:
            d0 = int((res_im.shape[0] - height) / 2)
            d1 = int((res_im.shape[1] - width) / 2)
            state = res_im[d0: d0 + height, d1: d1 + width, :]

        elif ratio < 1:
            d0 = int((height - res_im.shape[0]) / 2)
            d1 = int((width - res_im.shape[1]) / 2)
            state = np.zeros(self._state.shape)
            state[d0:d0+res_im.shape[0],d1:d1+res_im.shape[1],:] = res_im

        return state

    def _rotate(self, degree):
        """
        Rotate action.
        :param degree:
        :return:
        """
        rows, cols = self._state.shape[:2]
        matrix = cv.getRotationMatrix2D((cols/2, rows/2), degree, 1)
        state = cv.warpAffine(self._state, matrix,(cols, rows))

        return np.reshape(state, [28, 28, 1])

    def step_analysis(self, action, qout):
        """
        Step one step in the environment.
        :param action:
        :param qout:
        :return:
        """
        # Recover image when it is overaged
        self._age += 1
        if self._age > cf.max_age:
            action = np.argmax(qout)

        if action <= 9:
            state = self._state
            if self._label == action:
                reward = 1
            else:
                reward = -1
            done = True

        elif action == 10:
            state = self._flip(-1)
            reward = 0
            done = False

        elif action == 11:
            state = self._flip(0)
            reward = 0
            done = False

        elif action == 12:
            state = self._flip(1)
            reward = 0
            done = False

        elif action == 13:
            state = self._rotate(90)
            reward = 0
            done = False

        elif action == 14:
            state = self._rotate(-90)
            reward = 0
            done = False

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
