"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import cv2 as cv
import numpy as np


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

    def reset(self, state, label):
        """
        Reset the environment with new state and label.
        :param state:
        :param label:
        :return:
        """
        self._state = state
        self._label = label

    def step(self, action):
        """
        Step one step in the environment.
        :param action:
        :return:
        """
        if action == 0 or action == 1:
            state = self._state
            if self._label == action: reward = 1
            else: reward = -1
            done = True

        elif action == 2:
            state = self._flip(-1)
            reward = 0
            done = False

        elif action == 3:
            state = self._flip(0)
            reward = 0
            done = False

        elif action == 4:
            state = self._flip(1)
            reward = 0
            done = False

        elif action == 5:
            state = self._crop(0.9)
            reward = 0
            done = False

        elif action == 6:
            state = self._scale(1.1)
            reward = 0
            done = False

        elif action == 7:
            state = self._rotate(-15)
            reward = 0
            done = False

        elif action == 8:
            state = self._rotate(15)
            reward = 0
            done = False

        return state, reward, done

    def _flip(self, flip_code):
        """
        Flip action.
        :param flip_code:
        :return:
        """
        self._state = cv.flip(self._state, flip_code)

        return self._state

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
        self._state = np.zeros(self._state.shape)
        self._state[d0: d0 + crop_size0, d1: d1 + crop_size1, :] = crop_im

        return self._state

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
            self._state = res_im[d0: d0 + height, d1: d1 + width, :]

        elif ratio < 1:
            d0 = int((height - res_im.shape[0]) / 2)
            d1 = int((width - res_im.shape[1]) / 2)
            self._state = np.zeros(self._state.shape)
            self._state[d0:d0+res_im.shape[0],d1:d1+res_im.shape[1],:] = res_im

        return self._state

    def _rotate(self, degree):
        """
        Rotate action.
        :param degree:
        :return:
        """
        rows, cols = self._state.shape[:2]
        matrix = cv.getRotationMatrix2D((cols/2, rows/2), degree, 1)
        self._state = cv.warpAffine(self._state, matrix,(cols, rows))

        return self._state
