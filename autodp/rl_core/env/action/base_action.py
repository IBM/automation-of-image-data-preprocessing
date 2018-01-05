"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import abc

import numpy as np
import cv2 as cv

from autodp import cf


class BaseAction(metaclass=abc.ABCMeta):
    """
    This abstract class defines basic functions of an action object. It is
    necessary to extend this class in order to setup a new action class.
    """
    def __init__(self):
        """
        Temporarily do nothing for initialization.
        """
        pass

    @abc.abstractmethod
    def apply_action(self, action, state, label, param_list):
        """
        Main method to apply an action within an environment.
        :param action:
        :param state:
        :param label:
        :param param_list:
        :return:
        """

    @staticmethod
    def _flip(cur_state, flip_code):
        """
        Flip action.
        :param cur_state:
        :param flip_code:
        :return:
        """
        state = cv.flip(cur_state, flip_code)

        return np.reshape(state, [cf.ima_height, cf.ima_width, cf.ima_depth])

    @staticmethod
    def _crop(cur_state, ratio):
        """
        Crop action.
        :param cur_state:
        :param ratio:
        :return:
        """
        crop_size0 = int(cur_state.shape[0] * ratio)
        crop_size1 = int(cur_state.shape[1] * ratio)
        d0 = int((cur_state.shape[0] - crop_size0) / 2)
        d1 = int((cur_state.shape[1] - crop_size1) / 2)

        crop_im = cur_state[d0: d0 + crop_size0, d1: d1 + crop_size1, :]
        state = np.zeros(cur_state.shape)
        state[d0: d0 + crop_size0, d1: d1 + crop_size1, :] = crop_im

        return np.reshape(state, [cf.ima_height, cf.ima_width, cf.ima_depth])

    @staticmethod
    def _top_left_crop(cur_state, ratio):
        """
        Crop action.
        :param cur_state:
        :param ratio:
        :return:
        """
        crop_size0 = int(cur_state.shape[0] * ratio)
        crop_size1 = int(cur_state.shape[1] * ratio)
        d0 = cur_state.shape[0] - crop_size0
        d1 = cur_state.shape[1] - crop_size1

        crop_im = cur_state[d0: d0 + crop_size0, d1: d1 + crop_size1, :]
        state = np.zeros(cur_state.shape)
        state[d0: d0 + crop_size0, d1: d1 + crop_size1, :] = crop_im

        return np.reshape(state, [cf.ima_height, cf.ima_width, cf.ima_depth])

    @staticmethod
    def _bottom_right_crop(cur_state, ratio):
        """
        Crop action.
        :param cur_state:
        :param ratio:
        :return:
        """
        crop_size0 = int(cur_state.shape[0] * ratio)
        crop_size1 = int(cur_state.shape[1] * ratio)

        crop_im = cur_state[0:crop_size0, 0:crop_size1, :]
        state = np.zeros(cur_state.shape)
        state[0:crop_size0, 0:crop_size1, :] = crop_im

        return np.reshape(state, [cf.ima_height, cf.ima_width, cf.ima_depth])

    @staticmethod
    def _scale(cur_state, ratio):
        """
        Scale action.
        :param cur_state:
        :param ratio:
        :return:
        """
        height, width = cur_state.shape[:2]
        res_im = cv.resize(cur_state, (int(width*ratio), int(height*ratio)),
                           interpolation=cv.INTER_CUBIC)

        if ratio > 1:
            d0 = int((res_im.shape[0] - height) / 2)
            d1 = int((res_im.shape[1] - width) / 2)
            state = res_im[d0: d0 + height, d1: d1 + width, :]

        elif ratio < 1:
            d0 = int((height - res_im.shape[0]) / 2)
            d1 = int((width - res_im.shape[1]) / 2)
            state = np.zeros(cur_state.shape)
            state[d0:d0+res_im.shape[0],d1:d1+res_im.shape[1],:] = res_im

        return np.reshape(state, [cf.ima_height, cf.ima_width, cf.ima_depth])

    @staticmethod
    def _rotate(cur_state, degree):
        """
        Rotate action.
        :param cur_state:
        :param degree:
        :return:
        """
        rows, cols = cur_state.shape[:2]
        matrix = cv.getRotationMatrix2D((cols/2, rows/2), degree, 1)
        state = cv.warpAffine(cur_state, matrix,(cols, rows))

        return np.reshape(state, [cf.ima_height, cf.ima_width, cf.ima_depth])

















































