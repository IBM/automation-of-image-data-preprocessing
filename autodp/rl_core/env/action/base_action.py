import abc

import numpy as np
import cv2 as cv

from autodp import cf


class BaseAction(metaclass=abc.ABCMeta):
    """This abstract class defines basic functions of an action object."""
    @abc.abstractmethod
    def apply_action(self, action, state, label):
        """Main method to apply an action within an environment."""

    @staticmethod
    def _flip(cur_state, flip_code):
        """Flip action."""
        state = cv.flip(cur_state, flip_code)
        return np.reshape(state, [cf.ima_height, cf.ima_width, cf.ima_depth])

    @staticmethod
    def _rotate(cur_state, degree):
        """Rotate action."""
        rows, cols = cur_state.shape[:2]
        matrix = cv.getRotationMatrix2D((cols/2, rows/2), degree, 1)
        state = cv.warpAffine(cur_state, matrix,(cols, rows))
        return np.reshape(state, [cf.ima_height, cf.ima_width, cf.ima_depth])
