"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
from autodp.rl_core.env.action.base_action import BaseAction
from autodp import cf


@BaseAction.register
class MultiSimpleAction(BaseAction):
    """
    This class implements simple actions on an image such as flip, rotate, etc.
    """
    def __init__(self):
        """
        Temporarily do nothing for initialization.
        """
        pass

    def apply_action(self, action, cur_state, label, param_list):
        """
        Main method to apply an action within an environment.
        :param action:
        :param cur_state:
        :param label:
        :param param_list:
        :return:
        """
        tlcr, brcr = False, False
        if action < cf.num_class:
            state = cur_state
            if label == action:
                reward = cf.num_class - 1
            else:
                reward = -1
            done = True

        elif action == cf.num_class + 0:
            state = self._flip(cur_state, -1)
            reward = 0
            done = False

        elif action == cf.num_class + 1:
            state = self._flip(cur_state, 0)
            reward = 0
            done = False

        elif action == cf.num_class + 2:
            state = self._flip(cur_state, 1)
            reward = 0
            done = False

        elif action == cf.num_class + 3:
            state = self._rotate(cur_state, 1)
            reward = 0
            done = False

        elif action == cf.num_class + 4:
            state = self._rotate(cur_state, 5)
            reward = 0
            done = False

        elif action == cf.num_class + 5:
            state = self._rotate(cur_state, -1)
            reward = 0
            done = False

        elif action == cf.num_class + 6:
            state = self._rotate(cur_state, -5)
            reward = 0
            done = False

        elif action == cf.num_class + 7:
            state = self._top_left_crop(cur_state, param_list[-2])
            reward = 0
            done = False
            tlcr = True

        elif action == cf.num_class + 8:
            state = self._bottom_right_crop(cur_state, param_list[-1])
            reward = 0
            done = False
            brcr = True

        return state, reward, done, tlcr, brcr

































































