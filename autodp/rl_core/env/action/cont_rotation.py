"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
from autodp.rl_core.env.action.base_action import BaseAction
from autodp import cf


@BaseAction.register
class ContRotation(BaseAction):
    """
    This class implements a simple continuous action on an image such as
    rotation with a parametric degree.
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
        if action < cf.num_class:
            state = cur_state
            if label == action:
                reward = 1
            else:
                reward = -1
            done = True

        else:
            state = self._rotate(cur_state, param_list[0])
            reward = 0
            done = False

        return state, reward, done












































