"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import abc


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
    def apply_action(self, action, state, label):
        """
        Main method to apply an action within an environment.
        :param action:
        :param state:
        :param label:
        :return:
        """



































