"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import abc


class BaseRunner(metaclass=abc.ABCMeta):
    """
    This abstract class defines basic functions of a model runner. It is
    necessary to extend this class in order to build a complete runner.
    """
    def __init__(self):
        """
        Temporarily do nothing for initialization.
        """
        pass

    @abc.abstractmethod
    def train_model(self, cont):
        """
        Main method for training.
        :param cont:
        :return:
        """

    @abc.abstractmethod
    def test_model(self):
        """
        Main method for testing.
        :return:
        """



































