"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import abc


class BaseRunner(metaclass=abc.ABCMeta):
    """This abstract class defines basic functions of a model runner."""
    def __init__(self):
        pass

    @abc.abstractmethod
    def train_model(self, cont):
        """Main method for training."""

    @abc.abstractmethod
    def test_model(self, path, fh):
        """Main method for testing."""
