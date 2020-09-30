import abc


class BaseRunner(metaclass=abc.ABCMeta):
    """This abstract class defines basic functions of a model runner."""
    @abc.abstractmethod
    def train_model(self):
        """Main method for training."""

    @abc.abstractmethod
    def test_model(self, path):
        """Main method for testing."""
