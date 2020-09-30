import os
import abc


class BaseReader(metaclass=abc.ABCMeta):
    """This abstract class defines basic functions of a data reader."""
    def __init__(self, path, num_epoch):
        # Check if input files are available
        if not os.listdir(path):
            raise FileNotFoundError("Input files not found in %s" % path)

        # Store path to data
        self._path = path
        self._num_epoch = num_epoch

    @abc.abstractmethod
    def get_batch(self, batch_size, sess):
        """This function is used to read data as batch per time."""
