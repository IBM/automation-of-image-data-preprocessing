"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os
import abc

from autodp.config.cf_container import Config as cf


class BaseReader(metaclass=abc.ABCMeta):
    """
    This abstract class defines basic functions of a data reader. It is
    necessary to extend this class in order to build a complete reader
    for feeding a TensorFlow program.
    """
    def __init__(self, path, num_epoch):
        """
        Initialization.
        :param path:
        :param num_epoch:
        """
        # Check if input files are available
        if not os.listdir(path):
            raise FileNotFoundError("Input files not found in %s" % path)

        # Store path to data
        self._path = path
        self._num_epoch = num_epoch

    @abc.abstractmethod
    def get_batch(self, batch_size, sess):
        """
        This function is used to read data as batch per time.
        :param batch_size:
        :param sess:
        :return:
        """











