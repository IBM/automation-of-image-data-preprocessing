"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import random


class ExpBuffer(object):
    """
    This is a utility class used to stored train data of a reinforcement
    learning algorithm.
    """
    def __init__(self, buf_size=5000):
        """
        Initialize the buffer.
        :param buffer_size:
        """
        self._buffer = []
        self._buf_size = buf_size

    def add(self, exps):
        """
        Add a list of experiences into the buffer.
        :param exps: list of experiences
        :return:
        """
        if len(self._buffer) + len(exps) > self._buf_size:
            self._buffer[:(len(exps) + len(self._buffer) - self._buf_size)] = []
        self._buffer.extend(exps)

    def sample(self, size):
        """
        Sample size experiences from the buffer.
        :param size:
        :return:
        """
        return random.sample(self._buffer, min(size, len(self._buffer)))








