"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import abc


class BaseArch(metaclass=abc.ABCMeta):
    """
    A base class to build a concrete architecture.
    """
    def __init__(self, instance):
        """
        Initialize by storing the input instance.
        :param instance:
        """
        self._instance = instance

    @abc.abstractmethod
    def build_arch(self):
        """
        Build a concrete architecture.
        :return:
        """