"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
from autodp.rl_core.agent.base_agent import BaseAgent


class VanilaReinforce(BaseAgent):
    """
    This class implements the vanila version of the REINFORCE policy.
    """
    def __init__(self):
        """
        Initialization, call to father's init.
        """
        super().__init__()

    def _setup_policy(self):


    def load_specific_objects(self):
        """
        Do nothing.
        :return:
        """
        pass

    def save_specific_objects(self):
        """
        Do nothing.
        :return:
        """
        pass

    def train_policy(self, sess, train_reader, valid_reader, verbose):
        """
        Policy improvement and evaluation.
        :param sess:
        :param train_reader:
        :param valid_reader:
        :param verbose:
        :return:
        """
