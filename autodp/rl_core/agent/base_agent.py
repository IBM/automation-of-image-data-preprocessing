"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import abc


class BaseAgent(metaclass=abc.ABCMeta):
    """
    This abstract class defines methods exposed by a reinforcement learning
    agent.
    """
    def __init__(self, cont):
        """
        Initialization.
        """
        if cont:
            self._load_specific_objects()

    def _load_specific_objects(self):
        """
        This method can be overwritten to initialize specific objects needed
        to continue learning.
        :return:
        """
        # Load exp
        # if os.path.isfile(os.path.dirname(cf.save_model) + "/exp.pkl"):
        #     exp_buf = from_pickle(os.path.dirname(
        #         cf.save_model) + "/exp.pkl")
        pass

    @abc.abstractmethod
    def setup_policy(self):
        """
        Build one or more networks to approximate value, action-value and
        policy functions.
        :return:
        """

    @abc.abstractmethod
    def train_policy(self):
        """
        Policy improvement and evaluation.
        :return:
        """

    @abc.abstractmethod
    def predict(self):
        """
        Apply the policy to predict image classification.
        :return:
        """
















































