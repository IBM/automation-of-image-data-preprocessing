"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import abc
from importlib import __import__

from sentana.config.cf_container import Config as cf
from sentana.reader.data_reader import DataReader


class BaseGraph(metaclass=abc.ABCMeta):
    """
    This is a base class that helps build the tensorflow graph.
    """
    def __init__(self, data_path):
        """
        Initialization of building a graph.
        :param data_path:
        """
        self._build_model(data_path)

    @staticmethod
    def _declare_inputs(data_path):
        """
        Declare inputs of a tensorflow graph.
        :param data_path:
        :return: input tensors
        """
        dr = DataReader(data_path)
        instances, targets = dr.get_batch()

        return instances, targets

    @staticmethod
    def _inference(instances):
        """
        Main function for building a tensorflow graph.
        :param instance:
        :return:
        """
        sub_mods = cf.net_arch.split(sep=".")
        module = __import__(".".join(sub_mods[:-1]), fromlist=sub_mods[-1])
        net_arch_name = getattr(module, sub_mods[-1])
        net_arch = net_arch_name(instances)
        preds = net_arch.build_arch()

        return preds

    @staticmethod
    def _loss(preds, trues):
        """
        Define an objective function.
        :param preds: predicted values
        :param trues: target values
        :return: the loss
        """
        module = __import__("".join(["sentana.network.loss.", cf.loss_func]),
                            fromlist=cf.loss_func.title().replace("_", ""))
        loss_class = getattr(module, cf.loss_func.title().replace("_", ""))

        loss = loss_class(preds=preds, trues=trues)
        total_loss = loss.compute_loss()

        return total_loss

    @abc.abstractmethod
    def _train(obj_func):
        """
        Define the train operator.
        :param obj_func: the function to be optimized
        :return: the train operator
        """

    @abc.abstractmethod
    def _build_model(self):
        """
        Build the total graph.
        """

    @property
    def get_preds(self):
        """
        Get prediction output.
        :return:
        """
        return self._preds

    @property
    def get_error(self):
        """
        Get train/test error.
        :return:
        """
        return self._obj_func

    @property
    def get_train_step(self):
        """
        Get train operator.
        :return:
        """
        return self._train_step

    @property
    def get_targets(self):
        """
        Get true output.
        :return:
        """
        return self._targets