"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import abc
from importlib import __import__

from sentana.config.cf_container import Config as cf
from sentana.network.loss.loss import Loss


class BaseGraph(metaclass=abc.ABCMeta):
    """
    This is a base class that helps build the tensorflow graph.
    """
    def __init__(self, train=True):
        """
        Initialization.
        :param train: train or test
        """
        if train:
            self._train_step, self._obj_func = self._build_model_for_train()
        else:
            self._pred_output, self._obj_func = self._build_model_for_test()

    @staticmethod
    def _declare_inputs(dr):
        """
        Declare inputs of a tensorflow graph.
        :param: dr
        :return: input tensors
        """
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
        loss = Loss(preds=preds, trues=trues)
        total_loss = loss.compute_loss(cf.loss_func)

        return total_loss

    @abc.abstractmethod
    def _train(obj_func):
        """
        Define the train operator.
        :param obj_func: the function to be optimized
        :return: the train operator
        """

    @abc.abstractmethod
    def _build_model_for_train(self):
        """
        Build the total graph for training.
        """

    @abc.abstractmethod
    def _build_model_for_test(self):
        """
        Rebuild the model for test.
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