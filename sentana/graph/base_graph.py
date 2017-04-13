"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import abc
from importlib import __import__
import tensorflow as tf

from sentana.config.cf_container import Config as cf
from sentana.network.loss.loss import Loss


class BaseGraph(metaclass=abc.ABCMeta):
    """
    This is a base class that helps build the tensorflow graph.
    """
    @staticmethod
    def _declare_inputs():
        """
        Declare inputs of a tensorflow graph.
        :return: input tensors
        """
        if cf.read_type == "feeddict":
            instances = tf.placeholder(dtype=tf.float32, name="instance",
                            shape=[None, cf.ima_height, cf.ima_width, 3])
            targets = tf.placeholder(dtype=tf.float32, shape=[None],
                                    name="target")
        elif cf.read_type == "pipeline":
            # Todo
            pass

        else:
            raise NotImplementedError("Data reader type %s not "
                                      "supported yet!" % cf.read_type)

        return instances, targets

    @staticmethod
    def _inference(instances):
        """
        Main function for building a tensorflow graph.
        :param instance:
        :return:
        """
        sub_mods = cf.net_arch.split(sep=".")
        module = __import__("".join(sub_mods[:-1]), fromlist=sub_mods[-1])
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
