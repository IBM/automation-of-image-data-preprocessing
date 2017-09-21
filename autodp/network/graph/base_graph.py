"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import abc

import tensorflow as tf

from autodp.config.cf_container import Config as cf


class BaseGraph(metaclass=abc.ABCMeta):
    """
    This is a base class that helps build the tensorflow graph.
    """
    def __init__(self, net_arch, loss_func):
        """
        Initialization of building a graph.
        :param data_path:
        :param num_epoch:
        """
        self._net_arch = net_arch
        self._loss_func = loss_func
        self._build_model()

    def _inference(self, instance):
        """
        Main function for building a tensorflow graph.
        :param instance:
        :return:
        """
        sub_mods = self._net_arch.split(sep=".")
        module = __import__(".".join(sub_mods[:-1]), fromlist=sub_mods[-1])
        net_arch_name = getattr(module, sub_mods[-1])
        net_arch = net_arch_name(instance)
        preds = net_arch.build_arch()

        return preds

    def _loss(self, preds, trues):
        """
        Define an objective function.
        :param preds: predicted values
        :param trues: target values
        :return: the loss tensor
        """
        module = __import__("".join(["autodp.network.loss.", self._loss_func]),
                            fromlist=self._loss_func.title().replace("_", ""))
        loss_class = getattr(module, self._loss_func.title().replace("_", ""))

        loss = loss_class(preds=preds, trues=trues)
        total_loss = loss.compute_loss()

        return total_loss

    @staticmethod
    def _train(obj_func):
        """
        Define the train operator.
        :param obj_func: the function to be optimized
        :return: the train operator
        """
        # Create gradient steps
        tvars = tf.trainable_variables()
        grads = tf.gradients(obj_func, tvars)
        norm_grads, _ = tf.clip_by_global_norm(grads, cf.max_grad_norm)

        # Define a train step
        optimizer = tf.train.AdamOptimizer(cf.learning_rate)
        train_step = optimizer.apply_gradients(zip(norm_grads, tvars))

        return train_step

    def _train_loss(self, preds, trues):
        """
        Define a loss function and a train operator at the same time.
        :param preds:
        :param trues:
        :return:
        """
        self._obj_func = self._loss(preds, trues)
        self._train_step = self._train(self._obj_func)

    @abc.abstractmethod
    def _build_model(self, tfreader):
        """
        Build the total graph.
        :param tfreader:
        :return:
        """

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
    def get_instance(self):
        """
        Get input instances.
        :return:
        """
        return self._instance

























