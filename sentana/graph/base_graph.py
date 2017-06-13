"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import abc

import tensorflow as tf

from sentana.config.cf_container import Config as cf


class BaseGraph(metaclass=abc.ABCMeta):
    """
    This is a base class that helps build the tensorflow graph.
    """
    def __init__(self):
        """
        Initialization of building a graph.
        :param data_path:
        :param num_epoch:
        """
        self._build_model()

    @staticmethod
    def _declare_inputs(data_reader):
        """
        Declare inputs of a tensorflow graph.
        :param data_reader:
        :return: input tensors
        """
        input_tensors = data_reader.get_batch()

        return input_tensors

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

    def _train(self, obj_func):
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

    @abc.abstractmethod
    def _build_model(self):
        """
        Build the total graph.
        :param data_path:
        :param
        :return:
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

    @property
    def get_instances(self):
        """
        Get input instances.
        :return:
        """
        return self._instances
