import abc

import tensorflow as tf

from autodp import cf
from autodp.utils.misc import get_class


class BaseGraph(metaclass=abc.ABCMeta):
    """This is a base class that helps build the tensorflow graph."""
    def __init__(self, net_arch, loss_func, name):
        self._net_arch = net_arch
        self._loss_func = loss_func
        self._name = name
        self._build_model()

    def _inference(self, instance):
        """Main function for building a tensorflow graph."""
        net_arch_class = get_class(self._net_arch)
        net_arch = net_arch_class(instance, self._phase_train, self._keep_prob, self._name)
        preds = net_arch.build_arch()
        return preds

    def _loss(self, preds, trues):
        """Define an objective function."""
        loss_class = get_class(self._loss_func)
        loss = loss_class(preds=preds, trues=trues)
        total_loss = loss.compute_loss()
        return total_loss

    @staticmethod
    def _train(obj_func):
        """Define the train operator."""
        # Create gradient steps
        tvars = tf.trainable_variables()
        grads = tf.gradients(obj_func, tvars)
        norm_grads, _ = tf.clip_by_global_norm(grads, cf.max_grad_norm)

        # Define a train step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(cf.learning_rate)
            train_step = optimizer.apply_gradients(zip(norm_grads, tvars))
        return train_step

    def _train_loss(self, preds, trues):
        """Define a loss function and a train operator at the same time."""
        self._obj_func = self._loss(preds, trues)
        self._train_step = self._train(self._obj_func)

    @abc.abstractmethod
    def _build_model(self):
        """Build the total graph."""

    @property
    def get_error(self):
        """Get train/test error."""
        return self._obj_func

    @property
    def get_train_step(self):
        """Get train operator."""
        return self._train_step

    @property
    def get_instance(self):
        """Get input instances."""
        return self._instance

    @property
    def get_label(self):
        """Get true output."""
        return self._target

    @property
    def get_pred(self):
        """Get prediction output."""
        return self._pred

    @property
    def get_phase_train(self):
        """Get the phase_train opt."""
        return self._phase_train

    @property
    def get_keep_prob(self):
        """Get the keep_prob opt."""
        return self._keep_prob
