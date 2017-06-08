"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf

from sentana.graph.base_graph import BaseGraph
from sentana.config.cf_container import Config as cf
from sentana.reader.ril_reader import RILReader


@BaseGraph.register
class RILGraph(BaseGraph):
    """
    This class implements a sequential tensorflow graph for reinforcement
    learning.
    """
    def __init__(self, data_path=None):
        """
        Initialization of building a graph.
        :param data_path:
        """
        super().__init__(data_path, None)

    def _build_model(self, data_path, num_epoch):
        """
        Build the total graph.
        :param data_path:
        :param num_epoch:
        :return:
        """
        if data_path is not None:
            rr = RILReader(data_path)
            (self._instances, self._actions, self._targets) = \
                self._declare_inputs(rr)

        else:
            self._instances = tf.placeholder(tf.float32,
                shape=[None, cf.ima_height, cf.ima_width, 3])
            self._actions = tf.placeholder(tf.int32, shape=[None])
            self._targets = tf.placeholder(tf.float32, shape=[None])

        (self._qout, self._aout, self._qmax) = self._inference(self._instances)
        action_onehot = tf.one_hot(self._actions, depth=cf.num_action, axis=-1,
                                   dtype=tf.float32)
        self._preds = tf.reduce_sum(tf.multiply(
            self._qout, action_onehot), axis=1)
        self._obj_func = self._loss(self._preds, self._targets)
        self._train_step = self._train(self._obj_func)

    @property
    def get_next_actions(self):
        """
        Get best action for next step.
        :return:
        """
        return self._aout

    @property
    def get_actions(self):
        """
        Get action in.
        :return:
        """
        return self._actions

    @property
    def get_qmax(self):
        """
        Get max Q value.
        :return:
        """
        return self._qmax

    @property
    def get_qout(self):
        """
        Get Q values.
        :return:
        """
        return self._qout

