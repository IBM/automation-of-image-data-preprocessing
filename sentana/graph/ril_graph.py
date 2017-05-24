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
        super().__init__(data_path)

    def _build_model(self, data_path):
        """
        Build the total graph.
        :param data_path:
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

        (q_out, self._action_out, self._qmax) = self._inference(self._instances)
        action_onehot = tf.one_hot(self._actions, depth=cf.num_action, axis=-1,
                                   dtype=tf.float32)
        self._preds = tf.reduce_sum(tf.multiply(q_out, action_onehot), axis=1)
        self._obj_func = self._loss(self._preds, self._targets)
        self._train_step = self._train(self._obj_func)
