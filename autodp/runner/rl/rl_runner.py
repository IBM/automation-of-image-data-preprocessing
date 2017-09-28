"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf
import warnings
import numpy as np

from autodp.runner.base_runner import BaseRunner
from autodp.utils.misc import get_class
from autodp.config.cf_container import Config as cf


@BaseRunner.register
class RLRunner(BaseRunner):
    """
    This class implements a runner for reinforcement learning.
    """
    def __init__(self):
        """
        Temporarily do nothing for initialization.
        """
        pass

    def train_model(self, cont=False):
        """
        Main method for training rl policy.
        :param cont:
        :return:
        """
        with tf.Graph().as_default(), tf.Session() as sess:
            # Initialize rl agent
            rl_agent_class = get_class(cf.rl_agent)
            rl_agent = rl_agent_class(cont)

            # Setup neural network functions
            rl_graph = rl_agent.setup_policy()

            # Do initialization
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))

            # Load current model if continue training
            if cont:
                # Load the model
                saver = tf.train.Saver(tf.global_variables())
                ckpt = tf.train.get_checkpoint_state(cf.save_model + "/rl")
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    warnings.warn("Model not exist, train a new model now")

            # Initialize a data reader for train
            reader_class = get_class(cf.reader)
            train_reader = reader_class(cf.train_path, cf.num_epoch)

            # Initialize a data reader for validation
            valid_reader = reader_class(cf.valid_path)

            # Training
            rl_agent.train_policy(sess, rl_graph, train_reader, valid_reader)

    def test_model(self, fh=None):
        """
        Main method for testing.
        :param fh:
        :return:
        """
        with tf.Graph().as_default(), tf.Session() as sess:
            # Initialize rl agent
            rl_agent_class = get_class(cf.rl_agent)
            rl_agent = rl_agent_class()

            # Setup neural network functions
            rl_graph = rl_agent.setup_policy()

            # Do initialization
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))

            # Load the model
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(cf.save_model + "/rl")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                warnings.warn("Model not exist, train a new model now")

            # Initialize a data reader
            reader_class = get_class(cf.reader)
            reader = reader_class(cf.valid_path, 1)

            # Actual test
            reward, predict, actual = rl_agent.predict(sess, rl_graph,
                                                       reader, fh)

            # Accuracy
            tmp = np.abs(np.array(actual)-np.array(predict))
            bool_tmp = [bool(t) for t in tmp]
            accuracy = 1-sum(bool_tmp)/float(len(actual))

        return accuracy, reward, predict, actual
















































