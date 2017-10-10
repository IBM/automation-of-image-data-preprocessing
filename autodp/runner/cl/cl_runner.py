"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf

from autodp.runner.nn.nn_runner import NNRunner
from autodp.utils.misc import get_class
from autodp.config.cf_container import Config as cf


class CLRunner(NNRunner):
    """
    This class implements a way to allow to continue learning after using
    results of preprocessing.
    """
    def __init__(self):
        """
        Initialize by preprocessing image data.
        """
        self._preprocess_data()

    def _preprocess_data(self):
        """
        Method to do preprocessing.
        :return:
        """
        with tf.Graph().as_default(), tf.Session() as sess:
            # Initialize rl agent
            rl_agent_class = get_class(cf.rl_agent)
            rl_agent = rl_agent_class()

            # Initialize a data reader
            reader_class = get_class(cf.reader)
            reader = reader_class(cf.test_path, 1)

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

    def train_model(self):


