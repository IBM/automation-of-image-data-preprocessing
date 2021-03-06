import tensorflow as tf
import warnings
import numpy as np

from autodp.runner.base_runner import BaseRunner
from autodp.utils.misc import get_class
from autodp import cf


@BaseRunner.register
class RLRunner(BaseRunner):
    """This class implements a runner for reinforcement learning."""
    def train_model(self, verbose=True):
        """Main method for training rl policy."""
        with tf.Graph().as_default(), tf.Session() as sess:
            # Initialize rl agent
            rl_agent_class = get_class(cf.rl_agent)
            rl_agent = rl_agent_class()

            # Initialize a data reader for train
            reader_class = get_class(cf.reader)
            train_reader = reader_class(cf.train_path, cf.num_epoch)

            # Initialize a data reader for validation
            valid_reader = reader_class(cf.valid_path)

            # Do initialization
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # Training
            best_valid = rl_agent.train_policy(sess, train_reader, valid_reader, verbose)
        return best_valid

    def test_model(self, path=cf.test_path):
        """Main method for testing."""
        with tf.Graph().as_default(), tf.Session() as sess:
            # Initialize rl agent
            rl_agent_class = get_class(cf.rl_agent)
            rl_agent = rl_agent_class()

            # Initialize a data reader
            reader_class = get_class(cf.reader)
            reader = reader_class(path)

            # Do initialization
            sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))

            # Load the model
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(cf.save_model + "/rl")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                warnings.warn("Model not exist, train a new model now.")

            # Actual test
            reward, predict, actual, prob = rl_agent.predict(sess, reader)

            # Accuracy
            tmp = np.abs(np.array(actual)-np.array(predict))
            bool_tmp = [bool(t) for t in tmp]
            accuracy = 1-sum(bool_tmp)/float(len(actual))
        return accuracy, reward, predict, actual
