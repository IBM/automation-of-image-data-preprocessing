import os

import tensorflow as tf
import warnings

from autodp.runner.nn.nn_runner import NNRunner
from autodp.utils.misc import get_class
from autodp import cf
from autodp.network.graph.nn.nn_graph import NNGraph
from autodp.utils.tf_utils import copy_network


class CLRunner(NNRunner):
    """This class implements a way to allow to continue learning after using results of preprocessing."""
    def __init__(self, preprocess=True):
        if preprocess:
            self._preprocess_data()

    def _preprocess_data(self):
        """Method to do preprocessing."""
        with tf.Graph().as_default(), tf.Session() as sess:
            # Initialize rl agent
            rl_agent_class = get_class(cf.rl_agent)
            rl_agent = rl_agent_class()

            # Initialize data readers
            reader_class = get_class(cf.reader)
            train_reader = reader_class(cf.train_path)
            valid_reader = reader_class(cf.valid_path)
            test_reader = reader_class(cf.test_path)
            readers = [train_reader, valid_reader, test_reader]
            locations = ["pp_train", "pp_valid", "pp_test"]

            # Do initialization
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # Load the model
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(cf.save_model + "/rl")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise ("Model not exist.")

            # Do preprocessing
            rl_agent.preprocess(sess, readers, locations)

    def train_model(self, verbose=True):
        """Main method for training."""
        with tf.Graph().as_default(), tf.Session() as sess:
            # Initialize a data reader for train
            reader_class = get_class(cf.reader)
            train_reader = reader_class(os.path.join(cf.prep_path, "pp_train"), cf.num_epoch)

            # Initialize a data reader for validation
            valid_reader = reader_class(os.path.join(cf.prep_path, "pp_valid"))

            # Build graph and do initialization
            train_nng = NNGraph(net_arch=cf.nn_arch, loss_func=cf.nn_loss, name="main_graph")
            valid_nng = NNGraph(net_arch=cf.nn_arch, loss_func=cf.nn_loss, name="valid_nng")

            # Copy network between train and validation
            update_ops = copy_network(tf.trainable_variables())

            # Config trainable variables
            common_vars = tf.global_variables()[: 2 * (len(cf.kernel_size) + len(cf.fc_size))]

            # Do initialization
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # Load existing model to continue training
            saver = tf.train.Saver(common_vars)
            ckpt = tf.train.get_checkpoint_state(cf.save_model + "/rl")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                warnings.warn("Model not exist, train a new model now.")

            # Start to train
            best_valid = self._train(sess, train_reader, valid_reader, train_nng.get_train_step, train_nng.get_error,
                                     valid_nng.get_error, train_nng.get_instance, train_nng.get_label,
                                     valid_nng.get_instance, valid_nng.get_label, train_nng.get_phase_train,
                                     train_nng.get_keep_prob, update_ops, verbose)
        return best_valid
