import sys

import tensorflow as tf
import numpy as np

from autodp.runner.base_runner import BaseRunner
from autodp.utils.misc import clear_model_dir, get_class
from autodp.network.graph.nn.nn_graph import NNGraph
from autodp import cf
from autodp.utils.tf_utils import copy_network, update_network


@BaseRunner.register
class NNRunner(BaseRunner):
    """Main class to train/test a CNN model."""
    @staticmethod
    def _run_train_step(sess, train_op, error_op, fd):
        [_, err] = sess.run([train_op, error_op], feed_dict=fd)
        return err

    @staticmethod
    def _run_valid_step(sess, error_op, fd):
        return sess.run(error_op, feed_dict=fd)

    @staticmethod
    def _run_test_step(sess, pred, id, prob, fd):
        return sess.run([pred, id, prob], feed_dict=fd)

    def _train(self, sess, train_reader, valid_reader, train_op, train_err_op, valid_err_op, train_image_op,
               train_label_op, valid_image_op, valid_label_op, phase_train, keep_prob, update_ops, verbose):
        step, err_list = 0, []
        early_stop, best_valid = 0, sys.maxsize
        for (images, labels) in train_reader.get_batch(sess=sess):
            # Do training
            fd = {train_image_op: images, train_label_op: labels, phase_train: True, keep_prob: cf.keep_prob}
            err = self._run_train_step(sess, train_op, train_err_op, fd)
            err_list.append(err)
            step += 1

            # Do validation
            if step % cf.valid_step == 0:
                update_network(sess, update_ops)
                valid_err = self._valid(sess, valid_reader, valid_err_op, valid_image_op, valid_label_op)

                if valid_err < best_valid:
                    best_valid = valid_err

                    if verbose:
                        print("Step %d has err %g and reduces val err %g" % (step, np.mean(err_list), best_valid))

                    # Save model
                    clear_model_dir(cf.save_model + "/nn")
                    saver = tf.train.Saver(tf.global_variables())
                    saver.save(sess, cf.save_model + "/nn/model")

                err_list = []
        return best_valid

    def _valid(self, sess, reader, valid_err_op, image_op, label_op):
        err_list = []
        for (images, labels) in reader.get_batch(sess=sess):
            fd = {image_op: images, label_op: labels}
            err = self._run_valid_step(sess, valid_err_op, fd)
            err_list.append(err)
        return np.mean(err_list)

    def _test(self, sess, reader, pred, id, image_op, label_op, prob):
        p_list, i_list, pr_list = [], [], []
        for (images, labels) in reader.get_batch(sess=sess):
            fd = {image_op: images, label_op: labels}
            p, i, pr = self._run_test_step(sess, pred, id, prob, fd)
            p_list.extend(p)
            i_list.extend(i)
            pr_list.extend(pr)

        tmp = np.abs(np.array(i_list) - np.array(p_list))
        bool_tmp = [bool(t) for t in tmp]
        accuracy = 1 - sum(bool_tmp) / float(len(i_list))
        return accuracy

    def train_model(self, verbose=True):
        """Main method for training."""
        with tf.Graph().as_default(), tf.Session() as sess:
            # Initialize a data reader for train
            reader_class = get_class(cf.reader)
            train_reader = reader_class(cf.train_path, cf.num_epoch)

            # Initialize a data reader for validation
            valid_reader = reader_class(cf.valid_path)

            # Build graph and do initialization
            train_nng = NNGraph(net_arch=cf.nn_arch, loss_func=cf.nn_loss, name="main_graph")
            valid_nng = NNGraph(net_arch=cf.nn_arch, loss_func=cf.nn_loss, name="valid_nng")

            # Copy network between train and validation
            update_ops = copy_network(tf.trainable_variables())
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # Start to train
            best_valid = self._train(sess, train_reader, valid_reader, train_nng.get_train_step, train_nng.get_error,
                                     valid_nng.get_error, train_nng.get_instance, train_nng.get_label,
                                     valid_nng.get_instance, valid_nng.get_label, train_nng.get_phase_train,
                                     train_nng.get_keep_prob, update_ops, verbose)
        return best_valid

    def test_model(self, path=cf.test_path):
        """Main method for testing."""
        with tf.Graph().as_default(), tf.Session() as sess:
            # Initialize a data reader
            reader_class = get_class(cf.reader)
            reader = reader_class(path)

            # Build graph and do initialization
            nng1 = NNGraph(net_arch=cf.nn_arch, loss_func=cf.nn_loss, name="main_graph")
            nng2 = NNGraph(net_arch=cf.nn_arch, loss_func=cf.nn_loss, name="valid_nng")

            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # Load the model
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(cf.save_model + "/nn")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Model not exist.")

            # Start to test
            pred = tf.arg_max(tf.nn.softmax(nng2.get_pred), dimension=1)
            prob = tf.nn.softmax(nng2.get_pred)
            accuracy = self._test(sess, reader, pred, nng2.get_label, nng2.get_instance, nng2.get_label, prob)
        return accuracy
