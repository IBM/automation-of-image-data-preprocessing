"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import sys

import tensorflow as tf
import numpy as np
import warnings
import pandas as pd

from autodp.runner.base_runner import BaseRunner
from autodp.utils.misc import clear_model_dir
from autodp.network.graph.nn.nn_graph import NNGraph
from autodp.config.cf_container import Config as cf
from autodp.utils.misc import get_class
from autodp.utils.tf_utils import copy_network
from autodp.utils.tf_utils import update_network


@BaseRunner.register
class NNRunner(BaseRunner):
    """
    Main class to train/test a CNN model.
    """
    def __init__(self):
        """
        Temporarily do nothing for initialization.
        """
        pass

    @staticmethod
    def _run_train_step(sess, train_op, error_op, fd):
        """
        Train a batch of data.
        :param sess:
        :param train_op:
        :param error_op:
        :param fd:
        :return: error
        """
        [_, err] = sess.run([train_op, error_op], feed_dict=fd)

        return err

    @staticmethod
    def _run_valid_step(sess, error_op, fd):
        """
        Validate a batch of data.
        :param sess:
        :param error_op:
        :param fd:
        :return: error
        """
        err = sess.run(error_op, feed_dict=fd)

        return err

    @staticmethod
    def _run_test_step(sess, pred, id, prob, fd):
        """
        Test a batch of data.
        :param sess:
        :param pred:
        :param id:
        :param prob:
        :param fd:
        :return:
        """
        [p, i, pr] = sess.run([pred, id, prob], feed_dict=fd)

        return p, i, pr

    def _train(self, sess, train_reader, valid_reader, train_op, train_err_op,
               valid_err_op, train_image_op, train_label_op, valid_image_op,
               valid_label_op, update_ops):
        """
        Do training with a sqreader.
        :param sess:
        :param train_reader:
        :param valid_reader:
        :param train_op:
        :param train_err_op:
        :param valid_err_op:
        :param train_image_op:
        :param train_label_op:
        :param valid_image_op:
        :param valid_label_op:
        :param update_ops:
        :return:
        """
        step, err_list = 0, []
        early_stop, best_valid = 0, sys.maxsize
        for (images, labels) in train_reader.get_batch(sess=sess):
            # Do training
            fd = {train_image_op: images, train_label_op: labels}
            err = self._run_train_step(sess, train_op, train_err_op, fd)
            err_list.append(err)
            step += 1

            # Do validation
            if step % cf.valid_step == 0:
                update_network(sess, update_ops)
                valid_err = self._valid(sess, valid_reader, valid_err_op,
                                        valid_image_op, valid_label_op)

                if valid_err < best_valid:
                    best_valid = valid_err
                    early_stop = 0
                    print("Step %d has average error: %g and reduces "
                          "validation error: %g" % (step, np.mean(err_list),
                                                    best_valid))

                    # Save model
                    clear_model_dir(cf.save_model + "/nn")
                    saver = tf.train.Saver(tf.global_variables())
                    saver.save(sess, cf.save_model + "/nn/model")

                else:
                    print("Step %d has average error: %g" % (
                        step, np.mean(err_list)))
                    early_stop += 1

                err_list = []
                if early_stop > 3:
                    print("Exit due to early stopping")
                    break

    def _valid(self, sess, reader, valid_err_op, image_op, label_op):
        """
        Do validation with a sqreader.
        :param sess:
        :param reader:
        :param valid_err_op:
        :param image_op:
        :param label_op:
        :return:
        """
        err_list = []
        for (images, labels) in reader.get_batch(sess=sess):
            fd = {image_op: images, label_op: labels}
            err = self._run_valid_step(sess, valid_err_op, fd)
            err_list.append(err)

        valid_err = np.mean(err_list)

        return valid_err

    def _test(self, sess, reader, pred, id, image_op, label_op, prob):
        """
        Do testing with a sqreader.
        :param sess:
        :param reader:
        :param pred:
        :param id: either true labels or image ids
        :param image_op:
        :param label_op:
        :param prob:
        :return: accuracy (only useful if ids are true labels
        """
        p_list, i_list, pr_list = [], [], []
        for (images, labels) in reader.get_batch(sess=sess):
            fd = {image_op: images, label_op: labels}
            p, i, pr = self._run_test_step(sess, pred, id, prob, fd)
            p_list.extend(p)
            i_list.extend(i)
            pr_list.extend(pr)

        # Support kaggle output
        df = pd.concat([pd.DataFrame({"id": i_list, "label": p_list}),
                        pd.DataFrame(np.array(pr_list))], axis=1)
        df.to_csv(cf.result_path + "/result.csv", header=True, sep=",",
                  index=False)

        tmp = np.abs(np.array(i_list) - np.array(p_list))
        bool_tmp = [bool(t) for t in tmp]
        accuracy = 1 - sum(bool_tmp) / float(len(i_list))

        return accuracy

    def train_model(self, cont=False):
        """
        Main method for training.
        :param cont: new training or continue with current training
        :return:
        """
        with tf.Graph().as_default(), tf.Session() as sess:
            # Initialize a data reader for train
            reader_class = get_class(cf.reader)
            train_reader = reader_class(cf.train_path, cf.num_epoch)

            # Initialize a data reader for validation
            valid_reader = reader_class(cf.valid_path)

            # Build graph and do initialization
            train_nng = NNGraph(net_arch=cf.nn_arch, loss_func=cf.nn_loss,
                                name="main_graph")
            valid_nng = NNGraph(net_arch=cf.nn_arch, loss_func=cf.nn_loss,
                                name="valid_nng")

            # Copy network between train and validation
            update_ops = copy_network(tf.trainable_variables())

            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))

            # Load existing model to continue training
            if cont:
                # Load the model
                saver = tf.train.Saver(tf.global_variables())
                ckpt = tf.train.get_checkpoint_state(cf.save_model + "/nn")
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                else:
                    warnings.warn("Model not exist, train a new model now")

            # Start to train
            self._train(sess, train_reader, valid_reader,
                        train_nng.get_train_step, train_nng.get_error,
                        valid_nng.get_error, train_nng.get_instance,
                        train_nng.get_label, valid_nng.get_instance,
                        valid_nng.get_label, update_ops)

    def test_model(self, path=cf.test_path, fh=None):
        """
        Main method for testing.
        :param path:
        :param fh:
        :return:
        """
        with tf.Graph().as_default(), tf.Session() as sess:
            # Initialize a data reader
            reader_class = get_class(cf.reader)
            reader = reader_class(path)

            # Build graph and do initialization
            nng1 = NNGraph(net_arch=cf.nn_arch, loss_func=cf.nn_loss,
                           name="main_graph")
            nng2 = NNGraph(net_arch=cf.nn_arch, loss_func=cf.nn_loss,
                           name="valid_nng")

            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))

            # Load the model
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(cf.save_model + "/nn")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            else:
                raise IOError("Model not exist")

            # Start to test
            pred = tf.arg_max(tf.nn.softmax(nng2.get_pred), dimension=1)
            prob = tf.nn.softmax(nng2.get_pred)
            accuracy = self._test(sess, reader, pred, nng2.get_label,
                                  nng2.get_instance, nng2.get_label, prob)

        return accuracy

    # def _run_epoch(self, sess, graph, data_path, stage="train"):
    #     """
    #     Run 1 epoch.
    #     :param sess:
    #     :param graph:
    #     :param data_path:
    #     :param stage:
    #     :return:
    #     """
    #     # Store error so far
    #     err_list, pred_list, true_list = [], [], []
    #
    #     # Run epoch
    #     with bz2.BZ2File(data_path, "rb") as df:
    #         # Read data chunk by chunk
    #         while True:
    #             images, labels = self._get_batch(df, cf.batch_size)
    #             if len(images) == 0: break
    #
    #             # Batch is not empty
    #             if stage == "train":
    #                 err = self._run_train_step(sess, graph.get_train_step,
    #                     graph.get_error, graph.get_instances,
    #                     graph.get_targets, images, labels)
    #                 err_list.append(err)
    #
    #             else:
    #                 pred_op = tf.argmax(tf.nn.softmax(graph.get_preds), axis=1)
    #                 pred, err = self._run_test_step(sess, pred_op,
    #                     graph.get_error, graph.get_instances,
    #                     graph.get_targets, images, labels)
    #                 err_list.append(err)
    #                 pred_list.extend(pred)
    #                 true_list.extend(labels)
    #
    #         performance = -1
    #         if stage != "train":
    #             tmp = np.abs(np.array(true_list) - np.array(pred_list))
    #             bool_tmp = [bool(t) for t in tmp]
    #             performance = sum(bool_tmp) / len(true_list)
    #
    #     return performance, np.mean(err_list), pred_list, true_list










































