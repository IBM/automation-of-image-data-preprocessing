"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os
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
    def _run_train_step_with_tfreader(sess, train_op, error_op):
        """
        Train a batch of data.
        :param sess:
        :param train_op:
        :param error_op:
        :return: error
        """
        [_, err] = sess.run([train_op, error_op])

        return err

    @staticmethod
    def _run_valid_step_with_tfreader(sess, error_op):
        """
        Validate a batch of data.
        :param sess:
        :param error_op:
        :return: error
        """
        err = sess.run(error_op)

        return err

    @staticmethod
    def _run_test_step_with_tfreader(sess, preds, ids):
        """
        Test a batch of data.
        :param sess:
        :param preds:
        :param ids:
        :return:
        """
        [p, i] = sess.run([preds, ids])

        return p, i

    @staticmethod
    def _run_train_step_with_sqreader(sess, train_op, error_op, fd):
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
    def _run_valid_step_with_sqreader(sess, error_op, fd):
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
    def _run_test_step_with_sqreader(sess, preds, ids, fd):
        """
        Test a batch of data.
        :param sess:
        :param preds:
        :param ids:
        :param fd:
        :return:
        """
        [p, i] = sess.run([preds, ids], feed_dict=fd)

        return p, i

    def _train_with_tfreader(self, sess, train_op, train_err_op, valid_err_op):
        """
        Do training with a tfreader.
        :param sess:
        :param train_op:
        :param train_err_op:
        :param valid_err_op:
        :return:
        """
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            step, err_list = 0, []
            early_stop, best_valid = 0, sys.maxsize
            while not coord.should_stop():
                # Do training
                err = self._run_train_step_with_tfreader(sess, train_op,
                                                         train_err_op)
                err_list.append(err)
                step += 1

                # Do validation
                if step % cf.valid_step == 0:
                    copy_network(sess, tf.trainable_variables())
                    valid_err = self._valid_with_tfreader(sess, valid_err_op)

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

        except tf.errors.OutOfRangeError:
            print("Finish training successfully")

        finally:
            coord.request_stop()
            coord.join(threads)

    def _valid_with_tfreader(self, sess, valid_err_op):
        """
        Do validation with a tfreader.
        :param sess:
        :param valid_err_op:
        :return:
        """
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            err_list = []
            while not coord.should_stop():
                err = self._run_valid_step_with_tfreader(sess, valid_err_op)
                err_list.append(err)

        except tf.errors.OutOfRangeError:
            valid_err = np.mean(err_list)

        finally:
            coord.request_stop()
            coord.join(threads)

        return valid_err

    def _test_with_tfreader(self, sess, preds, ids):
        """
        Do testing with a tfreader.
        :param sess:
        :param preds:
        :param ids: either true labels or image ids
        :return: accuracy (only useful if ids are true labels
        """
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            p_list, i_list = [], []
            while not coord.should_stop():
                p, i = self._run_test_step_with_tfreader(sess, preds, ids)
                p_list.extend(p)
                i_list.extend(i)

        except tf.errors.OutOfRangeError:
            df = pd.DataFrame({"id": i_list, "label": p_list})
            df.to_csv(cf.result_path, header=True, sep=",", index=False)

            tmp = np.abs(np.array(i_list) - np.array(p_list))
            bool_tmp = [bool(t) for t in tmp]
            accuracy = 1 - sum(bool_tmp) / float(len(i_list))

        finally:
            coord.request_stop()
            coord.join(threads)

        return accuracy

    def _train_with_sqreader(self, sess, train_reader, valid_reader, train_op,
                             train_err_op, valid_err_op, train_image_op,
                             train_label_op, valid_image_op, valid_label_op):
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
        :return:
        """
        step, err_list = 0, []
        early_stop, best_valid = 0, sys.maxsize
        for (images, labels) in train_reader.get_batch():
            # Do training
            fd = {train_image_op: images, train_label_op: labels}
            err = self._run_train_step_with_sqreader(sess, train_op,
                                                     train_err_op, fd)
            err_list.append(err)
            step += 1

            # Do validation
            if step % cf.valid_step == 0:
                copy_network(sess, tf.trainable_variables())
                valid_err = self._valid_with_sqreader(sess, valid_reader,
                                                      valid_err_op,
                                                      valid_image_op,
                                                      valid_label_op)

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

    def _valid_with_sqreader(self, sess, reader, valid_err_op,
                             image_op, label_op):
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
        for (images, labels) in reader.get_batch():
            fd = {image_op: images, label_op: labels}
            err = self._run_valid_step_with_sqreader(sess, valid_err_op, fd)
            err_list.append(err)

        valid_err = np.mean(err_list)

        return valid_err

    def _test_with_sqreader(self, sess, reader, preds, ids,
                            image_op, label_op):
        """
        Do testing with a sqreader.
        :param sess:
        :param reader:
        :param preds:
        :param ids: either true labels or image ids
        :param image_op:
        :param label_op:
        :return: accuracy (only useful if ids are true labels
        """
        p_list, i_list = [], []
        for (images, labels) in reader.get_batch():
            fd = {image_op: images, label_op: labels}
            p, i = self._run_test_step_with_sqreader(sess, preds, ids, fd)
            p_list.extend(p)
            i_list.extend(i)

        df = pd.DataFrame({"id": i_list, "label": p_list})
        df.to_csv(cf.result_path, header=True, sep=",", index=False)

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
            if cf.reader.split(".")[-1] == "TFReader":
                train_nng = NNGraph(net_arch=cf.nn_arch, loss_func=cf.nn_loss,
                                    name="train_nng", tfreader=train_reader)
                valid_nng = NNGraph(net_arch=cf.nn_arch, loss_func=cf.nn_loss,
                                    name="valid_nng", tfreader=valid_reader)

            else:
                train_nng = NNGraph(net_arch=cf.nn_arch, loss_func=cf.nn_loss,
                                    name="train_nng")
                valid_nng = NNGraph(net_arch=cf.nn_arch, loss_func=cf.nn_loss,
                                    name="valid_nng")

            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))

            # Load existing model to continue training
            if cont:
                # Load the model
                saver = tf.train.Saver(tf.global_variables())
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(
                    cf.save_model + "/nn/model"))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                else:
                    warnings.warn("Model not exist, train a new model now")

            # Start to train
            if cf.reader.split(".")[-1] == "TFReader":
                self._train_with_tfreader(sess, train_nng.get_train_step,
                                          train_nng.get_error,
                                          valid_nng.get_error)

            else:
                self._train_with_sqreader(sess, train_reader, valid_reader,
                                          train_nng.get_train_step,
                                          train_nng.get_error,
                                          valid_nng.get_error,
                                          train_nng.get_instance,
                                          train_nng.get_label,
                                          valid_nng.get_instance,
                                          valid_nng.get_label)

    def test_model(self):
        """
        Main method for testing.
        :return:
        """
        with tf.Graph().as_default(), tf.Session() as sess:
            # Initialize a data reader
            reader_class = get_class(cf.reader)
            reader = reader_class(cf.test_path)

            # Build graph and do initialization
            if cf.reader.split(".")[-1] == "TFReader":
                nng1 = NNGraph(net_arch=cf.nn_arch, loss_func=cf.nn_loss,
                               name="nng1", tfreader=reader)
                nng2 = NNGraph(net_arch=cf.nn_arch, loss_func=cf.nn_loss,
                               name="nng2", tfreader=reader)

            else:
                nng1 = NNGraph(net_arch=cf.nn_arch, loss_func=cf.nn_loss,
                               name="nng1")
                nng2 = NNGraph(net_arch=cf.nn_arch, loss_func=cf.nn_loss,
                               name="nng2")

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
            if cf.reader.split(".")[-1] == "TFReader":
                accuracy = self._test_with_tfreader(sess, nng1.get_pred,
                                                    nng1.get_label)

            else:
                accuracy = self._test_with_sqreader(sess, reader,
                                                    nng1.get_pred,
                                                    nng1.get_label,
                                                    nng1.get_instance,
                                                    nng1.get_label)

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










































