"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os
import sys

import tensorflow as tf
import numpy as np
import warnings
import pickle
import bz2

from autodp.utils.misc import clear_model_dir
from autodp.graph.seq_graph import SeqGraph
from autodp.config.cf_container import Config as cf


class ModelRunner(object):
    """
    Main class to train/test models.
    """
    def __init__(self):
        pass

    def _get_batch(self, data_file, batch_size):
        """
        Get a batch of images for training.
        :param data_file:
        :param batch_size:
        :return:
        """
        image_batch, label_batch = [], []
        for _ in range(batch_size):
            try:
                line = pickle.load(data_file)
                image_batch.append(line["img"])
                label_batch.append(line["label"])
            except EOFError:
                break

        return image_batch, label_batch

    @staticmethod
    def _run_train_step(sess, train_step, error, image_input, label_input,
                        image_batch, label_batch):
        """
        Train a batch of data.
        :param sess:
        :param train_step:
        :param error:
        :param image_input:
        :param label_input:
        :param image_batch:
        :param label_batch:
        :return: error
        """
        [_, err] = sess.run([train_step, error],
                            feed_dict={image_input: image_batch,
                                       label_input: label_batch})

        return err

    @staticmethod
    def _run_test_step(sess, preds, error, image_input, label_input,
                       image_batch, label_batch):
        """
        Test a batch of data.
        :param sess:
        :param preds:
        :param error:
        :param image_input:
        :param label_input:
        :param image_batch:
        :param label_batch:
        :return: error, predictions and targets
        """
        to_test = [preds, error]
        [pred, err] = sess.run(to_test, feed_dict={image_input: image_batch,
                                                   label_input: label_batch})

        return pred, err

    def _run_epoch(self, sess, graph, data_path, stage="train"):
        """
        Run 1 epoch.
        :param sess:
        :param graph:
        :param data_path:
        :param stage:
        :return:
        """
        # Store error so far
        err_list, pred_list, true_list = [], [], []

        # Run epoch
        with bz2.BZ2File(data_path, "rb") as df:
            # Read data chunk by chunk
            while True:
                images, labels = self._get_batch(df, cf.batch_size)
                if len(images) == 0: break

                # Batch is not empty
                if stage == "train":
                    err = self._run_train_step(sess, graph.get_train_step,
                        graph.get_error, graph.get_instances,
                        graph.get_targets, images, labels)
                    err_list.append(err)

                else:
                    pred_op = tf.argmax(tf.nn.softmax(graph.get_preds), axis=1)
                    pred, err = self._run_test_step(sess, pred_op,
                        graph.get_error, graph.get_instances,
                        graph.get_targets, images, labels)
                    err_list.append(err)
                    pred_list.extend(pred)
                    true_list.extend(labels)

            performance = -1
            if stage != "train":
                tmp = np.abs(np.array(true_list) - np.array(pred_list))
                bool_tmp = [bool(t) for t in tmp]
                performance = sum(bool_tmp) / len(true_list)

        return performance, np.mean(err_list), pred_list, true_list

    def train_model(self, cont=False):
        """
        Main method for training.
        :param cont: new training or continue with current training
        :return:
        """
        with tf.Graph().as_default(), tf.Session() as sess:
            sg = SeqGraph()
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))

            if cont:
                # Load the model
                saver = tf.train.Saver(tf.global_variables())
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(
                    cf.save_model))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    warnings.warn("Model not exist, train a new model now")

            # Start to train
            best_valid = sys.maxsize
            for epoch in range(cf.num_epoch):
                _, train_err, _, _ = self._run_epoch(sess, sg, cf.train_path)
                perf, valid_err, _, _ = self._run_epoch(sess, sg, cf.valid_path,
                                                        stage="valid")
                print("Epoch %d has training error %g, validation error %g "
                      "and performance error %g" % (epoch, train_err,
                                                    valid_err, perf))

                if valid_err < best_valid:
                    best_valid = valid_err
                    print("Best valid err is %g and performance error is %g"
                          % (best_valid, perf))

                    # Save model
                    clear_model_dir(os.path.dirname(cf.save_model))
                    saver = tf.train.Saver(tf.global_variables())
                    saver.save(sess, cf.save_model)

    def test_model(self):
        """
        Main method for testing.
        :return:
        """
        with tf.Graph().as_default(), tf.Session() as sess:
            sg = SeqGraph()
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))

            # Load the model
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(cf.save_model))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Model not exist")

            # Start to test
            perf, err, pred_list, true_list = self._run_epoch(sess, sg,
                cf.test_path, stage="test")
            print("Test and performance error are %g and %g" % (err, perf))

        return pred_list, true_list
