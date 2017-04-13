"""
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os

import tensorflow as tf
import numpy as np
import warnings
from itertools import islice
import json

from onebm_afl.network.graph_builder import GraphBuilder
from onebm_afl.config.cf_container import Config as cf
from onebm_afl.utils.misc import clear_model_dir


class ModelRunner(object):
    """
    Main class to train/test models.
    """
    def __init__(self):
        pass

    def _run_train_step(self, instances, targets):
        """
        Train a batch of data.
        :param input_batch:
        :param output_batch:
        :return: error
        """
        [_, err] = self._sess.run([self._train_step, self._error],
                                feed_dict={self._instances: instances,
                                           self._targets: targets})

        return err

    def _get_batch(self, data_file):
        """
        Read data as batch per time.
        :param data_file:
        :return:
        """
        f_batch, l_batch, s_batch = [], [], []
        lines = islice(data_file, cf.batch_size)
        for line in lines:
            f_data = json.loads(line)["features"]
            label = json.loads(line)["label"]

            ft_batch, st_batch = [], []
            for i in range(len(f_data)):
                line, length = self._line_pad_or_cut(f_data[i])
                ft_batch.append(line)
                st_batch.append(length)

            f_batch.append(ft_batch)
            s_batch.append(st_batch)
            l_batch.append(label)

        f_batch = np.array(f_batch) / 5000.0
        l_batch = np.reshape(l_batch, [-1, cf.num_response])
        s_batch = np.array(s_batch)

        return f_batch, l_batch, s_batch

    def _run_epoch(self, data_file, train=True):
        """
        Run 1 epoch.
        :param data_file:
        :param train:
        :return:
        """
        # Store error so far
        errList, preds, trues = [], [], []

        # Read data chunk by chunk
        while True:
            input_batch, output_batch, length_batch = self._get_batch(data_file)
            if len(output_batch) == 0 :
                break
            else:
                # Batch is not empty
                if train:
                    err = self._run_train_step(input_batch, output_batch,
                                               length_batch)
                else:
                    pred_out, true_out, err = self._run_test_step(input_batch,
                                                           output_batch,
                                                           length_batch)
                    preds = np.concatenate((preds, pred_out))
                    trues = np.concatenate((trues, true_out))

                errList.append(err)

        return preds, trues, np.mean(errList)

    def train_model(self, cont=False):
        """
        Main method for training.
        :param cont: new training or continue with current training
        :return:
        """
        with tf.Graph().as_default(), tf.Session() as self._sess:
            gp = GraphBuilder(train=True)
            self._input_batch = gp.get_example
            self._output_batch = gp.get_target
            self._length_batch = gp.get_length
            self._train_step = gp.get_train_step
            self._error = gp.get_error
            self._sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))

            if cont:
                # Load the model
                saver = tf.train.Saver(tf.all_variables())
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(
                    cf.save_model))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(self._sess, ckpt.model_checkpoint_path)
                else:
                    warnings.warn("Model not exist! Train a new model now...")

            for epoch in range(cf.num_epoch):
                with open(cf.data_path, 'r') as data_file:
                    _, _, err = self._run_epoch(data_file, train=True)
                if epoch % 1 == 0:
                    print("Epoch %d has mean error: %g" % (epoch, err))

            # Save model
            clear_model_dir(os.path.dirname(cf.save_model))
            saver = tf.train.Saver(tf.global_variables())
            saver.save(self._sess, cf.save_model)

    def test_model(self):
        """
        Main method for testing.
        :return:
        """
        with tf.Graph().as_default(), tf.Session() as self._sess:
            gp = GraphBuilder(train=False)
            self._input_batch = gp.get_example
            self._output_batch = gp.get_target
            self._length_batch = gp.get_length
            self._pred_output = gp.get_pred_output
            self._error = gp.get_error
            self._sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))

            # Load the model
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(cf.save_model))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self._sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Model not exist!")

            with open(cf.test_path, 'r') as test_file:
                preds, trues, err = self._run_epoch(test_file, train=False)

            print ("Test error is: %g" % err)

        return preds, trues
