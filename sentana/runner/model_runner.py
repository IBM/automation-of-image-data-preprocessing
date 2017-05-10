"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os

import tensorflow as tf
import numpy as np
import warnings

from sentana.utils.misc import clear_model_dir
from sentana.graph.seq_graph import SeqGraph
from sentana.config.cf_container import Config as cf


class ModelRunner(object):
    """
    Main class to train/test models.
    """
    def __init__(self):
        pass

    def _run_train_step(self):
        """
        Train a batch of data.
        :param instances:
        :param targets:
        :return: error
        """
        [_, err] = self._sess.run([self._train_step, self._error])

        return err

    def _run_test_step(self):
        """
        Test a batch of data.
        :param instances:
        :param targets:
        :return: error
        """
        to_test = [self._preds, self._trues, self._error]
        [preds, trues, err] = self._sess.run(to_test)

        return preds, trues, err

    def train_model(self, cont=False):
        """
        Main method for training.
        :param cont: new training or continue with current training
        :return:
        """
        with tf.Graph().as_default(), tf.Session() as self._sess:
            sg = SeqGraph(cf.train_path)
            self._train_step = sg.get_train_step
            self._error = sg.get_error
            self._sess.run(tf.group(tf.global_variables_initializer(),
                                    tf.local_variables_initializer()))

            if cont:
                # Load the model
                saver = tf.train.Saver(tf.global_variables())
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(
                    cf.save_model))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(self._sess, ckpt.model_checkpoint_path)
                else:
                    warnings.warn("Model not exist, train a new model now")

            # Start to train
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                step, err_list = 0, []
                while not coord.should_stop():
                    err = self._run_train_step()
                    err_list.append(err)

                    step += 1
                    if step % 1 == 0:
                        print ("Step %d has error: %g, average error: %g" % (
                            step, err, np.mean(err_list)))

            except tf.errors.OutOfRangeError:
                # Save model
                clear_model_dir(os.path.dirname(cf.save_model))
                saver = tf.train.Saver(tf.global_variables())
                saver.save(self._sess, cf.save_model)

            finally:
                coord.request_stop()

            coord.join(threads)
            self._sess.close()

    def test_model(self):
        """
        Main method for testing.
        :return:
        """
        with tf.Graph().as_default(), tf.Session() as self._sess:
            sg = SeqGraph(cf.test_path)
            self._error = sg.get_error
            self._preds = sg.get_preds
            self._trues = sg.get_targets
            self._sess.run(tf.group(tf.global_variables_initializer(),
                                    tf.local_variables_initializer()))

            # Load the model
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(cf.save_model))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self._sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Model not exist")

            # Start to test
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            pred_list, true_list, err_list = [], [], []
            try:
                while not coord.should_stop():
                    preds, trues, err = self._run_test_step()
                    print(preds)
                    print(trues)
                    pred_list += preds
                    true_list += trues
                    err_list.append(err)

            except tf.errors.OutOfRangeError:
                print ("Test error is: %g" % np.mean(err_list))

            finally:
                coord.request_stop()

            coord.join(threads)
            self._sess.close()

        return pred_list, true_list
