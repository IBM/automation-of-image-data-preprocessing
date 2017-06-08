"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os
import sys
from os.path import expanduser

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

    @staticmethod
    def _run_train_step(sess, train_step, error):
        """
        Train a batch of data.
        :param sess:
        :param train_step:
        :param error:
        :return: error
        """
        [_, err] = sess.run([train_step, error])

        return err

    @staticmethod
    def _run_test_step(sess, preds, trues, error):
        """
        Test a batch of data.
        :param sess:
        :param preds:
        :param trues:
        :param error:
        :return: error
        """
        to_test = [preds, trues, error]
        [pred, true, err] = sess.run(to_test)

        return pred, true, err

    def train_model(self, cont=False):
        """
        Main method for training.
        :param cont: new training or continue with current training
        :return:
        """
        with tf.Graph().as_default(), tf.Session() as sess:
            sg = SeqGraph(cf.train_path, cf.num_epoch)
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
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                step, err_list = 0, []
                early_stop, best_valid = 0, sys.maxsize
                while not coord.should_stop():
                    err = self._run_train_step(sess, sg.get_train_step,
                                               sg.get_error)
                    err_list.append(err)

                    step += 1
                    if step % 1 == 0:
                        print ("Step %d has error: %g, average error: %g" % (
                            step, err, np.mean(err_list)))

                    if step % cf.valid_step == 0:
                        van_path = os.path.join(expanduser("~"), ".sentana",
                                                "vanila", "tmp_model.ckpt")
                        clear_model_dir(os.path.dirname(van_path))
                        saver = tf.train.Saver(tf.global_variables())
                        saver.save(sess, van_path)

                        valid_err, _, _ = self.test_model(cf.valid_path,
                                                          van_path)
                        if valid_err < best_valid:
                            best_valid = valid_err
                            early_stop = 0
                            print("Best valid err is %g" % best_valid)

                            # Save model
                            clear_model_dir(os.path.dirname(cf.save_model))
                            saver = tf.train.Saver(tf.global_variables())
                            saver.save(sess, cf.save_model)
                        else:
                            early_stop += 1

                        if early_stop >= 3:
                            print("Exit due to early stopping")
                            break

            except tf.errors.OutOfRangeError:
                print("Finish training without early stopping")

            finally:
                coord.request_stop()

            coord.join(threads)
            sess.close()

    def test_model(self, data_path=cf.test_path, model_path=cf.save_model):
        """
        Main method for testing.
        :param data_path:
        :param model_path:
        :return:
        """
        with tf.Graph().as_default(), tf.Session() as sess:
            sg = SeqGraph(data_path)
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))

            # Load the model
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_path))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Model not exist")

            # Start to test
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            pred_list, true_list, err_list = [], [], []
            try:
                while not coord.should_stop():
                    preds, trues, err = self._run_test_step(
                        sess, tf.argmax(sg.get_preds, axis=1),
                        sg.get_targets, sg.get_error)
                    pred_list.extend(preds)
                    true_list.extend(trues)
                    err_list.append(err)

            except tf.errors.OutOfRangeError:
                print ("Test error is: %g" % np.mean(err_list))

            finally:
                coord.request_stop()

            coord.join(threads)
            sess.close()

        test_err = sum(np.abs(np.array(pred_list)-np.array(
            true_list)))/len(true_list)

        return test_err, pred_list, true_list
