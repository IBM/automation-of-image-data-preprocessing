"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os
from os.path import expanduser
import sys

import tensorflow as tf
import warnings
from itertools import compress
import numpy as np
import bz2
import pickle
import cv2 as cv

from autodp.runner.base_runner import BaseRunner
from autodp.utils.misc import get_class

from sentana.graph.ril_graph import RILGraph
from sentana.config.cf_container import Config as cf
from sentana.utils.misc import clear_model_dir
from sentana.app.ril.batch_sim_env import BatchSimEnv
from sentana.app.ril.exp_buffer import ExpBuffer
from sentana.utils.misc import to_pickle
from sentana.utils.misc import from_pickle
from sentana.utils.misc import clear_model_dir


@BaseRunner.register
class RLRunner(BaseRunner):
    """
    This class implements a runner for reinforcement learning.
    """
    def __init__(self):
        """
        Temporarily do nothing for initialization.
        """
        pass

    @staticmethod
    def _add_extra_example(env):
        """
        Strategy to add extra examples to the training set.
        :param env:
        :return:
        """
        extra = []
        paths = env.get_paths()
        for path in paths:
            if path[-1][1] < 2:
                extra.extend(path * round((cf.max_age + 1.0) / len(path)))

        return extra

    @staticmethod
    def _done_analysis(env, fh, idx):
        """
        This method is used to output images as well as their
        preprocessing paths.
        :param env:
        :param fh:
        :param idx:
        :return:
        """
        trues = env.get_labels()
        paths = env.get_paths()
        for (i, path) in enumerate(paths):
            if path[-1][1] < 2:
                idx += 1

                # Compute label strength
                strength = "weak" if len(path) > cf.max_age else "strong"

                # Store info
                info = str(idx) + "\t\t" + str(trues[i]) + "\t\t" + str(
                    path[-1][1]) + "\t\t" + strength + "\t\t"

                # Traverse current path
                for (p, ex) in enumerate(path):
                    info += str(ex[1]) + " "
                    name = str(idx) + "_" + str(p) + "_" + str(ex[1]) + \
                           "_" + strength + "_" + str(trues[i]) + \
                           "_" + str(path[-1][1]) + ".jpg"
                    file = os.path.join(cf.analysis_path, name)
                    img = ex[0] * 255
                    cv.imwrite(file, img)

                # Store information to file
                info += "\n"
                fh.write(info)

        return idx

    @staticmethod
    def _compute_done(env):
        """
        Search for done images.
        :param env:
        :return:
        """
        rewards, dones, states, actions = [], [], [], []
        trues = env.get_labels()
        paths = env.get_paths()
        for path in paths:
            rewards.append(sum([ex[3] for ex in path]))
            dones.append(path[-1][4])
            states.append(path[-1][2])
            actions.append(path[-1][1])

        return rewards, dones, states, actions, trues

    def train_model(self, cont=False):
        """
        Main method for training rl policy.
        :param cont:
        :return:
        """
        with tf.Graph().as_default(), tf.Session() as sess:
            # Initialize rl agent
            rl_agent_class = get_class(cf.rl_agent)
            rl_agent = rl_agent_class(cont)

            # Do initialization
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))

            if cont:
                # Load the model
                saver = tf.train.Saver(tf.global_variables())
                ckpt = tf.train.get_checkpoint_state(cf.save_model + "/rl")
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    warnings.warn("Model not exist, train a new model now")



    def test_model(self):
        """
        Main method for testing.
        :return:
        """
        with tf.Graph().as_default(), tf.Session() as self._sess:
            #ril_path = os.path.join(expanduser("~"), ".sentana_ril")
            #clear_model_dir(ril_path)
            rg = RILGraph()
            self._sess.run(tf.group(tf.global_variables_initializer(),
                                    tf.local_variables_initializer()))

            # Load the model
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(cf.save_model))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self._sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Model not exist")

            # Actual test
            re_err, test_err, label_predict, label_actual = self._valid_test(
                cf.test_path, rg)

        return re_err, test_err, label_predict, label_actual

    def _valid_test(self, data_path, rg):
        """
        Common method for validation/test.
        :param data_path:
        :param rg:
        :return:
        """
        # Initialize an environment
        env = BatchSimEnv()
        image_batch, qouts, label_actual, label_predict = [], [], [], []
        batch_size, reward_all = cf.batch_size, 0

        # Start to test
        with bz2.BZ2File(data_path, "rb") as df:
            while True:
                # Add more images for batch processing
                if batch_size > 0:
                    images, labels = self._get_batch(df, batch_size)
                    image_batch.extend(images)
                    qouts.extend([0]*len(images))
                    env.add(image_batch=images, label_batch=labels)

                # If no image left, then exit
                if len(image_batch) == 0: break

                # Select actions using the policy network
                [actions, qout] = self._sess.run(
                    [rg.get_next_actions, rg.get_qout],
                    feed_dict={rg.get_instances: image_batch})
                qouts = list(np.array(qouts) + qout[:, 0] - qout[:, 1])

                # Do actions
                env.step(actions, qouts)

                # Collect predictions
                #complete = [not(a and b) for (a, b) in zip(
                #    np.logical_not(dones), ages)]
                #age_done = [not(a or b) for (a, b) in zip(dones, ages)]
                rewards, dones, states, new_acts, trues = self._compute_done(env)
                reward_all += sum(rewards)
                batch_size = sum(dones)
                image_batch = list(compress(
                    states, np.logical_not(dones)))
                label_predict.extend(list(compress(new_acts, dones)))
                label_actual.extend(list(compress(trues, dones)))
                #age_pred = [0 if q > 0 else 1 for q in list(
                #    compress(qouts, age_done))]
                #label_predict.extend(age_pred)
                #label_actual.extend(list(compress(trues, age_done)))
                qouts = list(compress(qouts, np.logical_not(dones)))
                env.update_done(dones)

                #print("Processed up to %d images" % len(label_predict))

        test_err = sum(np.abs(np.array(label_actual)-np.array(
            label_predict)))/len(label_actual)
        print("Test error is: %g" % test_err)

        return -reward_all, test_err, label_predict, label_actual

    def test_and_analysis(self):
        """
        Method for testing and analysis image paths.
        :return:
        """
        with tf.Graph().as_default(), tf.Session() as self._sess:
            #ril_path = os.path.join(expanduser("~"), ".sentana_ril")
            #clear_model_dir(ril_path)
            rg = RILGraph()
            self._sess.run(tf.group(tf.global_variables_initializer(),
                                    tf.local_variables_initializer()))

            # Load the model
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(cf.save_model))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self._sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Model not exist")

            # Actual test and analysis
            # Initialize an environment
            env = BatchSimEnv()
            image_batch, qouts, label_actual, label_predict = [], [], [], []
            batch_size, reward_all, idx = cf.batch_size, 0, 0
            clear_model_dir(cf.analysis_path)
            fh = open(os.path.join(cf.analysis_path, "analysis.txt"), "w")

            # Start to test
            with bz2.BZ2File(cf.test_path, "rb") as df:
                while True:
                    # Add more images for batch processing
                    if batch_size > 0:
                        images, labels = self._get_batch(df, batch_size)
                        image_batch.extend(images)
                        qouts.extend([0]*len(images))
                        env.add(image_batch=images, label_batch=labels)

                    # If no image left, then exit
                    if len(image_batch) == 0: break

                    # Select actions using the policy network
                    [actions, qout] = self._sess.run(
                        [rg.get_next_actions, rg.get_qout],
                        feed_dict={rg.get_instances: image_batch})
                    qouts = list(np.array(qouts) + qout[:, 0] - qout[:, 1])

                    # Do actions
                    env.step(actions, qouts)

                    # Do analysis on successful processed images
                    idx = self._done_analysis(env, fh, idx)

                    # Remove processed images
                    rewards, dones, states, acts, trues = self._compute_done(env)
                    reward_all += sum(rewards)
                    batch_size = sum(dones)
                    image_batch = list(compress(
                        states, np.logical_not(dones)))
                    label_predict.extend(list(compress(acts, dones)))
                    label_actual.extend(list(compress(trues, dones)))
                    qouts = list(compress(qouts, np.logical_not(dones)))
                    env.update_done(dones)

            test_err = sum(np.abs(np.array(label_actual)-np.array(
                label_predict)))/len(label_actual)
            print("Test error is: %g" % test_err)
            fh.close()

        return -reward_all, test_err















