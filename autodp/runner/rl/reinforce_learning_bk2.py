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

from sentana.graph.ril_graph import RILGraph
from sentana.config.cf_container import Config as cf
from sentana.utils.misc import clear_model_dir
from sentana.app.ril.batch_sim_env import BatchSimEnv
from sentana.app.ril.exp_buffer import ExpBuffer
from sentana.utils.misc import to_pickle
from sentana.utils.misc import from_pickle
from sentana.utils.misc import clear_model_dir


class ReInLearning(object):
    """
    This class implements a reinforcement learning algorithm with a deep Q
    network as its policy.
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

    def train_policy(self, cont=False):
        """
        Main method for training the policy.
        :param cont:
        :return:
        """
        with tf.Graph().as_default(), tf.Session() as self._sess:
            #ril_path = os.path.join(expanduser("~"), ".sentana_ril")
            #clear_model_dir(ril_path)
            rg = RILGraph()
            self._sess.run(tf.group(tf.global_variables_initializer(),
                                    tf.local_variables_initializer()))

            exp_buf = None
            if cont:
                # Load the model
                saver = tf.train.Saver(tf.global_variables())
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(
                    cf.save_model))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(self._sess, ckpt.model_checkpoint_path)
                else:
                    warnings.warn("Model not exist, train a new model now")

                # Load exp
                if os.path.isfile(os.path.dirname(cf.save_model) + "/exp.pkl"):
                    exp_buf = from_pickle(os.path.dirname(
                        cf.save_model) + "/exp.pkl")

            # Initialize an exp buffer for current epoch
            if exp_buf is None:
                exp_buf = ExpBuffer(10000)
            step_drop = (cf.exploration - cf.min_explore) / cf.anneal_step

            # Start training the policy network
            early_stop, best_valid = 0, sys.maxsize
            for epoch in range(cf.num_epoch):
                # Initialize an environment
                env = BatchSimEnv()
                image_batch, qouts, batch_size = [], [], cf.batch_size

                # Process an epoch
                num_step, reward_all, done_all = 0, 0, 0
                with bz2.BZ2File(cf.train_path, "rb") as df:
                    while True:
                        # Add more images for batch processing
                        if batch_size > 0:
                            images, labels = self._get_batch(df, batch_size)
                            image_batch.extend(images)
                            qouts.extend([0] * len(images))
                            env.add(image_batch=images, label_batch=labels)

                        # If no image left, then exit the current epoch
                        if len(image_batch) == 0:
                            print("Epoch %d, final step has accumulated "
                                  "rewards %g and processed %d images"
                                  % (epoch, reward_all, done_all))
                            break

                        # Do exploration
                        if np.random.rand(1) < cf.exploration:
                            #if np.random.rand(1) < 1.1:
                            actions = np.random.randint(0, cf.num_action,
                                                        len(image_batch))
                            #else:
                            #    actions = np.random.randint(0, 2,
                            #                                len(image_batch))

                        # Select actions using the policy network
                        else:
                            [actions, qout] = self._sess.run(
                                [rg.get_next_actions, rg.get_qout],
                                feed_dict={rg.get_instances: image_batch})
                            qouts = list(np.array(qouts)+qout[:, 0]-qout[:, 1])

                        # Delay adding extra examples to the buffer
                        # after doing actions
                        env.step(actions, qouts)
                        extra = self._add_extra_example(env)
                        exp_buf.add(extra)

                        # Decrease the exploration
                        if cf.exploration > cf.min_explore:
                            cf.exploration -= step_drop

                        # Train and update the policy network
                        if num_step % cf.update_freq == 0 and exp_buf.get_size() > cf.train_size:
                            train_batch = exp_buf.sample(cf.train_size)
                            #train_batch.extend(extra)
                            #train_batch = extra
                            i_states = np.array([e[0] for e in train_batch])
                            i_actions = np.array([e[1] for e in train_batch])
                            o_states = np.array([e[2] for e in train_batch])
                            i_rewards = np.array([e[3] for e in train_batch])
                            end_mul = np.array([1-e[4] for e in train_batch])

                            qmax = self._sess.run(rg.get_qmax,
                                feed_dict={rg.get_instances: o_states})
                            target = i_rewards + cf.gamma*qmax*end_mul
                            target[np.where(target < -1)] = -1
                            target[np.where(target > 1)] = 1
                            #    cf.gamma-0.02*i_ages)*qmax*end_mul

                            #for (i, q) in enumerate(qmax):
                            #    if q < 0: target[i] = -1

                            [_, err] = self._sess.run(
                                [rg.get_train_step, rg.get_error],
                                feed_dict={rg.get_instances: i_states,
                                           rg.get_actions: i_actions,
                                           rg.get_targets: target})

                        # Add extra examples to the buffer
                        #exp_buf.add(extra)

                        # Update input data after 1 step
                        rewards, dones, states, _, _ = self._compute_done(env)
                        reward_all += sum(rewards)
                        done_all += sum(dones)
                        num_step += 1
                        batch_size = sum(dones)
                        image_batch = list(compress(states,
                                                    np.logical_not(dones)))
                        qouts = list(compress(qouts, np.logical_not(dones)))
                        env.update_done(dones)

                        # Print rewards after every number of steps
                        if num_step % 100 == 0:
                            print("Epoch %d, step %d has accumulated "
                                  "rewards %g and processed %d images "
                                  "and train error %g" % (epoch, num_step,
                                    reward_all, done_all, err))

                re_err, valid_err, _, _ = self._valid_test(cf.valid_path, rg)
                if re_err < best_valid:
                    best_valid = re_err
                    early_stop = 0
                    print("Best valid reward is %g" % (-best_valid))

                    # Save model
                    clear_model_dir(os.path.dirname(cf.save_model))
                    saver = tf.train.Saver(tf.global_variables())
                    saver.save(self._sess, cf.save_model)

                    # Save exp
                    to_pickle(os.path.dirname(cf.save_model) + "/exp.pkl",
                              exp_buf)

                else:
                    early_stop += 1

                if early_stop >= 3000:
                    if epoch > 3:
                        print("Exit due to early stopping")
                        break

                    else: early_stop = 0

    def test_policy(self):
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















