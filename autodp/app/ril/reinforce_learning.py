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
from PIL import Image
import pandas as pd

from sentana.graph.ril_graph import RILGraph
from sentana.config.cf_container import Config as cf
from sentana.utils.misc import clear_model_dir
from sentana.app.ril.batch_sim_env import BatchSimEnv
from sentana.app.ril.exp_buffer import ExpBuffer


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

    def train_policy(self, cont=False):
        """
        Main method for training the policy.
        :param cont:
        :return:
        """
        with tf.Graph().as_default(), tf.Session() as self._sess:
            rg = RILGraph()
            self._sess.run(tf.group(tf.global_variables_initializer(),
                                    tf.local_variables_initializer()))

            if cont:
                # Load the model
                saver = tf.train.Saver(tf.global_variables())
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(
                    cf.save_model + "/normal/model"))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(self._sess, ckpt.model_checkpoint_path)
                else:
                    warnings.warn("Model not exist, train a new model now")

            # Initialize exp buffer
            exp_buf = ExpBuffer(20000)

            # Initialize an exp buffer for current epoch
            step_drop = (cf.exploration - cf.min_explore) / cf.anneal_step

            best_valid, err = sys.maxsize, -1
            # Start training the policy network
            for epoch in range(cf.num_epoch):
                # Initialize an environment
                env = BatchSimEnv()
                image_batch, batch_size = [], cf.batch_size

                # Process an epoch
                num_step, reward_all, done_all = 0, 0, 0
                with bz2.BZ2File(cf.train_path, "rb") as df:
                    while True:
                        # Add more images for batch processing
                        if batch_size > 0:
                            images, labels = self._get_batch(df, batch_size)
                            image_batch.extend(images)
                            env.add(image_batch=images, label_batch=labels)

                        # If no image left, then exit the current epoch
                        if len(image_batch) == 0:
                            print("Epoch %d, final step has accumulated "
                                  "rewards %g and processed %d images"
                                  % (epoch, reward_all, done_all))
                            break

                        # Do exploration
#                        if np.random.rand(1) < cf.exploration:
#                            actions = np.random.randint(0, cf.num_action,
#                                                        len(image_batch))
                        if np.random.rand(1) < cf.exploration:
                            if np.random.rand(1) < 0.8:
                                actions = np.random.randint(0, cf.num_action,
                                                            len(image_batch))
                            else:
                                actions = np.random.randint(0, 2,
                                                            len(image_batch))

                        # Select actions using the policy network
                        else:
                            [actions] = self._sess.run(
                                [rg.get_next_actions],
                                feed_dict={rg.get_instances: image_batch})


                        # Add extra examples to the buffer after doing actions
                        states, rewards, dones, _ = env.step(actions)

                        train_batch = []
                        for (i, a, s, r, d) in zip(image_batch, actions,
                                                   states, rewards, dones):
                            if r is not None:
                                train_batch.append((i, a, s, r, d))

                        exp_buf.add(train_batch)

                        # Decrease the exploration
                        if cf.exploration > cf.min_explore:
                            cf.exploration -= step_drop

                        # Train and update the policy network
                        #if len(train_batch) > 0:
                        if exp_buf.get_size() > 10*cf.train_size:
                            train_batch = exp_buf.sample(cf.train_size)
                            i_states = np.array([e[0] for e in train_batch])
                            i_actions = np.array([e[1] for e in train_batch])
                            o_states = np.array([e[2] for e in train_batch])
                            i_rewards = np.array([e[3] for e in train_batch])
                            end_mul = np.array([1-e[4] for e in train_batch])

                            qmax = self._sess.run(rg.get_qmax,
                                feed_dict={rg.get_instances: o_states})
                            target = i_rewards + cf.gamma*qmax*end_mul

                            [_, err] = self._sess.run(
                                [rg.get_train_step, rg.get_error],
                                feed_dict={rg.get_instances: i_states,
                                           rg.get_actions: i_actions,
                                           rg.get_targets: target})

                        # Update input data after 1 step
                        reward_all += sum([r for r in rewards if r is not None])
                        done_all += sum(dones)
                        num_step += 1
                        batch_size = sum(dones)
                        image_batch = list(compress(states,
                                                    np.logical_not(dones)))

                        # Print rewards after every number of steps
                        if num_step % 100 == 0:
                            print("Epoch %d, step %d has accumulated "
                                  "rewards %g and processed %d images "
                                  "and train error %g" % (epoch, num_step,
                                    reward_all, done_all, err))

                valid_err, _, _ = self._valid_test(cf.valid_path, rg)
                if valid_err < best_valid:
                    best_valid = valid_err
                    print("Best valid err is %g" % best_valid)

                    # Save model
                    clear_model_dir(cf.save_model + "/best")
                    saver = tf.train.Saver(tf.global_variables())
                    saver.save(self._sess, cf.save_model + "/best/model")

                clear_model_dir(cf.save_model + "/normal")
                saver = tf.train.Saver(tf.global_variables())
                saver.save(self._sess, cf.save_model + "/normal/model")

    def test_policy(self):
        """
        Main method for testing.
        :return:
        """
        with tf.Graph().as_default(), tf.Session() as self._sess:
            rg = RILGraph()
            self._sess.run(tf.group(tf.global_variables_initializer(),
                                    tf.local_variables_initializer()))

            # Load the model
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(
                cf.save_model + "/normal/model"))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self._sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Model not exist")

            # Actual test
            test_err, label_predict, label_actual = self._valid_test(
                cf.test_path, rg)

        return test_err, label_predict, label_actual

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
        batch_size = cf.batch_size

        # Start to test
        with bz2.BZ2File(data_path, "rb") as df:
            while True:
                # Add more images for batch processing
                if batch_size > 0:
                    images, labels = self._get_batch(df, batch_size)
                    image_batch.extend(images)
                    qouts.extend([[0]*2]*len(images))
                    env.add(image_batch=images, label_batch=labels)

                # If no image left, then exit
                if len(image_batch) == 0: break

                # Select actions using the policy network
                [actions, qout] = self._sess.run(
                    [rg.get_next_actions, rg.get_qout],
                    feed_dict={rg.get_instances: image_batch})
                qouts = list(np.array(qouts) + qout[:, :2])

                # Do actions
                states, rewards, dones, trues, new_acts = env.step_valid(
                    actions, qouts)

                batch_size = sum(dones)
                image_batch = list(compress(
                    states, np.logical_not(dones)))
                label_predict.extend(list(compress(new_acts, dones)))
                label_actual.extend(list(compress(trues, dones)))
                qouts = list(compress(qouts, np.logical_not(dones)))

        tmp = np.abs(np.array(label_actual)-np.array(label_predict))
        bool_tmp = [bool(t) for t in tmp]
        test_err = sum(bool_tmp)/len(label_actual)
        print("Test error is: %g" % test_err)

        return test_err, label_predict, label_actual

    def test_and_analysis(self):
        """
        Method for testing and analysis image paths.
        :return:
        """
        with tf.Graph().as_default(), tf.Session() as self._sess:
            rg = RILGraph()
            self._sess.run(tf.group(tf.global_variables_initializer(),
                                    tf.local_variables_initializer()))

            # Load the model
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(cf.save_model + "/best/model"))
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
                        qouts.extend([[0] * 2] * len(images))
                        env.add(image_batch=images, label_batch=labels)

                    # If no image left, then exit
                    if len(image_batch) == 0: break

                    # Select actions using the policy network
                    [actions, qout] = self._sess.run(
                        [rg.get_next_actions, rg.get_qout],
                        feed_dict={rg.get_instances: image_batch})
                    qouts = list(np.array(qouts) + qout[:, :2])

                    # Do actions
                    env.step_analysis(actions, qouts)

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

            tmp = np.abs(np.array(label_actual)-np.array(label_predict))
            bool_tmp = [bool(t) for t in tmp]
            test_err = sum(bool_tmp)/len(label_actual)
            print("Test error is: %g" % test_err)
            fh.close()

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
            if path[-1][1] < cf.num_class:
                idx += 1

                # Compute label strength
                strength = "weak" if len(path) > cf.max_age else "strong"

                # Store info
                info = str(idx) + "\t\t" + str(trues[i]) + "\t\t" + str(
                    path[-1][1]) + "\t\t" + strength + "\t\t"

                # Traverse current path
                list_im = []
                for (p, ex) in enumerate(path):
                    info += str(ex[1]) + " "
                    name = str(idx) + "_" + str(p) + ".jpg"
                    file = os.path.join(cf.analysis_path, name)
                    img = ex[0] * 255
                    cv.imwrite(file, img)
                    list_im.append(file)

                # Combine images
                imgs = [Image.open(i) for i in list_im]
                shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
                img_com = np.hstack((np.asarray(i.resize(shape)) for i in imgs))
                img_com = Image.fromarray(img_com)
                img_com.save(os.path.join(cf.analysis_path, str(idx) + ".jpg"))

                # Delete tmp files
                for i in list_im:
                    os.remove(i)

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

    def _predict(self):
        """
        Common method for validation/test.
        :param data_path:
        :param rg:
        :return:
        """
        with tf.Graph().as_default(), tf.Session() as self._sess:
            rg = RILGraph()
            self._sess.run(tf.group(tf.global_variables_initializer(),
                                    tf.local_variables_initializer()))

            # Load the model
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(cf.save_model + "/best/model"))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self._sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Model not exist")

            # Initialize an environment
            env = BatchSimEnv()
            image_batch, qouts, label_actual, label_predict, label_predict_acc = [], [], [], [], []
            batch_size = cf.batch_size

            # Start to test
            with bz2.BZ2File(cf.test_path, "rb") as df:
                while True:
                    # Add more images for batch processing
                    if batch_size > 0:
                        images, labels = self._get_batch(df, batch_size)
                        image_batch.extend(images)
                        qouts.extend([[0]*2]*len(images))
                        env.add(image_batch=images, label_batch=labels)

                    # If no image left, then exit
                    if len(image_batch) == 0: break

                    # Select actions using the policy network
                    [actions, qout] = self._sess.run(
                        [rg.get_next_actions, rg.get_qout],
                        feed_dict={rg.get_instances: image_batch})
                    qouts = list(np.array(qouts) + qout[:, :2])

                    # Do actions
                    states, rewards, dones, trues, new_acts = env.step_valid(
                        actions, qouts)

                    batch_size = sum(dones)
                    image_batch = list(compress(
                        states, np.logical_not(dones)))
                    label_predict_acc.extend(list(compress((np.exp(qouts) / np.sum(np.exp(qouts), axis=1))[:, 1], dones)))
                    label_predict.extend(list(compress((np.exp(qout[:, :2]) / np.sum(np.exp(qout[:, :2]), axis=1))[:, 1], dones)))
                    label_actual.extend(list(compress(trues, dones)))
                    qouts = list(compress(qouts, np.logical_not(dones)))

            df = pd.DataFrame({"id": label_actual, "pred": label_predict})
            df_acc = pd.DataFrame({"id": label_actual, "pred": label_predict_acc})
            df.to_csv("kaggle.csv", header=False, sep=',', index=False)
            df_acc.to_csv("kaggle_acc.csv", header=False, sep=',', index=False)

