"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os
from os.path import expanduser

import tensorflow as tf
import warnings
from itertools import islice
from itertools import compress
import ast
import numpy as np

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
        lines = islice(data_file, batch_size)
        for line in lines:
            line = ast.literal_eval(line)
            (image, label) = (line["img"], line["label"])
            image_batch.append(np.array(image))
            label_batch.append(label)

        return image_batch, label_batch

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

            if cont:
                # Load the model
                saver = tf.train.Saver(tf.global_variables())
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(
                    cf.save_model))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(self._sess, ckpt.model_checkpoint_path)
                else:
                    warnings.warn("Model not exist, train a new model now")

            # Initialize an exp buffer for current epoch
            exp_buf = ExpBuffer(cf.train_size*10)

            # Start training the policy network
            for epoch in range(cf.num_epoch):
                # Initialize an environment
                env = BatchSimEnv()
                image_batch, batch_size = [], cf.batch_size

                # Process an epoch
                num_step, reward_all = 0, 0
                with open(cf.train_path, "r") as df:
                    while True:
                        # Add more images for batch processing
                        if batch_size > 0:
                            images, labels = self._get_batch(df, batch_size)
                            image_batch.extend(images)
                            env.add(image_batch=images, label_batch=labels)

                        # If no image left, then exit the current epoch
                        if len(image_batch) == 0: break

                        # Do exploration
                        if np.random.rand(1) < cf.exploration:
                            actions = np.random.randint(0, cf.num_action,
                                                        len(image_batch))

                        # Select actions using the policy network
                        else:
                            actions = self._sess.run(rg.get_next_actions,
                                feed_dict={rg.get_instances: image_batch})

                        # Add extra examples to the buffer after doing actions
                        states, rewards, dones, _, ages = env.step(actions)
                        extra = []
                        for (i, a, s, r, d, g) in zip(image_batch, actions,
                                                      states, rewards,
                                                      dones, ages):
                            if a < 2:
                                extra.extend([(i, 1-a, s, -r, d, g),
                                              (i, a, s, r, d, g)])
                            else:
                                extra.extend([(i, a, s, r, d, g)])
                        exp_buf.add(extra)

                        # Decrease the exploration
                        if cf.exploration > cf.min_explore:
                            cf.exploration -= cf.dec_explore

                        # Train and update the policy network
                        if num_step % cf.update_freq == 0:
                            train_batch = exp_buf.sample(cf.train_size)
                            i_states = np.array([e[0] for e in train_batch])
                            i_actions = np.array([e[1] for e in train_batch])
                            o_states = np.array([e[2] for e in train_batch])
                            i_rewards = np.array([e[3] for e in train_batch])
                            end_mul = np.array([1-e[4] for e in train_batch])
                            i_ages = np.array([e[5] for e in train_batch])

                            qmax = self._sess.run(rg.get_qmax,
                                feed_dict={rg.get_instances: o_states})
                            target = i_rewards + (
                                cf.gamma-0.01*i_ages)*qmax*end_mul

                            _ = self._sess.run(rg.get_train_step,
                                feed_dict={rg.get_instances: i_states,
                                           rg.get_actions: i_actions,
                                           rg.get_targets: target})

                        # Update input data after 1 step
                        reward_all += sum(rewards)
                        num_step += 1
                        batch_size = sum(dones)
                        image_batch = list(compress(states,
                                                    np.logical_not(dones)))

                        # Print rewards after every number of steps
                        if num_step % 10 == 0:
                            print("Epoch %d, step %d has accumulated "
                                  "rewards %g and processed %d images"
                                  % (epoch, num_step, reward_all, sum(dones)))

            # Save model
            clear_model_dir(os.path.dirname(cf.save_model))
            saver = tf.train.Saver(tf.global_variables())
            saver.save(self._sess, cf.save_model)

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

            # Initialize an environment
            env = BatchSimEnv()
            image_batch, label_actual, label_predict = [], [], []
            batch_size = cf.batch_size

            # Start to test
            with open(cf.test_path, "r") as df:
                while True:
                    # Add more images for batch processing
                    if batch_size > 0:
                        images, labels = self._get_batch(df, batch_size)
                        image_batch.extend(images)
                        env.add(image_batch=images, label_batch=labels)

                    # If no image left, then exit
                    if len(image_batch) == 0: break

                    # Select actions using the policy network
                    actions = self._sess.run(rg.get_next_actions,
                        feed_dict={rg.get_instances: image_batch})

                    # Do actions
                    states, rewards, dones, trues, _ = env.step(actions)
                    print(actions)

                    # Collect predictions
                    batch_size = sum(dones)
                    image_batch = list(compress(states, np.logical_not(dones)))
                    label_predict.extend(list(compress(actions, dones)))
                    label_actual.extend(list(compress(trues, dones)))
                    print("Processed up to %d images" % len(label_predict))

            test_err = sum(np.abs(np.array(label_actual)-np.array(
                label_predict)))/len(label_actual)
            print("Test error is: %g" % test_err)

        return label_predict, label_actual


