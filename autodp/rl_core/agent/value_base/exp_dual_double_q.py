"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import sys
import os

import numpy as np
from itertools import compress
import tensorflow as tf
import warnings

from autodp.rl_core.agent.base_agent import BaseAgent
from autodp.config.cf_container import Config as cf
from autodp.utils.misc import get_class
from autodp.rl_core.env.batch_env import BatchSimEnv
from autodp.utils.misc import clear_model_dir
from autodp.rl_core.sup.exp_buffer import ExpBuffer
from autodp.utils.misc import from_pickle
from autodp.utils.misc import to_pickle
from autodp.utils.tf_utils import copy_network
from autodp.utils.tf_utils import update_network
from autodp.utils.misc import softmax


class ExpDualDoubleQ(BaseAgent):
    """
    This class implements a RL algorithm using a dual-q network.
    """
    def __init__(self):
        """
        Initialization.
        """
        self._exp_buf = None
        super().__init__()

    def _setup_policy(self):
        """
        Build 2 networks: one to approximate an action-value function
        and the other to select actions.
        :return:
        """
        # Construct 2 reinforcement learning graphs
        rl_graph_class = get_class(cf.rl_graph)
        self._main_graph = rl_graph_class(net_arch=cf.rl_arch,
                                          loss_func=cf.rl_loss,
                                          name="main_graph")
        self._target_graph = rl_graph_class(net_arch=cf.rl_arch,
                                            loss_func=cf.rl_loss,
                                            name="target_graph")

    def load_specific_objects(self):
        """
        This overwritten method initializes an experimental buffer object
        needed to continue learning.
        :return:
        """
        if os.path.isfile(cf.save_model + "/rl/exp_buf.pkl"):
            self._exp_buf = from_pickle(cf.save_model + "/rl/exp_buf.pkl")
        else:
            warnings.warn("Experimental buffer not exist, "
                          "continue learning with new buffer")
            self._exp_buf = ExpBuffer(cf.buf_size)

    def save_specific_objects(self):
        """
        This overwritten method stores an experimental buffer object
        needed to continue learning.
        :return:
        """
        to_pickle(cf.save_model + "/rl/exp_buf.pkl", self._exp_buf)

    def train_policy(self, sess, train_reader, valid_reader):
        """
        Policy improvement and evaluation.
        :param sess:
        :param train_reader:
        :param valid_reader:
        :return:
        """
        # Copy between the 2 graphs
        update_ops = copy_network(tf.trainable_variables(), cf.dbl_coef)

        # Initialize an exp buffer if it is not initialized yet
        if self._exp_buf is None:
            self._exp_buf = ExpBuffer(cf.buf_size)

        # Define the step drop for exploration
        step_drop = (cf.max_explore - cf.min_explore) / cf.anneal_step

        # Initialize variables
        env = BatchSimEnv()
        image_batch = []
        num_step = 0
        reward_all = 0
        done_all = 0
        early_stop = 0
        err_list = []
        best_valid = -sys.maxsize

        # Start training the policy network
        for (images, labels) in train_reader.get_batch(sess=sess):
            # Add more images for batch processing
            image_batch.extend(images)
            env.add(image_batch=images, label_batch=labels)

            while len(image_batch) > 0:
                # Select actions using the policy network
                actions = sess.run(self._main_graph.get_next_action,
                    feed_dict={self._main_graph.get_instance: image_batch})

                # Do exploration
                for i in range(len(actions)):
                    if np.random.rand(1) < cf.max_explore:
                        actions[i] = np.random.randint(0, cf.num_action)

                # Do actions
                env.step(actions)

                # Get train examples
                extra_batch = self._add_extra_example(env)
                self._exp_buf.add(extra_batch)

                # Decrease the exploration
                if cf.max_explore > cf.min_explore:
                    cf.max_explore -= step_drop

                # Train and update the policy network
                if self._exp_buf.get_size > 2*cf.batch_size:
                    # Get train batch
                    train_batch = self._exp_buf.sample(2*cf.batch_size)
                    i_states = np.array([e[0] for e in train_batch])
                    i_actions = np.array([e[1] for e in train_batch])
                    o_states = np.array([e[2] for e in train_batch])
                    i_rewards = np.array([e[3] for e in train_batch])
                    end_mul = np.array([1 - e[4] for e in train_batch])

                    # Compute target q values
                    target_action = sess.run(self._main_graph.get_next_action,
                        feed_dict={self._main_graph.get_instance: o_states})
                    target_qout = sess.run(self._target_graph.get_qout,
                        feed_dict={self._target_graph.get_instance: o_states})
                    qmax = target_qout[range(2*cf.batch_size), target_action]
                    target = i_rewards + cf.gamma * qmax * end_mul

                    # Train the 2 networks
                    [_, err] = sess.run([self._main_graph.get_train_step,
                        self._main_graph.get_error], feed_dict={
                        self._main_graph.get_instance: i_states,
                        self._main_graph.get_current_action: i_actions,
                        self._main_graph.get_label: target})
                    update_network(sess, update_ops)
                    err_list.append(err)

                # Update input data after 1 step
                rewards, dones, states, _, _ = self._compute_done(env)
                reward_all += sum(rewards)
                done_all += sum(dones)
                num_step += 1
                image_batch = list(compress(states, np.logical_not(dones)))
                env.update_done(dones)

                # Do validation after a number of steps
                if num_step % cf.valid_step == 0:
                    valid_reward, _, _, _ = self.predict(sess, valid_reader)
                    if valid_reward > best_valid:
                        best_valid = valid_reward
                        early_stop = 0

                        # Save model
                        clear_model_dir(cf.save_model + "/rl")
                        saver = tf.train.Saver(tf.global_variables())
                        saver.save(sess, cf.save_model + "/rl/model")

                        # Save specific objects
                        self.save_specific_objects()

                    else:
                        early_stop += 1

                    print("Step %d accumulated %g rewards, processed %d images,"
                          " train error %g and valid rewards %g" % (num_step,
                        reward_all, done_all, np.mean(err_list), best_valid))
                    err_list = []

                    if early_stop >= 200:
                        print("Exit due to early stopping")
                        return

    def predict(self, sess, reader, fh=None):
        """
        Apply the policy to predict image classification.
        :param sess:
        :param reader:
        :param fh:
        :return:
        """
        # Initialize variables
        env = BatchSimEnv()
        image_batch = []
        label_actual = []
        label_predict = []
        label_prob = []
        reward_all = 0
        if fh is not None:
            idx = 0
            clear_model_dir(cf.result_path)

        # Start to validate/test
        for (images, labels) in reader.get_batch(sess=sess):
            # Add more images for batch processing
            image_batch.extend(images)
            env.add(image_batch=images, label_batch=labels)

            while len(image_batch) > 0:
                # Select actions using the policy network
                [actions, qout] = sess.run(
                    [self._main_graph.get_next_action, self._main_graph.get_qout],
                    feed_dict={self._main_graph.get_instance: image_batch})

                # Do actions
                env.step(actions, qout[:, 0:cf.num_class])

                if fh is not None:
                    idx = self._done_analysis(env, fh, idx)

                # Collect predictions
                rewards, dones, states, acts, trues = self._compute_done(env)
                reward_all += sum(rewards)
                image_batch = list(compress(states, np.logical_not(dones)))
                label_predict.extend(list(compress(acts, dones)))
                label_actual.extend(list(compress(trues, dones)))
                prob = softmax(qout[:, 0:cf.num_class], axis=1)
                label_prob.extend(list(compress(prob, dones)))
                env.update_done(dones)

        return reward_all, label_predict, label_actual, label_prob














































