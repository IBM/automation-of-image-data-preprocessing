import os
import sys

import tensorflow as tf
import numpy as np
from itertools import compress
import bz2
from scipy.special import softmax

from autodp.rl_core.agent.base_agent import BaseAgent
from autodp.utils.misc import get_class, clear_model_dir
from autodp.rl_core.env.batch_env import BatchSimEnv
from autodp import cf


class VanilaReinforce(BaseAgent):
    """This class implements the vanila version of the REINFORCE policy."""
    def __init__(self):
        super().__init__()

    def _setup_policy(self):
        # Loss function must be the advantage log loss
        if cf.rl_loss != "autodp.network.loss.adv_log.AdvLog":
            raise ValueError("Policy gradient methods must be used with the advantage log loss.")

        # Construct a reinforcement learning graph
        rl_graph_class = get_class(cf.rl_graph)
        self._main_graph = rl_graph_class(net_arch=cf.rl_arch, loss_func=cf.rl_loss, name="main_graph")

    def train_policy(self, sess, train_reader, valid_reader, verbose):
        """Policy improvement and evaluation."""
        # Initialize variables
        env = BatchSimEnv()
        image_batch = []
        num_step = 0
        reward_all = 0
        done_all = 0
        err_list = []
        best_valid = -sys.maxsize

        # Start training the policy network
        for (images, labels) in train_reader.get_batch(sess=sess):
            # Add more images for batch processing
            image_batch.extend(images)
            env.add(image_batch=images, label_batch=labels)

            while len(image_batch) > 0.3 * cf.batch_size:
                # Select actions using the policy network
                action_probs = sess.run(self._main_graph.get_action_probs, {self._main_graph.get_instance: image_batch})
                actions = []
                for i in range(len(action_probs)):
                    actions.append(np.random.choice(np.arange(cf.num_action), p=action_probs[i]))

                # Do actions
                env.step(actions)

                # Process done images
                for path in env.get_paths():
                    if path[-1][4]:
                        # Traverse through a done image
                        for (t, p) in enumerate(path):
                            # The return after this timestep
                            sum_return = sum(cf.gamma ** i * ex[3] for i, ex in enumerate(path[t: ]))

                            # Update our policy estimator
                            [_, err] = sess.run([self._main_graph.get_train_step, self._main_graph.get_error],
                                                {self._main_graph.get_instance: np.asarray([p[0]]),
                                                 self._main_graph.get_current_action: np.asarray([p[1]]),
                                                 self._main_graph.get_label: np.asarray([sum_return]),
                                                 self._main_graph.get_phase_train: True,
                                                 self._main_graph.get_keep_prob: cf.keep_prob})
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

                        # Save model
                        clear_model_dir(cf.save_model + "/rl")
                        saver = tf.train.Saver(tf.global_variables())
                        saver.save(sess, cf.save_model + "/rl/model")

                    if verbose:
                        print("Step %d accumulated %g rewards, processed %d images, train err %g and val rewards %g."
                              % (num_step, reward_all, done_all, np.mean(err_list), best_valid))

                    err_list = []
        return -best_valid

    def predict(self, sess, reader):
        """Apply the policy to predict image classification."""
        # Initialize variables
        env = BatchSimEnv()
        image_batch = []
        label_actual = []
        label_predict = []
        label_prob = []
        reward_all = 0

        # Start to validate/test
        for (images, labels) in reader.get_batch(sess=sess):
            # Add more images for batch processing
            image_batch.extend(images)
            env.add(image_batch=images, label_batch=labels)

            while len(image_batch) > 0:
                # Select actions using the policy network
                action_probs = sess.run(self._main_graph.get_action_probs, {self._main_graph.get_instance: image_batch})
                actions = []
                for i in range(len(action_probs)):
                    actions.append(np.argmax(action_probs[i]))

                # Do actions
                env.step(actions, action_probs[:, 0:cf.num_class])

                # Collect predictions
                rewards, dones, states, acts, trues = self._compute_done(env)
                reward_all += sum(rewards)
                image_batch = list(compress(states, np.logical_not(dones)))
                label_predict.extend(list(compress(acts, dones)))
                label_actual.extend(list(compress(trues, dones)))
                prob = softmax(action_probs[:, 0:cf.num_class], axis=1)
                label_prob.extend(list(compress(prob, dones)))
                env.update_done(dones)
        return reward_all, label_predict, label_actual, label_prob

    def preprocess(self, sess, readers, locations):
        """Method to do preprocessing."""
        # Initialize variables
        env = BatchSimEnv()
        image_batch = []

        # Start to preprocess
        for (reader, location) in zip(readers, locations):
            # Initialize file handles
            clear_model_dir(os.path.join(cf.prep_path, location))
            fh = bz2.BZ2File(os.path.join(cf.prep_path, location, location + ".bz2"), "wb")

            # Preprocess images and store them
            for (images, labels) in reader.get_batch(sess=sess):
                # Add more images for batch processing
                image_batch.extend(images)
                env.add(image_batch=images, label_batch=labels)

                while len(image_batch) > 0:
                    # Select actions using the policy network
                    action_probs = sess.run(self._main_graph.get_action_probs,
                                            {self._main_graph.get_instance: image_batch})
                    actions = []
                    for i in range(len(action_probs)):
                        actions.append(np.argmax(action_probs[i]))

                    # Do actions
                    env.step(actions, action_probs[:, 0:cf.num_class])

                    # Collect predictions
                    _, dones, states, _, trues = self._compute_done(env)

                    # Store images
                    self._store_prep_images(fh, list(compress(states, dones)), list(compress(trues, dones)))

                    image_batch = list(compress(states, np.logical_not(dones)))
                    env.update_done(dones)

            # Finish, close files
            fh.close()
