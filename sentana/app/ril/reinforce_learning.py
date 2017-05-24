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
        with tf.Graph().as_default(), tf.Session() as self._sess:
            ril_path = os.path.join(expanduser("~"), ".sentana_ril")
            clear_model_dir(ril_path)
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

            # Initialize an environment
            image_batch, label_batch, batch_size = [], [], cf.batch_size
            env = BatchSimEnv()

            # Start training the policy network
            for epoch in range(cf.num_epoch):
                # Initialize an exp buffer for current epoch
                exp_buf = ExpBuffer()

                # Process an epoch
                num_step, reward_all = 0, 0
                with open(cf.train_path, "r") as df:
                    while True:
                        if batch_size > 0:
                            images, labels = self._get_batch(df, batch_size)
                            image_batch.extend(images)
                            label_batch.extend(labels)
                            env.add(image_batch=images, label_batch=labels)

                        if len(image_batch) == 0: break

                        if np.random.rand(1) < cf.exploration:
                            actions = np.random.randint(0, cf.num_action,
                                                       len(image_batch))
                        else:
                            actions = self._sess.run(rg.get_next_actions,
                                feed_dict={rg.get_instances: image_batch})

                        states, rewards, dones = env.step(actions)
                        extra = []
                        for (i, a, s, r, d) in zip(
                                image_batch, actions, states, rewards, dones):
                            if a < 2:
                                extra.extend([(i, 1-a, s, -r, d),
                                              (i, a, s, r, d)])
                            else:
                                extra.extend([(i, a, s, r, d)])
                        exp_buf.add(extra)

                        if cf.exploration > 0.1:
                            cf.exploration -= (0.9 / 10000)

                        if num_step % 5 == 0:
                            train_batch = exp_buf.sample(32)
                            i_states = np.array([e[0] for e in train_batch])
                            i_actions = np.array([e[1] for e in train_batch])
                            o_states = np.array([e[2] for e in train_batch])
                            i_rewards = np.array([e[3] for e in train_batch])
                            end_mul = np.array([1-e[4] for e in train_batch])

                            qmax = self._sess.run(rg.get_qmax,
                                feed_dict={rg.get_instances: o_states})
                            target = i_rewards + 0.99*qmax*end_mul
                            _ = self._sess.run(rg.get_train_step,
                                feed_dict={rg.get_instances: i_states,
                                           rg.get_actions: i_actions,
                                           rg.get_targets: target})

                        reward_all += sum(rewards)
                        num_step += 1
                        batch_size = len(dones) - sum(dones)
                        image_batch = list(compress(image_batch,
                                                    np.logical_not(dones)))
                        label_batch = list(compress(label_batch,
                                                    np.logical_not(dones)))

                        if num_step % 1 == 0:
                            print("Epoch %d, step %d has rewards %g" % (
                                epoch, num_step, reward_all))

    def test_policy(self):
        pass
