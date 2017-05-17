"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os

import tensorflow as tf
import warnings
from itertools import islice
import ast
import numpy as np

from sentana.graph.ril_graph import RILGraph
from sentana.config.cf_container import Config as cf
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
        lines = islice(data_file, batch_size)
        for line in lines:
            (image, label) = ast.literal_eval(line)
            image_batch.append(image)
            label_batch.append(label)

        return image_batch, label_batch

    def train_policy(self, cont=False):
        with tf.Graph().as_default(), tf.Session() as self._sess:
            ril_path = "~/.sentana_ril"
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

            # Start training the policy network
            for epoch in range(cf.num_epoch):
                with open(cf.train_path, "r") as df:
                    batch_size = cf.batch_size
                    while True:
                        if batch_size > 0:
                            images, labels = self._get_batch(df, batch_size)

                            image_batch = image_batch + images
                            label_batch = label_batch + labels

                        if np.random.rand(1) < cf.exploration:
                            action = np.random.randint(0, cf.num_action,
                                                       len(image_batch))
                        else:
                            action = self._sess.run(
                                rg.get_actions,
                                feed_dict={rg.get_instances: image_batch})


