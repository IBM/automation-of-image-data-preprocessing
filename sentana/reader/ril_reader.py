"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os

import tensorflow as tf

from sentana.reader.base_reader import BaseReader
from sentana.config.cf_container import Config as cf


@BaseReader.register
class RILReader(BaseReader):
    """
    This class contains implementations of a data reader for a reinforcement
    learning task.
    """
    def __init__(self, path):
        """
        Initialization.
        :param path:
        """
        super().__init__(path)

    def _read_and_decode(self, file_queue):
        """
        Decode data.
        :param file_queue:
        :return:
        """
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(file_queue)
        features = tf.parse_single_example(serialized_example, features={
            "image": tf.VarLenFeature(tf.string),
            "action": tf.FixedLenFeature([], tf.int64),
            "reward": tf.FixedLenFeature([], tf.float32)})

        # Convert from a scalar string tensor to a float tensor and reshape
        image = tf.reshape(tf.decode_raw(features["image"].values, tf.float32),
                           [cf.ima_height, cf.ima_width, 3])
        action = features["action"].values
        reward = features["reward"].values

        return image, action, reward

    def _read_input(self):
        """
        Reads input data num_epoch times.
        :return:
        """
        file_names = os.listdir(self._path)
        file_names = [f for f in file_names if not f.startswith(".")]
        file_names = [os.path.join(self._path, f) for f in file_names]

        file_queue = tf.train.string_input_producer(file_names,
                                                    num_epochs=cf.num_epoch)

        # Even when reading in multiple threads, share the file queue
        image, action, reward = self._read_and_decode(file_queue)

        # Collect examples into batch
        images, actions, rewards = tf.train.shuffle_batch(
            [image, action, reward], num_threads=3, batch_size=cf.batch_size,
            capacity=100 + 3 * cf.batch_size, allow_smaller_final_batch=True)

        return images, actions, rewards

    def get_batch(self):
        """
        This function implements the abstract method of the super class and
        is used to read data as batch per time.
        :return:
        """
        images, actions, rewards = self._read_input()

        return (images, actions, rewards)