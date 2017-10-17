"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os

import tensorflow as tf

from autodp.reader.base_reader import BaseReader
from autodp.config.cf_container import Config as cf


@BaseReader.register
class TFReader(BaseReader):
    """
    This class contains implementation of a data reader that will read
    data as batches of tensorflow records with shuffling.
    """
    def __init__(self, path, num_epoch=1):
        """
        Initialization.
        :param path:
        :param num_epoch:
        """
        super().__init__(path, num_epoch)

    def _read_and_decode(self, file_queue):
        """
        Decode data.
        :param file_queue:
        :return:
        """
        # Declare a tensorflow reader and read data serially
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(file_queue)
        features = tf.parse_single_example(serialized_example, features={
            "i": tf.VarLenFeature(tf.string),
            "l": tf.FixedLenFeature([], tf.int64)})

        # Convert from a scalar string tensor to a float tensor and reshape
        image = tf.reshape(tf.decode_raw(features["i"].values, tf.float32),
                           [cf.ima_height, cf.ima_width, cf.ima_depth])
        label = tf.cast(features["l"], tf.int32)

        return image, label

    def _read_input(self, batch_size):
        """
        Reads input data num_epoch times.
        :param batch_size:
        :return:
        """
        # Get full file name of all input files
        file_names = os.listdir(self._path)
        file_names = [os.path.join(self._path, f) for f in file_names]

        # Define a tensorflow queue
        file_queue = tf.train.string_input_producer(string_tensor=file_names,
                                                    num_epochs=self._num_epoch)

        # Read and decode data from queue
        image, label = self._read_and_decode(file_queue)

        # Collect examples into batch
        images, labels = tf.train.shuffle_batch(tensors=[image, label],
            num_threads=3, batch_size=batch_size, min_after_dequeue=100,
            capacity=100+3*cf.batch_size, allow_smaller_final_batch=True)

        return images, labels

    def get_batch(self, batch_size=cf.batch_size, sess=None):
        """
        This function implements the abstract method of the super class and
        is used to read data as batch per time.
        :param batch_size:
        :param sess:
        :return:
        """
        # Get tensorflow records
        images, labels = self._read_input(batch_size)

        # Return tensorflow records if sess is not passed else normal records
        if sess is not None:
            #print([v.name for v in tf.local_variables()])
            sess.run(tf.local_variables()[-1].initializer)

            # Start reading tfrecords
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                while not coord.should_stop():
                    [image_batch, label_batch] = sess.run([images, labels])
                    yield (image_batch, label_batch)

            except tf.errors.OutOfRangeError:
                pass

            finally:
                coord.request_stop()
                coord.join(threads)

        else:
            raise ValueError("Session cannot be None")












































































