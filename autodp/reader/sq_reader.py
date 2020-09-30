import os

import bz2
import pickle
import numpy as np

from autodp.reader.base_reader import BaseReader
from autodp import cf


@BaseReader.register
class SQReader(BaseReader):
    """This class implements a data reader that will read a file of data sequentially without shuffling."""
    def __init__(self, path, num_epoch=1):
        super().__init__(path, num_epoch)

        # Get full file name of input file
        file = os.listdir(self._path)[0]
        self._file = os.path.join(self._path, file)

    @staticmethod
    def _get_batch(data_file, batch_size):
        """Get a batch of data instances."""
        images, labels = [], []
        for _ in range(batch_size):
            try:
                line = pickle.load(data_file)
                images.append(line["i"].astype(np.float32))
                labels.append(line["l"])
            except EOFError:
                break

        return images, labels

    def get_batch(self, batch_size=cf.batch_size, sess=None):
        """This function implements the abstract method of the super class and is used to read data as batch."""
        for epoch in range(self._num_epoch):
            with bz2.BZ2File(self._file, "rb") as df:
                while True:
                    images, labels = self._get_batch(df, batch_size)
                    if len(images) == 0: break
                    yield (images, labels)
