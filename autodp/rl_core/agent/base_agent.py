"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os
import abc

import cv2 as cv
from PIL import Image
import numpy as np
import pickle

from autodp.config.cf_container import Config as cf
from autodp.utils.tf_utils import wrap_image


class BaseAgent(metaclass=abc.ABCMeta):
    """
    This abstract class defines methods exposed by a reinforcement learning
    agent.
    """
    def __init__(self):
        """
        Initialization.
        """
        # Setup neural network functions
        self._setup_policy()

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
            if path[-1][1] < cf.num_class:
                extra.extend(path)

        return extra

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
                    file = os.path.join(cf.result_path, name)
                    img = ex[0] * 255
                    cv.imwrite(file, img)
                    list_im.append(file)

                # Combine images
                imgs = [Image.open(i) for i in list_im]
                shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
                img_com = np.hstack((np.asarray(i.resize(shape)) for i in imgs))
                img_com = Image.fromarray(img_com)
                img_com.save(os.path.join(cf.result_path, str(idx) + ".jpg"))

                # Delete tmp files
                for i in list_im:
                    os.remove(i)

                # Store information to file
                info += "\n"
                fh.write(info)

        return idx

    @staticmethod
    def _store_prep_images(fh, images, labels):
        """
        Store preprocessed images.
        :param fh:
        :param images:
        :param labels:
        :return:
        """
        if cf.reader.split(".")[-1] == "TFReader":
            for (image, label) in zip(images, labels):
                tf_record = wrap_image(image, int(label))
                tf_writer = fh[np.random.randint(0, 5)]
                tf_writer.write(tf_record.SerializeToString())

        else:
            for (image, label) in zip(images, labels):
                line = {"i": image, "l": label}
                pickle.dump(line, fh, pickle.HIGHEST_PROTOCOL)

    @abc.abstractmethod
    def _setup_policy(self):
        """
        Build one or more networks to approximate value, action-value and
        policy functions.
        :return:
        """

    @abc.abstractmethod
    def load_specific_objects(self):
        """
        This method can be overwritten to initialize specific objects needed
        to continue learning.
        :return:
        """

    @abc.abstractmethod
    def save_specific_objects(self):
        """
        This method can be overwritten to store specific objects needed
        to continue learning.
        :return:
        """

    @abc.abstractmethod
    def train_policy(self, sess, train_reader, valid_reader):
        """
        Policy improvement and evaluation.
        :param sess:
        :param train_reader:
        :param valid_reader:
        :return:
        """

    @abc.abstractmethod
    def predict(self, sess, reader):
        """
        Apply the policy to predict image classification.
        :param sess:
        :param reader:
        :return:
        """




















































