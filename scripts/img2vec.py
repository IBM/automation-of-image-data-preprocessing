"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os

import bz2
import pickle

from autodp.utils.feature_extractor import YouTube8MFeatureExtractor


DIR = "/Users/minhtn/ibm/projects/autodp/storage/inputs/dogcat/data200/valid"


def convert():
    extractor = YouTube8MFeatureExtractor()
    file = os.path.join(DIR, os.listdir(DIR)[0])
    writer = bz2.BZ2File(os.path.join(DIR, "img2vec.bz2"), "wb")
    with bz2.BZ2File(file, "rb") as df:
        try:
            while True:
                line = pickle.load(df)
                vec = extractor.extract_rgb_frame_features(line["i"])
                pickle.dump({"i": vec, "l": line["l"]}, writer,
                            pickle.HIGHEST_PROTOCOL)

        except EOFError:
            print ("Finished !")
            writer.close()


if __name__ == "__main__":
    convert()














