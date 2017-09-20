"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os

import csv
import numpy as np


IM_DIR = "/Users/minhtn/ibm/origin_data/sentana/you_data/Agg_AMT_Candidates"
LAB = "/Users/minhtn/ibm/origin_data/sentana/you_data/twitter_three_agrees.txt"


def preprocess():
    with open(LAB, mode="r") as infile:
        reader = csv.reader(infile, delimiter=" ")
        label_dict = {row[0]: row[1] for row in reader}

    for label in np.unique(list(label_dict.values())):
        os.mkdir(os.path.join(IM_DIR, os.pardir, "images", label))

    files = os.listdir(IM_DIR)
    files = [f for f in files if not f.startswith(".")]
    for file in files:
        label = label_dict[file]
        src = os.path.join(IM_DIR, file)
        dst = os.path.join(IM_DIR, os.pardir, "images", label, file)
        os.rename(src=src, dst=dst)


if __name__ == "__main__":
    preprocess()










