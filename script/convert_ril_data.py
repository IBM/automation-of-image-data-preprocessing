"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os
import cv2 as cv
import numpy as np
import json


IM_DIR = "/Users/minhtn/ibm/origin_data/sentana/you_data/images"
OUT_DIR = "/Users/minhtn/ibm/projects/sentana/data/you_data/ril.img"
IM_SIZE = 256


def process_image(im_path):
    # Load image
    im = cv.imread(im_path)

    # Resize image
    ml = np.min(im.shape[:-1])
    resize_im = cv.resize(im, (int(im.shape[1] / ml * IM_SIZE),
                               int(im.shape[0] / ml * IM_SIZE)))

    # Crop image
    dim0 = int((resize_im.shape[0] - IM_SIZE) / 2)
    dim1 = int((resize_im.shape[1] - IM_SIZE) / 2)
    crop_im = resize_im[dim0: dim0+IM_SIZE, dim1: dim1+IM_SIZE, :]

    # Normalize image
    dst = np.zeros(shape=crop_im.shape)
    norm_im = cv.normalize(crop_im, dst, alpha=0, beta=1,
                           norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    # Store image
    #names = im_path.split(sep="/")
    #cv.imwrite("/" + os.path.join(*names[:-1], "crop_" + names[-1]), norm_im)

    return norm_im


def process():
    # Create label indices
    labels = os.listdir(IM_DIR)
    labels = [l for l in labels if not l.startswith(".")]

    # Process images for each label
    with open(OUT_DIR, "w") as f:
        for label in labels:
            files = os.listdir(os.path.join(IM_DIR, label))
            files = [f for f in files if not f.startswith(".")]
            for file in files:
                sd_image = process_image(os.path.join(IM_DIR, label, file))
                label_idx = int(label.split(sep="_")[1])
                line = {"img": np.array(sd_image).tolist(), "label": label_idx}
                json.dump(line, f)
                f.write("\n")

    f.close()


if __name__ == "__main__":
    process()