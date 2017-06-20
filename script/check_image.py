"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os
from PIL import Image

IM_DIR = "/Users/minhtn/ibm/origin_data/sentana/you_data/images/label_0"


def letscheck():
    for file in os.listdir(IM_DIR):
        if not file.startswith("."):
            try:
                im=Image.open(os.path.join(IM_DIR, file))
            except IOError:
                print(file)


if __name__ == "__main__":
    letscheck()