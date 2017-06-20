"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os
import wget


URL_DIR = "../Dataset_Vso/ANP_Images/URL1553"
IMAGE_DIR = "../data/cc_data"


def download():
    files = os.listdir(URL_DIR)
    files = [f for f in files if not f.startswith(".")]
    for file in files:
        os.mkdir(os.path.join(IMAGE_DIR, file.split(sep=".")[0]))
        with open(os.path.join(URL_DIR, file), "r") as f:
            lines = f.readlines()
            for line in lines:
                url = line.split(sep=" ")[1]
                image = url.split(sep="/")[-1]
                wget.download(url, os.path.join(IMAGE_DIR,
                                                file.split(sep=".")[0],
                                                image))


if __name__ == "__main__":
    download()
