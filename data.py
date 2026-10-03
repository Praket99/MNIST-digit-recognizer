
#this whole sectiom of code is copied from Chat-GPT
#this code is used to download the MNIST dataset and load it into numpy arrays for training and testing the neural network

import os
import gzip
import urllib.request
import numpy as np


BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz"
}


def download_mnist():

    os.makedirs("mnist_data", exist_ok=True)

    for filename in FILES.values():

        path = os.path.join("mnist_data", filename)

        if not os.path.exists(path):

            print("Downloading:", filename)

            url = BASE_URL + filename

            urllib.request.urlretrieve(url, path)

            print("Downloaded!")


def load_images(filename):

    with gzip.open(filename, "rb") as f:

        data = np.frombuffer(f.read(), np.uint8, offset=16)

    data = data.reshape(-1, 784)

    return data.astype(np.float32) / 255.0


def load_labels(filename):

    with gzip.open(filename, "rb") as f:

        data = np.frombuffer(f.read(), np.uint8, offset=8)

    return data.astype(np.int64)


def load_data():

    download_mnist()

    x_train = load_images(
        "mnist_data/train-images-idx3-ubyte.gz"
    )

    y_train = load_labels(
        "mnist_data/train-labels-idx1-ubyte.gz"
    )

    x_test = load_images(
        "mnist_data/t10k-images-idx3-ubyte.gz"
    )

    y_test = load_labels(
        "mnist_data/t10k-labels-idx1-ubyte.gz"
    )

    return x_train, y_train, x_test, y_test