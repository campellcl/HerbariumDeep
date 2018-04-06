"""
GoogLeNet.py
An implementation of the Inception v1 (GoogLeNet) convolutional neural network in TensorFlow.
source:
"""

__created__ = "4/3/2018"
__author__ = "Chris Campell"

from tensorflow.examples.tutorials.mnist import input_data


def main():
    # TODO: Read more into creating a Tf-slim dataset descriptor to load MNIST data:
    # https://github.com/tensorflow/models/tree/master/research/slim
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print(mnist.train.images)

    pass


if __name__ == '__main__':
    main()
