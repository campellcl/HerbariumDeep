"""
GoogLeNet.py
An implementation of the GoogLeNet Convolutional Neural Network (also known as Inception v1) with PyTorch.
sources:
    * http://cat2.mit.edu/dh_upload/backup/transfer/miniplaces/miniplaces/model/pytorch/googlenet.py
"""

__created__ = '4/2/2018'
__author__ = "Chris Campell"

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

'''
Dataset parameters:
'''
# The receptive field is 224 x 224:
load_size = 224


class GoogLeNet(nn.Module):

    def __init__(self):
        super(GoogLeNet, self).__init__()
        # The size of the receptive field is 224 x 224 with 3 channel RGB and normalization via mean subtraction.
        # TODO: Where is the receptive field loaded?

        # the pre-layers are the initial pre-processing steps.
        self.pre_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, )
            # Performing a Batch2d normalization standardizes pixel intensities. We don't want to learn to tell day
            #   from night, we want the features in the image. Not the discrepancy between intensities.
        )
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=())

    def forward(self, x):
        return x
