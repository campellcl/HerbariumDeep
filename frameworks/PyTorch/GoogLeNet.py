"""
GoogLeNet.py
An implementation of the GoogLeNet Convolutional Neural Network (also known as Inception v1) with PyTorch.
sources:
    * http://cat2.mit.edu/dh_upload/backup/transfer/miniplaces/miniplaces/model/pytorch/googlenet.py
    * https://github.com/antspy/inception_v1.pytorch/blob/master/inception_v1.py
    * https://github.com/vadimkantorov/metriclearningbench/blob/master/inception_v1_googlenet.py
"""

__created__ = '4/2/2018'
__author__ = "Chris Campell"

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


'''
Dataset parameters:
'''
# The size (in pixels) to scale images to before loading:
load_size = 256
# The receptive field is size 224 x 224:
crop_size = 224
# No specified batch size so use default of 120:
batch_size = 120
# We need the mean of each color channel in the input image to normalize by:
channel_means = None
# The standard deviation of each channel should be 1 so that the range after normalization for the tensor is [-1, 1]
channel_std = [1, 1, 1]


class GoogLeNet(nn.Module):

    def __init__(self):
        super(GoogLeNet, self).__init__()
        # conv2d 0
        # TODO: Two implementations say padding should be 3 here, one says p=1.
        # TODO: Why would this be the case? Do the Andrew Ng math.
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3)
        # Note: the ReLU would be defined here but it is run during the call to the forward pass function.
        self.max_pool0 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        # TODO: What is the Local Resp Norm? Can still issue library calls but would be good to know the math.
        # self.loc_resp_norm = nn.CrossMapLRN2d(size=)

    def forward(self, x):
        output = self.max_pool0(F.relu(self.conv0(x)))
        return x

if __name__ == '__main__':
    '''
    The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors with a 
        normalized range [-1, 1]. We must provide the mean of each channel to the transforms.Normalize function. The
        code commented out below computes the mean of each input channel. 
    '''
    # Compute the means for each channel (r,g,b) of the input image:
    # https://stackoverflow.com/questions/47124143/mean-value-of-each-channel-of-several-images
    # channel_means = [trainset.train_data[...,i].mean() for i in range(trainset.train_data.shape[-1])]

    transform = transforms.Compose(
        [transforms.Scale(load_size),
         transforms.CenterCrop(crop_size),
         transforms.ToTensor(),
         transforms.Normalize(mean=[125.306918046875, 122.950394140625, 113.86538318359375], std=[1, 1, 1])])

    # Download (or load if already present) the CIFAR10 training dataset:
    trainset = torchvision.datasets.CIFAR10(root='./demos/data', train=True, download=True, transform=transform)

    # Download (or load if already present) the CIFAR10 test dataset:
    testset = torchvision.datasets.CIFAR10(root='./demos/data', train=False, download=True, transform=transform)

    # Load the dataset while applying the specified transform:
    # train_loader = torch.utils.data.DataLoader(root='./demos/data')

