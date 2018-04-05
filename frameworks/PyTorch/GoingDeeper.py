"""
GoingDeeper.py
Automated species detection of herbarium specimen via convolutional neural network.
Source: Going Deeper in the Automated Identification of Herbarium Specimens (BMC Evolutionary Biology 2017)
"""

__author__ = "Chris Campell"
__created__ = "3/27/2018"

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class GNBM(nn.Module):
    """
    GoogleNet with Batch Normalization (GNBM).
    """

    def __init__(self):
        """
        __init__: Initializes the network.
        """
        super(GNBM, self).__init__()
        # Input -> conv1: 1 input patch (7x7), output size (112x112x64), kernel_size: 2
        self.conv_1 = nn.Conv1d(in_channels=7, out_channels=(112*112*64), kernel_size=2)
        # self.conv1 = nn.Conv2d(in_channels=7, out_channels=(112*112*64), kernel_size=2)

        # Max Pooling over a (3 x 3) window with kernel size of 2:
        self.max_pool_1 = nn.MaxPool2d()



def main():
    pass


if __name__ == '__main__':
    main()


