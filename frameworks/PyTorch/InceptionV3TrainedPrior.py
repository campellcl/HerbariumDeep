"""
InceptionV3TrainedPrior.py
An example utilizing the Inception v3 torchvision.models implementation with pre-trained weights.
"""

import torch as pt
import torchvision.models as models
import torchvision.transforms as transforms


def main():
    """

    :return:
    """
    inception_v3 = models.inception_v3(pretrained=True)
    '''
    All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images 
        of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a 
        range of [0, 1] and then normalized.
    see: http://pytorch.org/docs/master/torchvision/models.html
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


if __name__ == '__main__':
    main()
