"""
InceptionV3TrainedPrior.py
An example utilizing the Inception v3 torchvision.models implementation with pre-trained weights.
source: http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""

import torch as pt
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import os
import numpy as np
import matplotlib.pyplot as plt


def imshow_tensor(input, title=None):
    """
    imshow_tensor: Matplotlib imshow function for PyTorch Tensor Objects.
    :source URL: http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    :param input: The input image as a Tensor.
    :param title: The title for the image.
    :return:
    """
    # Note: not sure what the point of this transposition is:
    input = input.numpy().transpose((1, 2, 0))
    # Normalize the input Tensor:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    # Restrict to [0, 1] interval:
    input = np.clip(input, a_min=0, a_max=1)
    plt.imshow(input)
    if title is not None:
        plt.title(title)
    # plt.pause(0.001)    # pause a second so that plots are updated?
    plt.show()


def main():
    """

    :return:
    """
    # Get a batch of training data:
    inputs, classes = next(iter(data_loaders['train']))
    # Make a grid from batch:
    out = torchvision.utils.make_grid(inputs)
    # Display the images:
    imshow_tensor(input=out, title=[class_names[x] for x in classes])
    # inception_v3 = models.inception_v3(pretrained=True)


if __name__ == '__main__':
    data_dir = '../../data/ImageNet/SubSets/hymenoptera_data/'
    input_load_size = 256
    receptive_field_size = 224

    '''
    Training Data and Validation Data Input Pipeline:
        Data Augmentation and Normalization as described here: http://pytorch.org/docs/master/torchvision/models.html
    '''
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(receptive_field_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_load_size),
            transforms.CenterCrop(receptive_field_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    print('INIT: Loaded image datasets.')
    data_loaders = {x: pt.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
                    for x in ['train', 'val']}
    print('INIT: Created data loaders.')
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print('Number of Images in Each Dataset: %s' % dataset_sizes)
    class_names = image_datasets['train'].classes
    print('All class labels in the dataset: %s' % class_names)
    use_gpu = pt.cuda.is_available()
    print('CUDA is enabled?: %s\nWill use GPU to train?: %s' % (use_gpu, use_gpu))
    main()
