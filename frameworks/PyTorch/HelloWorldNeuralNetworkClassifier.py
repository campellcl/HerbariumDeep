"""
HelloWorldNeuralNetworkClassifier.py
Implementation of the Training a Classifier section of the PyTorch: A 60 Minute Blitz tutorial. Performs the following:
1. Loads and normalizes the CIFAR10 training and test datasets with torchvision
2. Defines a CNN
3. Defines a loss function
4. Trains the network on the training data
5. Evaluates the neural network via the loss function on the test data.
Source: http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

__author__ = "Chris Campell"
__version__ = "3/23/2018"

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


'''
We can view some of the training images:
'''
def imshow(img):
    img = img / 2 + 0.5         # un-normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # grab some random training images:
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # show images
    imshow(torchvision.utils.make_grid(images))



'''
The torchvision package contains the CIFAR10 dataset. This dataset has the classes: 
    ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. 
The images in CIFAR-10 are of size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size. 
'''
if __name__ == '__main__':
    # The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to
    #   Tensors of normalized range [-1, 1].
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    '''
    NOTE: This script was executed on a windows 10 x86_64 machine, which required moving torch.utils.data.DataLoader's
        to the main method. This script will not run without 'if __name__ == '__main__' on a windows machine. It will
        not run if you issue calls to torch.utils.data.DataLoader() from the global scope (i.e. in this method). 
    '''

    # Download (or load if already present) the CIFAR10 training dataset:
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Download (or load if already present) the CIFAR10 test dataset:
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    main()






