"""
HelloWorldNeuralNet.py
Follows the example online at: http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
A more user-friendly tutorial can be found at: https://towardsdatascience.com/pytorch-tutorial-distilled-95ce8781a89c
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    LeNet Convolutional Neural Network
    """
    def __init__(self):
        """
        This is the place where you instantiate all your modules you can later access them using the same names you've
            given them in here.
        """
        super(Net, self).__init__()

        # Input -> C1: feature maps 6 @ (28 x 28)
        # 1 input image channel, 6 output layers, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)

        # C1 -> C3: feature maps 16 @ (10 x 10)
        # 6 input image channel, 16 output layers, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        '''
        The following are linear/affine operations  such as (y = Wx + b). Such methods are in nn.Linear 
            as opposed to nn.Conv2d.
        '''
        # C3 -> S4: feature maps 16 @ (5 x 5) <last feature map layer>
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)

        # S4 -> C5: Fully Connected Layer (120 x 1)
        self.fc2 = nn.Linear(in_features=120, out_features=84)

        # C5 -> F6: Fully Connected Layer (84 x 1)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        """
        forward: Performs a forward pass through the network. Architecturally, this method defines the connections
            between layers in the network (how to get the output of the neural net). This function is called when
            applying the neural net to an input autograd.Variable:
                net = Net()
                net(input) # calls net.forward(input)
        :param x: The input to the neural network.
        :return x: The input to the neural network after undergoing a forward pass through the net.
        """
        '''
        F is a module housing functions (such as pooling) as well as various activation functions. 
        '''
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(input=F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(input=F.relu(self.conv2(x)), kernel_size=(2, 2))
        '''
        The view function takes a Tensor and re-shapes it. Here we are resizing x to a matrix of size:
            (-1 x self.num_flat_features(x)). Of course the -1 isn't really negative one; it means to auto-infer the 
            dimensionality from the other dimensions (see: http://pytorch.org/docs/master/tensors.html#torch.Tensor.view)
        In this particular case the inferred dimension is 1, as we are changing x to a vector with 
            (1 x self.num_flat_features(x))
        '''
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Here we pass through the output layer and return the result:
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        """
        num_flat_features: Returns the number of flat features in the network. A flat feature is a 'flattened' 2D array
            created by adding the results of multiple 2D kernels (one for each channel in the input layer); see:
            https://www.quora.com/What-is-the-meaning-of-flattening-step-in-a-convolutional-neural-network
        :param x: The input data.
        :return num_features: The number of flat features?
        """
        size = x.size()[1:]  # get every dimension except the batch dimension (input dimension of size 1)
        # size = torch.Size([16, 5, 5])
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# Instantiate the network:
net = Net()
print(net)

# We don't need to define the backward function (where gradients are computed via back-prop) this is done for us via
#   autograd.
# Learnable parameters of a model are returned by:
params = list(net.parameters())
print('number of network parameters: %d' % len(params))
print('conv1\'s .weight: %s' % (params[0].size(),))

# Note that the input and output to the forward propagation function is an autograd.Variable. This network expects an
#   input of size (32 x 32). Here is how we would feed the network the input:
input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print('Network Output with Random Input Data:')
print(out)

# Zero the gradient buffers of all parameters and backprops with random gradients:
net.zero_grad()
out.backward(torch.randn(1, 10))

# There are several different loss functions in the nn package. The loss function computes how far off the output is
#   from the target value.
# The nn.MSELoss function computes the mean-squared error between the input and the target.
output = net(input)
target = Variable(torch.arange(1, 11)) # create a dummy target, for example's sake
criterion = nn.MSELoss()
loss = criterion(output, target)
print('For a dummy target, the MSE Loss is:')
print(loss)

# We can follow loss backward through the network and view the computation graph:
'''
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
'''
print('Computation Graph for loss:')
print('\tMSELoss function: %s' % loss.grad_fn)
print('\tLinear function: %s' % loss.grad_fn.next_functions[0][0])
print('\tReLU function: %s' % loss.grad_fn.next_functions[0][0].next_functions[0][0])

# To backpropagate error through the network call loss.backward(). However, existing gradients must be cleared or they
#   will be accumulated.
print()
net.zero_grad()     # zero the gradient buffers.
print('conv1.bias.grad before backward (backpropagation):')
print(net.conv1.bias.grad)
loss.backward()
print('conv1.bias.grad after backward (backpropagation):')
print(net.conv1.bias.grad)

