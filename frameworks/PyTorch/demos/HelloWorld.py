"""
HelloWorld.py
My first PyTorch program.
"""
import torch
import numpy as np
from torch.autograd import Variable

# Construct a 5x3 matrix of zeros
x = torch.Tensor(5, 3)
print(x)

# Construct a 5x3 matrix of random digits:
y = torch.rand(5, 3)

# Add two Tensors of the same dimensions together:
print(x + y)
print(torch.add(x, y))

# Provide an output tensor as an argument
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result)

# Compute the addition in-place:
y.add_(x)
print(y)

# Use standard NumPy-like indexing:
print(x[:, 1])

# Resizing/reshaping tensors:
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # Here -1 means to infer the size from other dimensions.
print(x.size(), y.size(), z.size())

# Bridge to numpy with simple operations:
# Convert a Torch Tensor to a NumPy Array:
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

# We can go the other way and convert a numpy array to a Torch Tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# We can move the tensors to the GPU for calculation using CUDA:
if torch.cuda.is_available():
    x = torch.randn(4, 4)
    y = torch.randn(4, 4)
    x = x.cuda()
    y = y.cuda()
    print('GPU\'s are awesome! See:')
    print(x + y)

# Create a variable of all ones
x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)

# Operate on variable:
y = x + 2
print(y)
# Since y is the result of an operation it has a gradient function (grad_fn):
print(y.grad_fn)
# We can interact with variables with some intrinsic methods:
z = y * y * 3
out = z.mean()
print(z, out)

# To perform backpropagation:
out.backward()
# print the gradients d(out)/dx:
print(x.grad)
