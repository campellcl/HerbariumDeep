"""
InceptionV3TrainedPrior.py
An example utilizing the Inception v3 torchvision.models implementation with pre-trained weights.
source: http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""
import warnings
import torch as pt
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import copy
from frameworks.PyTorch.logger import Logger



with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
# plt.ion()   # interactive mode


class AverageMeter(object):
    """
    AverageMeter: Computes and stores the average and current value.
    source: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    accuracy: Computes the precision@k for the specified values of k
    source: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    maxk = max(topk)
    batch_size = target.size(0)

    # http://pytorch.org/docs/stable/torch.html#torch.topk
    # TODO: This dim choice may not be correct investigating why output is 5x2 and lables is 5x1
    # max_preds, max_pred_indices = pt.max(output.data, dim=1)
    _, pred = pt.topk(output.data, k=maxk, dim=0)
    # _, preds = pt.topk(max_preds, k=maxk, sorted=False)
    # _, pred = pt.topk(output, k=maxk, dim=0, sorted=True)
    # _, pred = output.topk(maxk, 1, True, True)
    # _, preds = pt.max(outputs.data, dim=1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


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
    fig = plt.figure()
    has_title = title is not None
    # fig.add_suplot(Rows,Cols,Pos)
    # Below code does not work because we are dealing with a tensor object.
    # for position in range(num_image):
    #     sub_plt = fig.add_subplot(1, num_image, position+1)
    #     if has_title:
    #         sub_plt.set_title(title[position])
    #         plt.imshow(input)
    # fig.show()
    # a = fig.add_subplot(1, num_image, 0)
    # if has_title:
    #     a.set_title(title[0])
    # b = fig.add_subplot(1, num_image, 1)
    # if has_title:
    #     b.set_title(title[1])
    plt.imshow(input)
    if title is not None:
        plt.title(title)
    # plt.pause(0.001)    # pause a second so that plots are updated?
    # plt.figure(num='Training Data and Ground Truth Labels')
    plt.show()


def get_top_1_error(model):
    # This is the same as the overall accuracy (how many times is the network correct out of all of the test samples).
    if has_test_set:
        # entire_test_set = pt.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=True, num_workers=1)
        # sequential_test_loader = plt.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=1)
        data_iter = iter(test_loader)
        data_loader = test_loader
    else:
        # entire_val_set = pt.utils.data.DataLoader(valset, batch_size=len(valset), shuffle=True, num_workers=1)
        # sequential_val_loader = plt.utils.data.DataLoader(valset, batch_size=1, shuffle=True, num_workers=1)
        data_iter = iter(val_loader)
        data_loader = val_loader
    # Ground truth images and class labels:
    # images, labels = data_iter.next()
    # predict:
    # if use_gpu:
    #     outputs = model(Variable(images.cuda()))
    # else:
    #     outputs = model(Variable(images))
    # true class labels:
    # _, predicted = pt.max(outputs.data, 1)
    # evaluate the model:
    original_model_state_is_training = model.training
    if model.training:
        model.train(False)
    correct = 0
    total = 0
    for data in data_loader:
        images, labels = data
        if use_gpu:
            outputs = model(Variable(images.cuda()))
        else:
            outputs = model(Variable(images))
        _, predicted = pt.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()
    if original_model_state_is_training:
        model.train(True)
    return 100 * correct / total


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """
    train_model: Trains the model.
    :source URL: http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    :param model:
    :param criterion:
    :param optimizer:
    :param scheduler: An LR scheduler object from torch.optim.lr_scheduler.
    :param num_epochs:
    :return:
    """
    top1 = AverageMeter()
    top5 = AverageMeter()
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Initialize TensorBoard logger:
    logger = Logger('./TBLogs')
    ittr = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase:
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                # Set the model to training mode (see: http://pytorch.org/docs/master/nn.html#torch.nn.Module.train):
                model.train(True)
            else:
                # Set the model to eval mode (see: http://pytorch.org/docs/master/torchvision/models.html):
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data:
            for data in data_loaders[phase]:
                # get the inputs
                inputs, labels = data
                # print('inputs size: %s' % (inputs.size(),))
                # wrap them in a TensorFlow Variable:
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # Zero the parameter gradients:
                optimizer.zero_grad()

                # Compute the Forward pass:
                outputs = model(inputs)
                _, preds = pt.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # Compute accuracy for TensorBoard logs:
                _, argmax = pt.max(outputs, 1)
                accuracy = (labels == argmax.squeeze()).float().mean()

                # If in the training phase then backpropagate and optimize by taking step in the gradient:
                if phase == 'train':
                    #============ TensorBoard logging ============#
                    # Source: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/main.py
                    # Log the scalar values
                    info = {
                        'loss-Train': loss.data[0],
                        'accuracy-Train': accuracy
                    }

                    step = ittr - 1
                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, step+1)
                    # Log values and gradients of the parameters (histogram)
                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        logger.histo_summary(tag, value.data.cpu().numpy(), step+1)
                        # Since param.requires_grad = False can't log gradient data (transfer learning). Would have to
                        # log gradient data of the last two fc layers only.
                        # logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), setp+1)
                    #============ Backpropagation and Optimization ============#
                    loss.backward()
                    optimizer.step()


                # update loss and accuracy statistics:
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += pt.sum(preds == labels.data)
                # Measure accuracy and record:
                # print('size of outputs.data: %s' % (outputs.data.size(),))
                # print('outputs.data: %s' % outputs)
                # print('size of lables: %s' % labels.size())
                # prec1, prec5 = accuracy(output=outputs, target=labels, topk=(1, 5))
                # top1.update(prec1[0], inputs.size(0))
                # top5.update(prec5[0], inputs.size(0))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]




            print('[{}]:\t Epoch Loss: {:.4f} Epoch Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
            # print('[%s]:\t Epoch Top-5 Error on %sset: %.3f' % (phase, phase, top1.val))

            # deep copy the model's weights if this epoch was the best performing:
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print('Accuracy (Top-1 Error or Precision at 1) of the network on %d %s images: %.2f%%'
              % (dataset_sizes[phase], phase, epoch_acc * 100))

        # top_1_err = get_top_1_error(model=model)
        # print('Overall accuracy (Top-1 Error) of the network on %d %s images: %.2f %%'
        #       % (dataset_sizes['test'] if has_test_set else dataset_sizes['val'],
        #          'test' if has_test_set else 'val',
        #          top_1_err))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):
    """
    visualize_model: Generic function to display the models predictions for a few images.
    :source URL: http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    :param model:
    :param num_images:
    :return:
    """
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    for i, data in enumerate(data_loaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = pt.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow_tensor(inputs.cpu().data[j])

            if images_so_far == num_images:
                model.train(mode=was_training)
                return
    model.train(mode=was_training)


def main():
    """

    :return:
    """
    # Get a batch of training data:
    inputs, classes = next(iter(data_loaders['train']))
    # Make a grid from batch:
    out = torchvision.utils.make_grid(inputs)
    # Display several training images:
    imshow_tensor(input=out, title=[class_names[x] for x in classes])
    # Load a pre-trained model:
    resnet_18 = models.resnet18(pretrained=True)
    source_model = 'resnet_18'
    # model_pretrained_accuracies_url = 'http://pytorch.org/docs/master/torchvision/models.html'
    print('Loaded %s source model pre-trained on ImageNet.' % source_model)
    print('The initial error rates for the %s model with 1-crop (224 x 224) on the entire ImageNet database are as follows:'
          '\n\tTop-1 error: 30.24%%'
          '\n\tTop-5 error: 10.92%%' % source_model)
    # Freeze all of the network except the final layer (as detailed in Going Deeper in the Automated Id. of Herb. Spec.)
    for param in resnet_18.parameters():
        param.requires_grad = False
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = resnet_18.fc.in_features
    resnet_18.fc = nn.Linear(num_ftrs, 2)
    if use_gpu:
        resnet_18 = resnet_18.cuda()
    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized:
    optimizer_conv = optim.SGD(resnet_18.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay the learning rate by a factor of 0.1 every 7 epochs:
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer_conv, step_size=7, gamma=0.1)

    # Train and evaluate:
    resnet_18 = train_model(model=resnet_18, criterion=criterion, optimizer=optimizer_conv,
                               scheduler=exp_lr_scheduler, num_epochs=25)


if __name__ == '__main__':
    print('WARNING: Future Warnings are set to Ignore.')
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.resetwarnings()
    # warnings.filterwarnings('once', FutureWarning)
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
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_load_size),
            transforms.CenterCrop(receptive_field_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    # Training set data loader:
    trainset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
    # Validation set data loader:
    valset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
    image_datasets = {
        'train': trainset,
        'val': valset,
    }
    # Does a test set directory exist, or only a validation set?
    has_test_set = os.path.isdir(os.path.join(data_dir, 'test'))
    if has_test_set:
        # Test set data loader:
        testset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
        image_datasets['test'] = testset
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    #                   for x in ['train', 'val']}
    # Four async threads per data loader:
    num_workers = 4
    # Shuffle the data as it is loaded:
    shuffle = True
    # How many images the data loader grabs during one call to next(iter(data_loader)):
    batch_sizes = {'train': 5, 'val': 5, 'test': 6}
    # Instantiate training dataset loader:
    train_loader = pt.utils.data.DataLoader(trainset, batch_size=batch_sizes['train'], shuffle=shuffle, num_workers=num_workers)
    print('Training data loader instantiated with:'
          '\n\tshuffle data: %s'
          '\n\tnumber of workers (async threads): %d'
          '\n\tbatch size (during iteration):%d'
          % (shuffle, num_workers, batch_sizes['train']))
    # Instantiate validation dataset loader:
    val_loader = pt.utils.data.DataLoader(valset, batch_size=batch_sizes['val'], shuffle=shuffle, num_workers=num_workers)
    print('Validation data loader instantiated with:'
          '\n\tshuffle data: %s'
          '\n\tnumber of workers (async threads): %d'
          '\n\tbatch size (during iteration):%d'
          % (shuffle, num_workers, batch_sizes['val']))
    # data_loaders = {x: pt.utils.data.DataLoader(image_datasets[x], batch_size=5, shuffle=True, num_workers=4)
    #                 for x in ['train', 'val']}
    data_loaders = {
        'train': train_loader,
        'val': val_loader,
    }
    if has_test_set:
        # Instantiate testing dataset loader:
        test_loader = pt.utils.data.DataLoader(valset, batch_size=batch_sizes['test'], shuffle=shuffle, num_workers=num_workers)
        print('Testing data loader instantiated with:'
              '\n\tshuffle data: %s'
              '\n\tnumber of workers (async threads): %d'
              '\n\tbatch size (during iteration):%d'
              % (shuffle, num_workers, batch_sizes['test']))
        data_loaders['test'] = test_loader

    dataset_sizes = {x: len(image_datasets[x]) for x in list(image_datasets.keys())}
    print('Number of Images in Each Dataset: %s' % dataset_sizes)
    class_names = image_datasets['train'].classes
    print('All class labels in the dataset: %s' % class_names)
    use_gpu = pt.cuda.is_available()
    print('CUDA is enabled?: %s\nWill use GPU to train?: %s' % (use_gpu, use_gpu))
    main()
