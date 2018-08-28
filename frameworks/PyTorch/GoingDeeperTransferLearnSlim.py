import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import pandas as pd
import os
from torch.autograd import Variable
import time
import copy
import shutil

# Hyper Parameters
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Transfer Learning Demo on Going Deeper Herbaria 1K Dataset')
parser.add_argument('STORE', metavar='DIR', help='Data storage directory.')
parser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true',
                    help='Enable verbose print statements (yes, no)?')
parser.add_argument('--arch', '-a', metavar='ARCH', default='inception_v3', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: inception_v3)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
# parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
#                     metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


def get_data_loaders_and_properties():
    """
    get_data_loaders: Creates either two or three instances of torch.utils.data.DataLoader depending on the datasets
        present in the storage directory provided via command line argument 'args.STORE' at runtime. Instantiates and
        returns a DataLoader for the training dataset, test dataset, and validation dataset (if present).
    :returns data_loaders, data_props:
        :return data_loaders: A dictionary of torch.utils.DataLoader instances.
        :return data_props: A dictionary of properties relating to the data sets comprised of the following:
            :return has_test_set: A boolean variable indicating if a test set folder is present (DataLoader context).
            :return has_train_set: A boolean variable indicating if a training set folder is present (DataLoader context).
            :return has_val_set: A boolean variable indicating if a validation set folder is present (DataLoader context).
            :return class_names['train']: A list of the class names (folder names in training directory).
            :return class_names['test']: A list of the class names (folder names in the testing directory).
            :return num_classes['train']: The number of classes in the training dataset.
            :return num_classes['test']: The number of classes in the testing dataset.
            :return dataset_sizes['train']: The number of samples in the training dataset.
            :return dataset_sizes['test']: The number of samples in the testing dataset.
    """
    ''' Specified in the Research Paper: '''
    img_pxl_load_size = 1024
    receptive_field_pxl_size = 299
    # How many images the DataLoader will grab during one call to next(iter(data_loader)):
    batch_sizes = {'train': 128, 'test': 128}
    ''' Hyperparameters specified by me: '''
    # Declare number of asynchronous threads per data loader (I chose number of CPU cores):
    num_workers = 6
    shuffle = True
    pin_memory = True
    '''
    Training Data and Validation Data Input Pipeline:
        Data Augmentation and Normalization as described here: http://pytorch.org/docs/master/torchvision/models.html
    '''
    # train_pop_means, test_pop_means, train_pop_std_devs, test_pop_std_devs = \
    #     get_image_channel_means_and_std_deviations(df_train=df_train, df_test=df_test)
    # print('train_pop_mean shape: %s' % np.array(train_pop_means).shape)
    # train_img_pop_means = [0.74535418, 0.70882273, 0.61583241]
    # train_img_pop_std_devs = [0.04480927, 0.04685673, 0.05492202]
    # See: https://pytorch.org/docs/stable/torchvision/models.html
    train_img_pop_means_imgnet = [0.485, 0.456, 0.406]
    train_img_pop_std_devs_imgnet = [0.229, 0.224, 0.225]
    data_transforms = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_pxl_load_size),
            torchvision.transforms.CenterCrop(receptive_field_pxl_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(train_img_pop_means_imgnet, train_img_pop_std_devs_imgnet)
        ]),
        'test': torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_pxl_load_size),
            torchvision.transforms.CenterCrop(receptive_field_pxl_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(train_img_pop_means_imgnet, train_img_pop_std_devs_imgnet)
        ])
    }
    data_loaders = {}
    data_props = {'class_names': {}, 'num_classes': {}, 'num_samples': {}}

    # Training set image folder:
    if os.path.isdir(args.STORE + '\\images\\train'):
        data_props['has_train_set'] = True
        train_img_folder = torchvision.datasets.ImageFolder(args.STORE + '\\images\\train',
                                                            transform=data_transforms['train'])
        # train_img_dataset = HerbariumDataset(data_dir=args.STORE + '\\images\\train', transform=data_transforms['train'], extensions=IMG_EXTENSIONS)
        # Classes present in the training image set:
        data_props['class_names']['train'] = train_img_folder.classes
        # Number of classes present in the training image set:
        data_props['num_classes']['train'] = len(train_img_folder.classes)
        # Number of samples present in the training image set:
        data_props['num_samples']['train'] = len(train_img_folder)
        # Instantiate the training dataset DataLoader:
        train_loader = torch.utils.data.DataLoader(train_img_folder, batch_size=batch_sizes['train'], shuffle=shuffle,
                                            num_workers=num_workers, pin_memory=pin_memory)
        data_loaders['train'] = train_loader
        if args.verbose:
            print('Training data loader instantiated with:'
                  '\n\tshuffle data: %s'
                  '\n\tpin memory: %s'
                  '\n\tnumber of workers (async threads): %d'
                  '\n\tbatch size (during iteration):%d'
                  % (shuffle, pin_memory, num_workers, batch_sizes['train']))
    else:
        data_props['has_train_set'] = False

    # Testing set image folder:
    if os.path.isdir(args.STORE + '\\images\\test'):
        data_props['has_test_set'] = True
        test_img_folder = torchvision.datasets.ImageFolder(args.STORE + '\\images\\test',
                                                           transform=data_transforms['test'])
        # test_img_dataset = HerbariumDataset(data_dir=args.STORE + '\\images\\test', transform=data_transforms['test'], extensions=IMG_EXTENSIONS)
        # Classes present in the testing image set:
        data_props['class_names']['test'] = test_img_folder.classes
        # Number of classes present in the testing image set:
        data_props['num_classes']['test'] = len(test_img_folder.classes)
        # Number of samples present in the testing image set:
        data_props['num_samples']['test'] = len(test_img_folder)
        # Instantiate the testing dataset DataLoader:
        test_loader = torch.utils.data.DataLoader(test_img_folder, batch_size=batch_sizes['test'], shuffle=shuffle,
                                               num_workers=num_workers, pin_memory=pin_memory)
        data_loaders['test'] = test_loader
        if args.verbose:
            print('Testing data loader instantiated with:'
                    '\n\tshuffle data: %s'
                    '\n\tpin memory: %s'
                    '\n\tnumber of workers (async threads): %d'
                    '\n\tbatch size (during iteration):%d'
                    % (shuffle, pin_memory, num_workers, batch_sizes['test']))
    else:
        data_props['has_test_set'] = False
    return data_loaders, data_props


def save_checkpoint(state, is_best, file_path='../../data/PTCheckpoints/checkpoint.pth.tar'):
    """
    save_checkpoint: Saves a checkpoint representing the model's process during training, which can be loaded resuming
        the training process at a later time.
    :param state: A dictionary that contains state information crucial to resuming the training process.
        state['epoch']: The current epoch during training.
        state['arch']: The architecture of the model (inception_v3, resnet18, etc...).
        state['state_dict]: A deepcopy of the model's state dictionary (model.state_dict()). For more information see:
            https://pytorch.org/docs/stable/nn.html#torch.nn.Module.state_dict
        state['best_acc']: The best top-1-accuracy at this point in the model's training.
        state['optimizer']: A deep copy of the optimizer's state dictionary.
    :param is_best: A boolean flag indicating if this particular checkpoint is the best performing in the model's
        history.
    :param file_path: The path to which the saved checkpoint is to be written.
    :return:
    """
    torch.save(state, file_path)
    if is_best:
        shutil.copyfile(file_path, 'model_best.pth.tar')


# Data Sets:
args = parser.parse_args()
use_gpu = torch.cuda.is_available()
# Load the dataframes into memory from the hard drive:
df_train = pd.read_pickle(args.STORE + '\\images\\df_train.pkl')
df_test = pd.read_pickle(args.STORE + '\\images\\df_test.pkl')
if not df_train.empty and not df_test.empty:
    print('Loaded both training and testing metadata into memory. Images already physically partitioned on HDD.')
else:
    print('Could not load either df_train or df_test with data. Exiting...')
    exit(-1)
data_loaders, data_props = get_data_loaders_and_properties()

if __name__ == '__main__':
    # Load the model:
    pretrained_model = True
    model = models.__dict__[args.arch](pretrained=pretrained_model)

    # Loss and Optimizer:
    # Freeze the entire network except for the final two layers (as detailed in Going Deeper in Automated Id.)
    # Parameters of newly constructed modules have requires_grad=True by default:
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    # Use training classes because we can't predict what the network hasn't seen:
    model.fc = nn.Linear(in_features=num_ftrs, out_features=data_props['num_classes']['train'], bias=True)

    # NOW:
    model = model.cuda()

    # Criterion:
    criterion = nn.CrossEntropyLoss()
    # Optimizer:
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=args.lr)

    # Train the model:
    num_epochs = 1

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_num_correct = 0

        for i, (images, labels) in enumerate(data_loaders['train']):
            images = Variable(images).cuda(async=True)
            labels = Variable(labels).cuda(async=True)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            _, preds = torch.max(outputs[0].data, 1)
            loss = criterion(outputs[0], labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_num_correct += torch.sum(preds == labels.data).item()

            if (i+1) % 2 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                      % (epoch+1, num_epochs, i+1, data_props['num_samples']['train']//data_loaders['train'].batch_size, loss.data[0]))

        epoch_loss = running_loss / data_props['num_samples']['train']
        losses.append(epoch_loss)
        epoch_acc = running_num_correct / data_props['num_samples']['train']
        accuracies.append(epoch_acc)

        print('[{}]:\t Epoch Loss: {:.4f} Epoch Acc: {:.4f}'.format(
                'train', epoch_loss, epoch_acc))

        # deep copy the model's weights if this epoch was the best performing:
        if epoch_acc >= best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print('Checkpoint: This epoch had the best accuracy. Saving the model weights.')
            # create checkpoint:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': copy.deepcopy(model.state_dict()),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()
            }, is_best=True, file_path='../../data/PTCheckpoints/model_best.pth.tar')

    print('==' * 15 + 'Finished Training' + '==' * 15)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # Test the model after training:
    # model.eval()
    # correct = 0
    # total = 0
    # for images, labels in data_loaders['test']:
    #     images = Variable(images).cuda()
    #     outputs = model(images)
    #     _, predicted = torch.max(outputs.data, 1)
    #     total += labels.size(0)
    #     correct += (predicted.cpu() == labels).sum()

    # print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

