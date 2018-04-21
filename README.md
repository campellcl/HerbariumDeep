# HerbariumDeep
Automated herbarium specimen identification via convolutional neural network.
## Installation
Not all necessary data is provided with the repository (due to the size of the datasets).
Before proceeding be sure to download the compressed zipped data from the following source
URLs and move the extracted folders to the HerbariumDeep/data/ folder.

Required External Data:
1. [Herbarium255_Images](http://otmedia.lirmm.fr/LifeCLEF/GoingDeeperHerbarium/Herbaria255_Images.zip)

### TensorFlow Installation

#### TensorFlow Slim (TFSlim) Installation
1. Follow the instructions listed
[here](https://github.com/tensorflow/models/tree/master/research/slim).
    * When specifying a `$DATA_DIR` be sure to use:
    `$DATA_DIR=../../data/TFRecords/<dataset_name>`

## Resources
### TensorFlow Resources:
#### TensorFlow Slim (TFSlim) Helpful Resources:
* [We Need to Go Deeper: A Practical Guide to Tensorflow and Inception](https://medium.com/initialized-capital/we-need-to-go-deeper-a-practical-guide-to-tensorflow-and-inception-50e66281804f)

### PyTorch Resources:
#### Using TensorBoard in PyTorch:
* [PyTorch Hack - Use TensorBoard for plotting Training Accuracy and Loss](https://beerensahu.wordpress.com/2018/04/18/pytorch-hack-use-tensorboard-for-plotting-training-accuracy-and-loss/)
    * [Referenced Repository](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard)

### GoogLeNet (Inception v1) Resources:
* [Inception v3](https://research.googleblog.com/2016/03/train-your-own-image-classifier-with.html)
#### GoogLeNet TensorFlow Implementation Tutorials:
* [Inception Modules Explained and Implemented](https://hacktilldawn.com/2016/09/25/inception-modules-explained-and-implemented/)

#### GoogLeNet PyTorch Implementation Tutorials:

#### GoogLeNet TensorFlow Implementations:
* [Google Inception Models](https://github.com/khanrc/mnist/blob/master/inception.py)
#### GoogLeNet PyTorch Implementations:
* [Inception v1](https://github.com/antspy/inception_v1.pytorch)

## Datasets:
#### MNIST Handwritten Digits Dataset:
* [MINST LeCun](http://yann.lecun.com/exdb/mnist/)
* [MNIST in CSV Form](https://pjreddie.com/projects/mnist-in-csv/)
#### ImageNet LSVRC Training and Validation Sets:
* [Academic Torrents](http://academictorrents.com/browse.php?search=imagenet)
