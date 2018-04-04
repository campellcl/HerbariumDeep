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

##### TensorFlow Slim (TFSlim) Helpful Resources
* [We Need to Go Deeper: A Practical Guide to Tensorflow and Inception](https://medium.com/initialized-capital/we-need-to-go-deeper-a-practical-guide-to-tensorflow-and-inception-50e66281804f)
