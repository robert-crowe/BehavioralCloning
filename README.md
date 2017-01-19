# Use Deep Learning to Clone Driving Behavior

Robert Crowe

v1.0 18 Jan 2017

## Overview of Approach

I used the 
[Nvidia paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
as a starting point to design my model.  There were however several details that were left out, and
I found empirically that the model overfit without modifications.  The main change that I made was to
add a dropout layer in the classifier.

I followed the Nvidia approach fairly closely in preprocessing the images.  I cropped them to a similar
window, capturing the road ahead without the other scenary, basically creating a simple region of
interest.  I also converted to YUV, and ended up using all three dimensions after experimenting with only
using the Y dimension.

I used the dataset provided by Udacity to train my model.  I found the car difficult to drive well
in the simulator.  The data was broken into training, validation, and test sets, which were fed 
using a generator.

## Hyperparameter Selection

I started by selecting a learning rate and optimizer pair through experimentation, and found that 
a rate of 0.0001 with an Adamax optimizer seemed to work well.  I selected the batch size based on 
previous experience and experimentation, and found that 64 seemed to be a good compromise between
training performance and overfitting.  I used early stopping based on validation loss to select the
number of epochs.

The regularization value turned out to be a key hyperparameter.  I found L2 to work slightly better
than L1, and narrowed the value to 0.0012 through experimentation.  This allowed a good result without
overfitting, and kept the basic model clean and simple.

## Iterative Refinement

Please see the included spreadsheet (Iterations.xlsx) for a record of the process of iteratively
exploring and refining different options for model structure, preprocessing, and hyperparameter
selection.