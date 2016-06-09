# Video Segmentation
Could a computer model differentiate between television commercials and shows?  This project uses a convolutional neural network (CNN) to learn features which correspond to the two types of content. The following work forms the foundation of a system which could automatically filter commercials.

## Table of Contents
1. [The Data](#the-data)
2. [The CNN Model](#the-cnn-model)
  * [Training](#training)
  * [Architecture](#architecture)
  * [Results](#results)
3. [Next Steps](#next-steps)

## The Data
80 clips of television shows and 200 commercials were scrapped from Youtube.  A total of 60,000 images - 25K commercial frames and 35K television - were extracted from the videos and used to train the neural network.  Below are some example images of frames taken from the videos.  

![Image](/images/show_commercials.jpg)

Could you pick out which images are commercials and which are television shows?  From let to right: show, commercial, commercial, show.

## The CNN Model
Why use a CNN for this task instead of a vector or tree based method?  CNNs are the most flexible tool for image recognition.  They exhibit properties such as translational invariance that make them well suited for this classification task.

### Training
The training of the model was accomplished with Keras and Theano on an AWS GPU instance.  Despite training on a GPU optimized EC2 instance and using a fairly simple architecture (see below), the time needed to train the model on the full image resolution was prohibitively expensive.   Therefore, the images were re-sized to 100x100 pixels.

### Architecture
Layers in the squential CNN model are listed below with additional details.
- Convolutional (64 filters of 3x3 height/width)
- Convolutional (32 filters of 3x3 height/width)
- Pooling (2x2 height/width)
- Convolutional (32 filters of 3x3 height/width)
- Pooling (2x2 height/width)
- Dense (128)
- Softmax output

Additional detail of the model architecture can be found in the file: model_baseline_reg.py

### Results
The model results were encouraging.  The test accuracy and recall rate were 95%.

## Next Steps
- Try and capture time related features with optical flow
- Perhaps transfer learning with imageNet would provide additional gains
- Segment images
- Transition to tensor flow for distributed training
