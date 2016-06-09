# Video Segmentation
Could a computer model differentiate between television commercials and shows?  This project uses a convolutional neural network (CNN) to learn features which correspond to the two types of content. The following work forms the foundation of a system which could automatically filter commercials.

## Table of Contents
1. [Data](#the-data)
2. [Convolutional Neural Network Model](#convolutional-neural-network-model)
  * [Training](#training)
  * [Architecture](#architecture)
  * [Results](#results)
3. [Next Steps](#next-steps)

## Data
80 clips of television shows and 200 commercials were scrapped from Youtube.  A total of 60,000 images - 25K commercial frames and 35K television - were extracted from the videos and used to train the neural network.  Below are some example images of frames taken from the videos.  

![Image](/images/show_commercials.jpg)

Could you pick out which images are commercials and which are television shows?  From let to right: show, commercial, commercial, show.

## Convolutional Neural Network Model
Why use a CNN for this task instead of a vector or tree based method?  CNNs are the most flexible tool for image recognition.  They exhibit properties such as translational invariance that make them well suited for this classification task.

### Training
The training of the model was accomplished with Keras and Theano on an AWS GPU instance.

### Architecture

### Results
TBD
## Next Steps
- Experiment with optical flow
- Experiment with transfer learning
- Segment images
- Transition to tensor flow (distributed training)
