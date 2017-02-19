**Behavioral Cloning Project**

###Goals
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

###Files
* model.py 
* drive.py 
* model.h5  
* writeup_report.md

###Model Architecture and Training Strategy

To construct model architecture we need to predict steering angle from the raw pixel data. Because this is a regression problem, final output must be a single continuous value. 

My model consists of a convolution neural network and starts with the model similar to the [Nvidia paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). 
The network consitsts of 5 convolutional layers and 3 fully connected layers. The input is 160x320x3 image and the final output layer is a single value for steering angle.  

The images are preprocessed before convolutions such as cropping, resizing, and normalization.

Each convolutional layer uses 2x2 max pooling and dropout for reducing overfitting.  

ELU activation is used for each convlutional and fully connected layer. 

The output is flattened after final convolution and fed into the fully connected layers. The festureas are reduced and the final output is made by final layer.  


###Data Collection

I firtst record the training data while driving around the track, but I mainly used the data provided from UDACITY. The sample images are captured from center, right, and left camera. 
Just input raw images to the network didn't work. The network didn't converge to good solution. I had to preprocess the image. 

#### Images from multiple cameras

The center image is used for the car to run on the center of the track. The right and left images are used for recovery data after angle correction. I added a collection angle of 0.07 to the left image and -0.07 to the right image.

#### Crop

Irrelevant portions like sky and tree can be cropped for efficient training.  

#### Resize

Because reducing the size of image doesn't affect traininng process, I reduced image size down to 40x160.

#### Normalization

The image has 0-255 RGB value in each pixel. This is a big value for network, so it is normalized by keras.layers.normalization.BatchNormalization. 

#### Flip

I rondomly 50% chosed to flip image along y-axis and invert the steering angle. This can reduce the bias in the steering angles toward particular side.   


###Training

I used Adam optimizer because it tunes the learning rate automatically and we don't have to mannualy input. I used mean squared error for loss function. 
Usually network requires a large quantity of data. To load all of the data into memory exhaust the resources. The Kera data generator function allows to load data split by batch size per time. 