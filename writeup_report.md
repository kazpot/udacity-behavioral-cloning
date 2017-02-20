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

![nvidia](https://github.com/kazsky/UDACITY_Behavioral_Cloning/blob/master/img/nvidia_paper.png)


###Data Collection

I first recorded the training data while driving around the track, but I only used the data provided from UDACITY. The sample images are captured from center, right, and left camera. 
Just input raw images to the network didn't work. The network didn't converge to good solution. I had to preprocess the image. 

![img1](https://github.com/kazsky/UDACITY_Behavioral_Cloning/blob/master/img/original_img.png)

#### Images from multiple cameras

The center image is used for the car to run on the center of the track. The right and left images are used for recovery data after angle correction. I added a collection angle of 0.07 to the left image and -0.07 to the right image.

![img2](https://github.com/kazsky/UDACITY_Behavioral_Cloning/blob/master/img/sample.png)

Without correction angle, most of steering angles are zero as below figure shows. X-axis is range of steering angle in radian and y-axis is count of angles in the range of angle. 

![img3](https://github.com/kazsky/UDACITY_Behavioral_Cloning/blob/master/img/count_per_angle_range_no_augmentation.png)

With correction angle, most of steering angles are still around zero, but distribution becomes better.

![img4](https://github.com/kazsky/UDACITY_Behavioral_Cloning/blob/master/img/count_per_angle_range_augmented.png)

#### Crop

Irrelevant portions like sky and tree were cropped for reducing amount of informatoin and efficient training.  

#### Resize

Because reducing the size of image doesn't affect traininng process, I reduced image size down to 40x160.

#### Normalization

The image has RGB(0-255) value in each pixel. This is a big value for network, so I normalized RGB values using keras.layers.normalization.BatchNormalization. 

#### Flip

I rondomly chosed to flip image along y-axis and invert the steering angle. This reduced the bias in the steering angles toward particular side. 
The track in simulater has more left curves more than rihgt one. I flipped the set of images with 50% probability. 

###Training Process

The set of images was preprocessed by crop, resize, normalization, and filp. It was fed into the training model. 
However this consumes large amount of memory. Hence python image generator was very useful for handling only a single batch size on memory at a time. 
I used Adam optimizer because it tunes the learning rate automatically and we don't have to mannualy input and mean squared error for loss function. 
Once the network is trained with mean squared error close to zero, the model definition and trained weights saved as json and h5 foramt. 

Tuning parameters for Training:

~~~
batch_size = 128
nb_epoch = 5
~~~

I ran the simulator in autonomous mode and started the driving server with the below command.

~~~
python drive.py model.json
~~~

###Output

The driving server keeps sending the predicted steering angle to the car using the network. 
