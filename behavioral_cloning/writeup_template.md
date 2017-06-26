# Behavioral Cloning Project**


[//]: # (Image References)

[image1]: ./examples/Nvidia_neural_net.png "Model Visualization"
[image2]: ./examples/original_image.jpg "Original Image"
[image2]: ./examples/cropped_img.jpg "Cropped Image"
[image4]: ./examples/flipped_img.png "Flipped Image"
[image5]: ./examples/left_image.png "Left Recovery Image"
[image6]: ./examples/right_image.png "Right Recovery Image"

### Rubric Points
---


My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* Video is uploaded [here](https://youtu.be/Z0lQ7I6sfYQ)

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```


The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy
Model is based on NVidia's [deep learning paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
Model used : ![Model][image1]

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3 5x5 and 1 3x3 filter sizes and depths between 24 and 64 (model.py lines 28-42)

The model includes cropping of sky and bottom non road pixels (code line 26)

The model includes RELU layers to introduce nonlinearity (code line 30), and the data is normalized in the model using a Keras lambda layer (code line 22). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 29, 31, 33, 39, 41, 48, 51, 54). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 153-154). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 151).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road using the left and right camera images.

I also restricted the number of examples that drive straight since they were overrepresented and model was initially biased towards staying straight. I also randomly added variable brightness to the images so it would generalize better.

I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Here are some of the example image:
Original Image : ![Model][image2]
Cropped Image : ![Model][image3]
Flipped Image : ![Model][image4]
Left Recovery Image : ![Model][image5]
Right Recovery Image : ![Model][image6]


