#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/signs_distribution.png "Visualization"
[image2]: ./examples/before_grayscale.png "Grayscaling"
[image3]: ./examples/after_grayscale.png "Grayscaling"
[image4]: ./examples/sign_grayscaled_normalized.png "Random Noise"
[image5]: ./examples/newsigns.png "Unseen Traffic Signs"

---


####1.  

###Data Set Summary & Exploration

####1.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

####2. visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed between all classes

![alt text][image1]

###Design and Test a Model Architecture

####1. Data

As a first step, I decided to convert the images to grayscale because i think the color information won't add any value to classification, also to reduce the size of feature set.

Here is an example of a traffic sign image before and after grayscaling.

Before : ![alt text][image2]
After : ![alt text][image3]

As a last step, I normalized the image data so the optimizer converges faster.

I haven't generated additional data.


####2. I used model similar to Lenet

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x48				|
| Dropout               | keep probability 0.5
| Convolution 5x5	    | 1x1 stride, Valid padding, outputs 10x10x24      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x48				|
| Flatten	        |  outputs 1200      									|
| Fully connected		| Output = 120       									|
| RELU					|												|
| Dropout               | keep probability 0.5
| Fully connected		| Output = 43       									|
| Softmax				|        									|
|						|												|
|						|												|
 


####3. I trained it with reduce_mean loss function, used Adam optimizer with a learning rate of 0.001. I used a batchsize of 3000 and trained the network for 200 epochs.


####4. I started with a simple 3 layer architecture but it wasn't converging on a solution, i decided to use LetNet implementation and added 2 extra dropout layers because the model wasn't generalizing on new samples.


My final model results were:
* validation set accuracy of 95.5 
* test set accuracy of 95.2

###Test a Model on New Images

####1. Training new images

Here are five German traffic signs that I found on the web:

Images used and its predictions:
![alt text][image5] 

The accuracy was pretty good, it only got 5 of the 6 images right, the 1 it didn't get right is very close to the other prediction

####2.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No entry      		| No entry   									| 
| Stop Sign      		| Stop sign   									| 
| Yield					| Yield											|
| U-turn     			| U-turn 										|
| Turn right ahead					| Turn right ahead										|
| 80 km/h	      		| 50 km/h					 				|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .45         			|  No entry   									| 
| .31     				| Stop sign 										|
| .30					| Yield											|
| .34	      			| U-Turn				 				|
| .28				    | Turn right ahead      							|
| .32				    | 80 km/h      							|




### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)


