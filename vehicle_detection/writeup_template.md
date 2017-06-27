
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car1.png
[image8]: ./examples/car2.png
[image9]: ./examples/noncar1.png
[image10]: ./examples/noncar2.png
[image2]: ./examples/car-and-hog.jpg
[image3]: ./examples/boxes.png
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/headmap.png
[image6]: ./examples/figure_hog_subsampling.png
[image7]: ./examples/initial_train.png
[image8]: ./examples/

---

###Histogram of Oriented Gradients (HOG)

####1. 

The code for this step is contained in hog_sample.py

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

Vehicle : ![Vehicle][image1]
Non Vehicle : ![Non Vehicle][image9]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![Hog features][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I ended up using the parameters from lesson which worked well for the classifier.

####3. classifier using HOG and color features

I trained a linear SVM using HOG and color features(spatial and histogram features). Code for it is in search_classify.py

I searched through the parameters and found best parameters for this model : 

Parameters used: {'kernel': ('linear', 'rbf'), 'C': [0.1, 1, 10, 100]}
Best params:  {'kernel': 'rbf', 'C': 10}
3769.13 Seconds to train SVC...
Test Accuracy of SVC =  0.9944


###Sliding Window Search

####1. I noticed the classifier was detecting partial cars very well, hence i only usee one window size to do the sliding search, i used heat map to combine multiple detections into one. This is implemented in search_classify.py

Windows searched: ![alt text][image3]

####2. 

Ultimately I searched on one using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image6]
---

### Video Implementation

####1. 
Here's a [link to my video result](https://youtu.be/dSou-dsaINE)


####2. To filter out wrong detections and reduce combine all detections of same car, i used cropped the sky to reduce false detections and applied heatmap with a threshold across 4 frames of the video. Code for this can be found in hog_sample.py lines 133 through 146

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  


---

###Discussion

####1. There were a lot of false detections when with the linear kernel, i searched for best fit and found 'rbf' kernel did very well for car detections. Cutting out the sky part helped in reduction of more false detections.

####2. This sliding window model is not really detecting cars further away or on the other side of the freeway, i need to take multiple window sizes into consideration to detect those.


