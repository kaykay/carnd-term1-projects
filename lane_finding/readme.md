## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort.png "Undistorted"
[image2]: ./examples/undistorted_warped.jpg "Road Transformed"
[image3]: ./examples/thresholded.png "Binary Example"
[image4]: ./examples/warped_before_after.png "Warp Example"
[image5]: ./examples/polyfit.png "After polyfitting"
[image6]: ./examples/projection_onto_lane.png "Projection onto lane"
[image7]: ./examples/points_for_warping.png "Points used for warping"

---

### Writeup / README


### Camera Calibration

#### 1. I used chess board images provided in the class to perform the calibration and correct the distortion, code for this is in undistort.py file.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![distortion correction][image1]

#### 2. Thresholding

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 14 through 40 in `thresholding.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![Thresholding][image3]

#### 3. Perspective transform

The code for my perspective transform includes a function called `unwarp_params()`, which appears in lines 16 through 46 in the file `transform.py`.  The `unwarp_params()` functions takes as inputs an image (`img`), as well as distortion parameters mtx and dist.  I chose to hardcode the source and destination points in the following manner:

```python
src = np.float32([[575, 462],[708.7, 462],[1062.4, 686.4],[255.6, 686.4]])
offsetx = 250
dest = np.float32([[offsetx, 0], [1280 - offsetx, 0], [1280 - offsetx, 720],[offsetx, 720]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![warped image][image4]

#### 4. I used the sliding window search method by applying convolution as described in class to find lane line pixels, code for this can be found in finding_lines.py from line 29 through 68

Then i used gathered x points for left and right lanes along the sliding windows found and fit second order polynomial.

Thresholded image : ![Image][image3]

Detected lanes and Polynomial fit of the thresholded image : ![Polynomial fit][image5]


#### 5. I used [measuring curvature](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) as described there to find the curvature in meters from the polynomial fit of the lines, code is in finding_lanes.py from line 161 through 175. I also found the distance between the lanes on 142 through 143.

#### 6. Plotting back the fitted lane onto original image

I implemented this step in lines 181 through 199 in my code in `finding_lines.py` .  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Link to new submission [final video](https://youtu.be/ocXSjyDup5w)
#### 2. Link to previous submission [final video](https://www.youtube.com/watch?v=pAWVu4iSTS4)
---

### Discussion

#### 1. Detecting failures : I didn't get around to detecting failures robustly, i did a very simple check to see if the width of lanes falls into certain known range before drawing onto the road, more robust methods need to be used such as maintaining average of previous n frames, thresholding on curvatures, etc..
