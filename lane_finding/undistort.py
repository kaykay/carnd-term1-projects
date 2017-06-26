import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

dist_file = "camera_cal/wide_dist_pickle.p"
# performs image distortion correction and 
# returns the undistorted image
def cal_undistort(img, mtx, dist):
    # Use cv2.calibrateCamera() and cv2.undistort()
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

#Read the mtx and dist params previously saved from pickle file
def read_calibration_params(file_name=dist_file):
    data = pickle.load(open(file_name, 'rb'))
    return (data['mtx'], data['dist'])

#if this file is executed, comput the distortion
#correction parameters 'mtx', 'dist' and save them
#in pickle file.
if __name__=="__main__":
    plt.ion()
    #File to store learned Distortion parameters
    
	
    # prepare object points
    nx = 9#TODO: enter the number of inside corners in x
    ny = 6#TODO: enter the number of inside corners in y
    
    objpoints = [] #3D Object points
    imgpoints = [] #Image points on 2d plane.
    
    # Make a list of calibration images
    cal_images = glob.glob('camera_cal/calibration*.jpg')
    #Object points
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) #x, y coordinates
    
    for fname in cal_images:
        img = cv2.imread(fname)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
	# If found, draw corners
        if ret == True:
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            # plt.imshow(img)
            objpoints.append(objp) #Same for all images in this set
            imgpoints.append(corners)
    img = cv2.imread('camera_cal/calibration1.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undistorted = cal_undistort(img, mtx, dist)
    #Write to pickle file.
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( dist_file, "wb" ) )
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undistorted)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    plt.waitforbuttonpress()
    print("Wrote calibration parameters to :", dist_file)
    
