import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from undistort import cal_undistort, read_calibration_params
# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()

warp_params_file = "camera_cal/wide_warp_pickle.p"

mtx, dist = read_calibration_params()

# Get perspective M warp parameter and src, dest coordinates from straight line image.
def unwarp_params(undist, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    fig,ax = plt.subplots(1)
    ax.imshow(undist)
    
    gray = cv2.cvtColor(undist,cv2.COLOR_BGR2GRAY)
    imgshape = gray.shape[::-1]
    offset = 50
    #    dest = np.float32( [[1060, 450], [1060, 670], [270, 670], [270, 450]])
    src = np.float32([[575, 462],[708.7, 462],[1062.4, 686.4],[255.6, 686.4]])
    ax.plot(*src[0], '.')
    ax.plot(*src[1], '.')
    ax.plot(*src[2], '.')
    ax.plot(*src[3], '.')
    polygon = mpatches.Polygon(src, True, alpha=0.4, fill=True, color='yellow')
    ax.add_patch(polygon)
    plt.savefig('examples/unwarp_point_selection_straight.jpg')
    plt.show()
    offsetx = 250
    dest = np.float32([[offsetx, 0], [1280 - offsetx, 0], [1280 - offsetx, 720],[offsetx, 720]])
    M = cv2.getPerspectiveTransform(src, dest)
    Minv = cv2.getPerspectiveTransform(dest, src)
    warped = cv2.warpPerspective(undist, M, imgshape)
    
    return warped, M, Minv, src, dest
#Given an undistorted image and perspective_M, src and dest parameters, unwarp the image
def unwarp(img, M, src, dest):
    imgshape = img.shape[1::-1]
    return cv2.warpPerspective(img, M, imgshape)

#Return M, src, dest from stored pickle file
def read_unwarp_params(pickle_file=warp_params_file):    
    data = pickle.load(open(pickle_file, 'rb'))
    return (data['M'], data['Minv'], data['src'], data['dest'])

#If invoked save unwarp parameters in pickle file
#Also, display example images.
if __name__=="__main__":
# Read in an image
    img = mpimg.imread('test_images/straight_lines2.jpg')
    undist = cal_undistort(img, mtx, dist)
    top_down, perspective_M, Minv, src, dest = unwarp_params(undist, mtx, dist)
    #Write to pickle file.
    dist_pickle = {}
    dist_pickle["src"] = src
    dist_pickle["dest"] = dest
    dist_pickle["M"] = perspective_M
    dist_pickle["Minv"] = Minv
    
    pickle.dump( dist_pickle, open(warp_params_file, "wb" ))
    plt.imsave('examples/unwarped_straight.jpg', top_down)

    img = mpimg.imread('test_images/test3.jpg')
    undist = cal_undistort(img, mtx, dist)
    top_down = unwarp(img, perspective_M, src, dest)
    plt.imsave('examples/unwarped_curved.jpg', top_down)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(top_down)
    ax2.set_title('Undistorted and Warped Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
