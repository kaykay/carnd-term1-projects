import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
from undistort import cal_undistort, read_calibration_params
from thresholding import pipeline as threshold
from transform import unwarp, read_unwarp_params

image = mpimg.imread('test_images/test3.jpg')
#undistort
mtx, dist = read_calibration_params()
undistort = cal_undistort(image, mtx, dist)
#warp
perspective_M, Minv, src, dest = read_unwarp_params()
warped = unwarp(undistort, perspective_M, src, dest)
#threshold
binary_warped = threshold(warped)
binary_warped = binary_warped * 255
plt.imshow(binary_warped, cmap='gray')
plt.show()


def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(warped, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(warped.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
        if (conv_signal[l_min_index:l_max_index] > 5.).sum() > 0:
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
        #Only update center if light pixels are detected
        if (conv_signal[r_min_index:r_max_index] > 5.).sum() > 0:
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        
	# Add what we found for that layer
        window_centroids.append((l_center,r_center))

    return window_centroids



mtx, dist = read_calibration_params()
perspective_M, Minv, src, dest = read_unwarp_params()
font = cv2.FONT_HERSHEY_SIMPLEX

Prev_result = [None]

import queue
prev_poly = queue.Queue(4)

def process_img(image):
    undistort = cal_undistort(image, mtx, dist)
    #warp
    warped = unwarp(undistort, perspective_M, src, dest)
    #threshold
    binary_warped = threshold(warped)
    #binary_warped = binary_warped * 255
#    plt.imshow(binary_warped, cmap='gray')
#    plt.show()
    

    # Read in a thresholded image
    warped = binary_warped
    # window settings
    window_width = 50 
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 100 # How much to slide left and right for searching
    window_centroids = find_window_centroids(warped, window_width, window_height, margin)
    
    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows 	
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
    
        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    
        warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
    
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
    
 
    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

    # Display the final results
#    plt.imshow(output)
#    plt.title('window fitting results')
#    plt.show()

    #Fit a polygon
    ploty = np.linspace(0, 719, num=720)
    leftx = np.zeros_like(ploty)
    rightx = np.zeros_like(ploty)
    for level in range((int)(len(leftx) / window_height)):
        leftx[level*window_height:(level + 1) * window_height] = window_centroids[level][0]
        rightx[level*window_height:(level + 1) * window_height] = window_centroids[level][1]

    lane_center = leftx[0] + (rightx[0] - leftx[0]) / 2
    img_center = 1280/2

    bottom_dist = rightx[0] - leftx[0]
    top_dist = rightx[-1] - leftx[-1]
    
    #number of pixels off center
    off_center_in_pixels = img_center - lane_center
    
    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y
    
    xm_per_pix = 3.7/700
    off_center_in_meters = off_center_in_pixels * xm_per_pix
    bottom_dist_metres = bottom_dist *  xm_per_pix
    top_dist_metres = top_dist * xm_per_pix
    
    avg_dist = np.mean(rightx - leftx) * xm_per_pix
    
    print("Average distance in meters: ", np.mean(rightx - leftx) * xm_per_pix)
    left_fit = np.polyfit(ploty, leftx, 2)
    right_fit = np.polyfit(ploty, rightx, 2)

    if (Prev_result[0] != None and (bottom_dist_metres < 3.85 or bottom_dist_metres > 4.15)):
        left_fit, right_fit = Prev_result[0]
    else:
        Prev_result[0] = (left_fit, right_fit)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

#   mark_size = 3
#   plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
#   plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
#   plt.xlim(0, 1280)
#   plt.ylim(0, 720)
#   plt.plot(left_fitx, ploty, color='green', linewidth=3)
#   plt.plot(right_fitx, ploty, color='green', linewidth=3)
#   plt.gca().invert_yaxis()
#   plt.show()
    
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    print(left_curverad, right_curverad)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    
    #Draw image
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undistort, 1, newwarp, 0.3, 0)
    if (off_center_in_meters > 0):
        result = cv2.putText(result,'Vehicle is: ' + str(off_center_in_meters) +  ' left of center', (500,30), font, 1,(255,255,255),2)
    else:
        result = cv2.putText(result,'Vehicle is: ' + str(abs(off_center_in_meters)) +  ' right of center', (500,30), font, 1,(255,255,255),2)
        
    result = cv2.putText(result,'left curvature: ' + str(left_curverad ), (500,60), font, 1,(255,255,255),2)
    result = cv2.putText(result,'Right curvature: ' + str(right_curverad), (500,90), font, 1,(255,255,255),2)
#    result = cv2.putText(result,'bottom dist: ' + str(bottom_dist_metres), (500,120), font, 1,(255,255,255),2)
#    result = cv2.putText(result,'bottom dist: ' + str(top_dist_metres), (500,150), font, 1,(255,255,255),2)

    return result;




result = process_img(image)
plt.imshow(result)
plt.show()

from moviepy.editor import VideoFileClip
output = 'project_video_out.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_img) #NOTE: this function expects color images!!
white_clip.write_videofile(output, audio=False)

