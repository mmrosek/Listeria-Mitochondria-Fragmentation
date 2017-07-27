#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 15:25:31 2017

@author: mmrosek
"""

from skimage.filters import threshold_otsu, rank, threshold_local
import imageio
import skimage.filters as filters
import skimage.morphology as morphology
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import cv2
import skimage
from skimage import segmentation
import time

### Read in video file and convert frames to RGB ###
def read_in_video(video_path, color, max_frames=np.inf):
    # Read in video
    reader = imageio.get_reader(str(video_path))
    # Generates dictionary containing information about video
    info_dict = reader.get_meta_data()
    height, width = info_dict['size']
    # Determine number of frames to extract
    if max_frames < np.inf:
        nframes = max_frames
    else:
        nframes = info_dict['nframes']
    if color in "RGBrgb": 
        # Pre-allocates array for video file to populate with rgb frames
        video_array = np.zeros(shape=(nframes, height, width, 3), dtype=np.uint8)
        # Populate video_array with video frames
        for idx, im in enumerate(reader):
            video_array[idx] = im
            if idx >= nframes-1:
                break       
    else:
        video_array = np.zeros(shape=(nframes, height, width), dtype=np.uint8)
        for idx, im in enumerate(reader):
            # Converts rgb frames to grayscale
            video_array[idx] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            if idx >= nframes-1:
                break
    return(video_array)

#file_list = ['DsRed2-HeLa_2_21_LLO (Converted).mov','DsRed2-HeLa_3_15_LLO1part1 (Converted).mov','DsRed2-HeLa_3_15_LLO2part1 (Converted).mov','DsRed2-HeLa_3_1_LLO (Converted).mov',
#'DsRed2-HeLa_3_23_LLO_1 (2) (Converted).mov','DsRed2-HeLa_3_31_LLO_1 (2) (Converted).mov',
#'DsRed2-HeLa_3_31_LLO_2 (2) (Converted).mov','DsRed2-HeLa_3_31_LLO_3 (2) (Converted).mov',
#'DsRed2-HeLa_3_8_LLO002 (Converted).mov',  'DsRed2-HeLa_3_9_LLO (Converted).mov',
#'DsRed2-HeLa_4_5_LLO1 (Converted).mov','DsRed2-HeLa_4_5_LLO2 (Converted).mov']

#video = file_list[0]

video = 'DsRed2-HeLa_3_31_LLO_3 (2) (Converted).mov_cut_by_10.mov'

offset = -0.2

# Read in video file
video_frames = read_in_video("/home/mmrosek/Documents/shannon_vids/{}".format(video), "gray")

####################################################################################################

def convex_hull_and_thresh_imgs(image, block_size_img_to_keep, dilation_disk_size, median_disk_size):

    ret , thresh_hull = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    adaptive_thresh_to_keep = threshold_local(image, block_size_img_to_keep, offset = offset) # Offset = 0 results in large portion of image being white
    thresh_to_keep = image > adaptive_thresh_to_keep
    
    # Dilation
    dilation_disk = morphology.disk(dilation_disk_size)
    new_mask = morphology.binary_dilation(thresh_hull, selem = dilation_disk)
        
    # Median filtering --> maybe cut?
    median_filtering_disk = morphology.disk(median_disk_size)
    filtered = np.array(filters.median(new_mask, selem = median_filtering_disk), dtype = np.bool)
    
    # Removing small objects
    new_mask = morphology.remove_small_objects(filtered, min_size = 512, connectivity = 8)
    
    # Computing convex hull
    hull = morphology.convex_hull_object(new_mask)

    return(hull,thresh_to_keep)

####################################################################################################

# Maybe try shittier threshold for first frame when cant segment?


### Frame-to-frame cell segmentation ###



def overlap(hull_prev_frame, hull_curr_frame):
    
    return ( np.around( ( ( ( np.sum(hull_prev_frame) - np.sum(hull_curr_frame ^ hull_prev_frame) ) / np.sum(hull_prev_frame) ) ) ) )


segmentation_vid_save_path = '/home/mmrosek/Documents/segmented_cell_vid_arrays/' 
dict_save_path = segmentation_vid_save_path
frame_num = 0
block_size_img_to_keep = 25

dilation_disk_size_list = [6.9, 6.9, 6.9]

median_disk_size_list = [7,7,7]

hull_dict_correct = {}

hull_dict_new = {}

hull_prev_frame = 1
labels_prev_frame = 1
   
num_labels_first_frame = 3 # Indicate how many cells to segment
num_labels_prev_frame = num_labels_first_frame


first_frame = 0

initial_hull_overlap_threshold = 0.9

post_zero_hull_overlap_threshold = 0.9



#new_disk_size_hull_overlap_threshold = 0.9

start = time.time()

for label_num in range(num_labels_first_frame):
    
    initial_label_num = label_num
    
    dilation_disk_size = dilation_disk_size_list[initial_label_num]
    
    median_disk_size = median_disk_size_list[initial_label_num]
    
    broken_frame_dict = {}
    
    print('Label Num: {}'.format(label_num+1))
    
    # Initializing array for hull_i+1
    label_vid_array = np.zeros(shape = video_frames.shape, dtype = 'uint8')
       
    for frame_num in range(first_frame,video_frames.shape[0]):
        
        if frame_num % 100 == 0:
            print('Frame: {}'.format(frame_num))
          
        # Calculate convex hull and thresholded image of current frame
        hulls_curr_frame, thresh_img_curr_frame = convex_hull_and_thresh_imgs(video_frames[frame_num], block_size_img_to_keep, dilation_disk_size , median_disk_size)
        
        # Generating labels for convex hulls of current frame
        labels_curr_frame, num_labels_curr_frame = skimage.measure.label(hulls_curr_frame, return_num = True)
        
        
        
        if num_labels_curr_frame == num_labels_prev_frame: # If we have the same number of labels in the curr frame as prev frame...
        
        
            if ( frame_num == first_frame ):
                
                overlap_list = [1]

            
            else:
        
                for curr_frame_label_num in range(num_labels_curr_frame): # Saving each segmented hull in hull_dict
                    
                    hull_dict_new[curr_frame_label_num+1] = labels_curr_frame == curr_frame_label_num+1
                
                overlap_list = [overlap(hull_dict_correct[i+1], hull_dict_new[i+1]) for i in range(num_labels_curr_frame)] 
            
            
            # Checking to ensure overlap of hull_i+1 is > 80% hull_i+1 from previous frame
            if ( frame_num == first_frame ) or ( np.min(overlap_list) > initial_hull_overlap_threshold )  : # Need to make 0.95 an argument to function
                     
                if frame_num % 5 == 0:
            
                    print( frame_num, label_num, overlap, dilation_disk_size )
            
                # Isolate part of thresholded image overlapped by hull_i+1
                hull_segmentation_curr_frame = hull_dict_correct[label_num+1] * thresh_img_curr_frame
            
                # Add segmented part of thresholded image to new array
                label_vid_array[frame_num] = hull_segmentation_curr_frame
                
                hull_dict_correct = hull_dict_new
                
                # Saving array of labels in current frame for use in hull calculation of future frame(s)
                labels_prev_frame = labels_curr_frame
                
                # Saving number of labels in current frame for use in hull calculation of future frame(s)
                num_labels_prev_frame = num_labels_curr_frame
    
                # Saving array of hull of interest in current frame for use in hull calculation of next frame
                hull_prev_frame = hull_curr_frame
        
       
        
        else: # If one or more of the hulls doesn't meet the threhsold....
            
            print( 'Overlap didnt pass threshold: {}'.format( overlap_list ) )
            
            
            ### Solution when can't segment one of a pair of cells ###

            frame_copy = video_frames[frame_num].copy()
            
            for label_num_to_keep in range(num_labels_curr_frame):

                labels_to_zero = list(range(1,num_labels_prev_frame+1))
    
                labels_to_zero.remove(curr_label_num)

            # Creates boolean array indicating which locations in labels_prev_frame are contained in labels_to_zero
            frame_copy_mask = np.in1d(labels_last_correct_frame, labels_to_zero).reshape(labels_last_correct_frame.shape)

            frame_copy[frame_copy_mask] = 0
            
            hulls_curr_frame, thresh_img_curr_frame = convex_hull_and_thresh_imgs(frame_copy, block_size_img_to_keep, dilation_disk_size , median_disk_size)
        
            labels_curr_frame, num_labels_curr_frame = skimage.measure.label(hulls_curr_frame, return_num = True)
            
            
            for curr_frame_label_num in range(num_labels_curr_frame): # Checking for match b/w prev. frame hulls and curr. frame hulls after zeroing
                
                hull_curr_frame = labels_curr_frame == curr_frame_label_num + 1
            
                overlap = ( ( ( np.sum(hull_prev_frame) - np.sum(hull_curr_frame ^ hull_prev_frame) ) / np.sum(hull_prev_frame) ) ) 
                
                
                if ( overlap > post_zero_hull_overlap_threshold ) or ( frame_num == first_frame ): 
              
                    print( 'Overlap passed threshold: {0}, label: {1}'.format( overlap, curr_frame_label_num ) )               
                    
                    hull_segmentation_curr_frame = hull_curr_frame * thresh_img_curr_frame
                
                    label_vid_array[frame_num] = hull_segmentation_curr_frame
                    
                    hull_prev_frame = hull_curr_frame
                
                    num_labels_prev_frame = num_labels_curr_frame
                            
                    print('New label num frame {}'.format(frame_num))
                    
                    # Re-assigning label_num to speed up label search process for next frame
                    #label_num = curr_frame_label_num
                    
                    break
            
            
# 4.8, 3.5 for 3_31          

c1,t1 = convex_hull_and_thresh_imgs(video_frames[20], 25, 4.9, 3.5)

plt.imshow(t1, cmap = 'gray')

plt.imshow(c1, cmap='gray')

plt.imshow(c1*t1, cmap='gray')

labels1, num1 = skimage.measure.label(c1, return_num = True)

num1

plt.imshow(labels1 == 4, cmap='gray')

plt.imshow((labels1==4) * t1, cmap='gray')

#######################################################################

c2,t2 = convex_hull_and_thresh_imgs(video_frames[21], 25, 4.7, 3.4)

labels2, num_labels2 = skimage.measure.label(c2, return_num = True)

plt.imshow(t2, cmap = 'gray')

plt.imshow(c2, cmap='gray')

plt.imshow(c2*t2, cmap='gray')

plt.imshow((labels2==2) * t2, cmap='gray')



### Solution when can't segment one of a pair of cells ###

th2_test = video_frames[0].copy()

labels_to_zero = list(range(1,num1+1))

current_label = 3

labels_to_zero.remove(current_label)

# test which values in labels1 are in labels_to_zero
zero_labels1_mask = np.in1d(labels1, labels_to_zero).reshape(labels1.shape)

th2_test[zero_labels1_mask] = 0

plt.imshow(th2_test, cmap='gray')

####################################


c2,t2 = convex_hull_and_thresh_imgs(th2_test, 25, 4.2, 3)

labels2, num_labels2 = skimage.measure.label(c2, return_num = True)

plt.imshow(t2, cmap = 'gray')

plt.imshow(c2, cmap='gray')

plt.imshow(c2*t2, cmap='gray')

plt.imshow((labels2==2) * t2, cmap='gray')

            