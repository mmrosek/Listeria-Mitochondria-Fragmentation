#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:23:35 2017

@author: mmrosek
"""
from skimage.filters import threshold_otsu, rank, threshold_local
import imageio
import skimage.filters as filters
import skimage.morphology as morphology
import matplotlib.pyplot as plt
import numpy as np
import cv2
import skimage
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

video = 'DsRed2-HeLa_3_23_LLO_1 (2) (Converted).mov_every_10th_frame.mov'
offset = -0.2

# Read in video file
video_frames = read_in_video("/home/mmrosek/Documents/shannon_vids/{}".format(video), "gray")


####################################################################################################

def hull_and_thresh_hist_eq(image, block_size_img_to_keep, dilation_disk_size, median_disk_size, hist_eq = False, low_res_thresh_for_hull = False, block_size_thresh_hull = 105):
    
    if hist_eq == True:
        
        image = cv2.equalizeHist(image)
    
    if low_res_thresh_for_hull == False:

        adaptive_thresh_hull = threshold_local(image, block_size_thresh_hull, offset = offset) # Higher block size = less granular, cells more easily separated
        thresh_hull = image > adaptive_thresh_hull
        
    else:
        
        # Use when cells are close together --> less granular so helps keep them separated but less detailed
        ret , thresh_hull = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
 
    adaptive_thresh_to_keep = threshold_local(image, block_size_img_to_keep, offset = offset) # Offset = 0 results in large portion of image being white
    thresh_to_keep = image > adaptive_thresh_to_keep
    
    # Dilation
    dilation_disk = morphology.disk(dilation_disk_size)
    new_mask = morphology.binary_dilation(thresh_hull, selem = dilation_disk)
        
    # Median filtering
    median_filtering_disk = morphology.disk(median_disk_size)
    filtered = np.array(filters.median(new_mask, selem = median_filtering_disk), dtype = np.bool)
    
    # Removing small objects
    new_mask = morphology.remove_small_objects(filtered, min_size = 640, connectivity = 8)
    
    # Computing convex hull
    hull = morphology.convex_hull_object(new_mask)

    return(hull,thresh_to_keep)

####################################################################################################

### Frame-to-frame cell segmentation ###


def hull_overlap(hull_prev_frame, hull_curr_frame):
    
    return ( ( ( np.sum(hull_prev_frame) - np.sum(hull_curr_frame ^ hull_prev_frame) ) / np.sum(hull_prev_frame) ) ) 


segmentation_vid_save_path = '/home/mmrosek/Documents/segmented_cell_vid_arrays/' 
dict_save_path = segmentation_vid_save_path
block_size_img_to_keep = 25

dilation_disk_size_list = [6, 5, 4, 2.5, 2.5]

median_disk_size_list = [1, 1.5, 1.5, 1.5, 1.5]

label_of_interest_list = [1, 4, 3, 10, 14] 


#dilation_disk_size_list = [6]
#
#median_disk_size_list = [1]
#
#label_of_interest_list = [1]



low_res_label_list = []


# No significance, just to avoid error on first frame
hull_prev_frame = 1 
labels_prev_frame = 1
   
first_frame = 0

first_hull_overlap_threshold = 0.95

diff_label_hull_overlap_threshold = 0.95

new_disk_size_hull_overlap_threshold = 0.85

label_count = 1

frames_in_low_res_count = 0

start = time.time()

for label_num in label_of_interest_list:
           
    dilation_disk_size = dilation_disk_size_list[label_count-1]
    
    median_disk_size = median_disk_size_list[label_count-1]
    
    broken_frame_dict = {}
    
    print("Label count: {}\n".format(label_count))
    
    print('Label Num, Dil Disk, Med Disk: {}'.format((label_num, dilation_disk_size, median_disk_size)))
    
    # Initializing array for hull_i+1
    label_vid_array = np.zeros(shape = video_frames.shape, dtype = 'uint8')
       
    for frame_num in range(first_frame,video_frames.shape[0]):
        
        image = video_frames[frame_num]
        
        if label_count in low_res_label_list:
            
            frames_in_low_res_count += 1 # Eventually need to try high res thresholding so remove label_count from low_res_list periodically
            
            if frames_in_low_res_count == 20:
                
                low_res_label_list.remove(label_count)
                
                print('\nRemoved label from low res label list\n')
            
            # Calculate convex hull and thresholded image of current frame
            hulls_curr_frame, thresh_img_curr_frame = hull_and_thresh_hist_eq(image, block_size_img_to_keep, dilation_disk_size , median_disk_size, hist_eq = False, low_res_thresh_for_hull = True)
        
        else:
         
            # Calculate convex hull and thresholded image of current frame
            hulls_curr_frame, thresh_img_curr_frame = hull_and_thresh_hist_eq(image, block_size_img_to_keep, dilation_disk_size , median_disk_size, hist_eq = False)
        
        
        # Generating labels for convex hulls of current frame
        labels_curr_frame, num_labels_curr_frame = skimage.measure.label(hulls_curr_frame, return_num = True)
        
        # Isolate a single hull
        hull_curr_frame = labels_curr_frame == label_num # No +1 b/c label_num is not coming from range()
        
        overlap = hull_overlap(hull_prev_frame, hull_curr_frame)  
        
        # Checking to ensure overlap of hull_i+1 is > 80% hull_i+1 from previous frame
        if ( overlap > first_hull_overlap_threshold ) or ( frame_num == first_frame ): 
                 
            if frame_num % 100 == 0:
        
                print("Frame: {0}, overlap: {1}, disk: {2}, label: {3}\n".format(frame_num, overlap, dilation_disk_size, label_num ))
        
            # Isolate part of thresholded image overlapped by hull_i+1
            hull_segmentation_curr_frame = hull_curr_frame * thresh_img_curr_frame
        
            # Add isolated part of thresholded image to new array
            label_vid_array[frame_num] = hull_segmentation_curr_frame
            
            # Converting number of labels from current frame to number of labels from the now previous frame
            num_labels_prev_frame = num_labels_curr_frame

            # Converting hulls from current frame to hulls from the now previous frame
            hull_prev_frame = hull_curr_frame
        
        
        else: # If hull_i+1 doesn't overlap enough with hull_i+1 from previous frame, see if it matches with another label
            
            print( 'Overlap didnt pass threshold: {}'.format( overlap ) )
            
       
            for curr_frame_label_num in range(1, num_labels_curr_frame + 1): # Check if hull of interest matches hull with diff label_num
                
                hull_curr_frame = labels_curr_frame == curr_frame_label_num 
                
                overlap = hull_overlap(hull_prev_frame, hull_curr_frame) 
                
                
                if ( overlap > diff_label_hull_overlap_threshold ) or ( frame_num == first_frame ): # Need to make 0.95 an argument to function
              
                    print( 'Overlap passed threshold for label: {0}'.format(curr_frame_label_num ) )               
                    
                    hull_segmentation_curr_frame = hull_curr_frame * thresh_img_curr_frame
                
                    label_vid_array[frame_num] = hull_segmentation_curr_frame
                    
                    hull_prev_frame = hull_curr_frame
                
                    num_labels_prev_frame = num_labels_curr_frame
                            
                    print('New label num frame {}\n'.format(frame_num))
                    
                    # Re-assigning label_num to speed up label search process for next frame
                    label_num = curr_frame_label_num
                    
                    break
                
            
            if curr_frame_label_num == num_labels_curr_frame: # If the above loop didn't work, try different dilation disk sizes with each label
       
                # None of the existing hulls are the right shape, trying different dilation disk sizes
                x = dilation_disk_size
                
                if label_count in low_res_label_list:
                    
                    low_res_dil_disk_sizes = [ max(1.05, x ** ( 1 - ( i/25 )**3 ) ) if i % 2 == 0 else min(40, ( x ** ( 1 + ( i/25 )**3 ) )) for i in range(20)]                    
                
                elif x >= 3:
                    
                    new_dil_disk_sizes = [ max(1.05, x ** ( 1 - ( i/14 )**3 ) ) if i % 2 == 0 else min(40, ( x ** ( 1 + ( i/14 )**3 ) )) for i in range(16)]
                    
                else:
            
                    new_dil_disk_sizes = [ max(1.05, x ** ( 1 - ( i/10 )**3 ) ) if i % 2 == 0 else min(40, ( x ** ( 1 + ( i/10 )**3 ) )) for i in range(18)]
                    
                max_overlap = 0
                
                opt_dil_disk_size = .1
                
                opt_label_num = .1
                             
                print('Trying new disk sizes frame {}'.format(frame_num))
                
                dil_disk_count = 0
                
                for dil_disk_size in new_dil_disk_sizes:
                    
                    dil_disk_count += 1
                                   
                    if label_count in low_res_label_list:
                        
                        hulls_curr_frame, thresh_img_curr_frame = hull_and_thresh_hist_eq(image, block_size_img_to_keep, dil_disk_size , median_disk_size, hist_eq = False, low_res_thresh_for_hull = True)
                    
                    else:
                     
                        hulls_curr_frame, thresh_img_curr_frame = hull_and_thresh_hist_eq(image, block_size_img_to_keep, dil_disk_size , median_disk_size, hist_eq = False)
                                
                   
                    labels_curr_frame, num_labels_curr_frame = skimage.measure.label(hulls_curr_frame, return_num = True)
                    
                    for curr_frame_label_num in range(1, num_labels_curr_frame + 1): # Check if hull_i+1 matches hull with diff label_num
                                     
                        hull_curr_frame = labels_curr_frame == curr_frame_label_num 
                        
                        test_overlap = hull_overlap(hull_prev_frame, hull_curr_frame) 
                    

                        if (test_overlap - 0.001) > max_overlap: # Saving optimal disk size based on max overlap
                            
                            max_overlap = test_overlap
                            
                            opt_dil_disk_size = dil_disk_size
                            
                            opt_label_num = curr_frame_label_num
                            
                    
                    if ( dil_disk_count == len(new_dil_disk_sizes) ): # Tested each disk size...
                    
                        # If overlap passes new_disk_size_hull_overlap_threshold or we are already using low res threshold...
                        if ( max_overlap > new_disk_size_hull_overlap_threshold ) or ( label_count in low_res_label_list ):
                        
                            print('Max overlap: {0} for new disk size: {1}'.format(max_overlap, opt_dil_disk_size))
                            
                            if label_count in low_res_label_list:
        
                                print('Low res label: {}'.format(label_count))
                                
                                hulls_curr_frame, thresh_img_curr_frame = hull_and_thresh_hist_eq(image, block_size_img_to_keep, opt_dil_disk_size , median_disk_size, hist_eq = False, low_res_thresh_for_hull = True)
                            
                            else:
                             
                                hulls_curr_frame, thresh_img_curr_frame = hull_and_thresh_hist_eq(image, block_size_img_to_keep, opt_dil_disk_size , median_disk_size, hist_eq = False)                      
                                       
                                    
                            labels_curr_frame, num_labels_curr_frame = skimage.measure.label(hulls_curr_frame, return_num = True)
                            
                            hull_curr_frame = labels_curr_frame == opt_label_num
                            
                            hull_segmentation_curr_frame = hull_curr_frame * thresh_img_curr_frame
                
                            label_vid_array[frame_num] = hull_segmentation_curr_frame
                            
                            hull_prev_frame = hull_curr_frame 
                            
                            broken_frame_dict[frame_num] = (max_overlap, opt_dil_disk_size)
                            
                            dilation_disk_size = opt_dil_disk_size
                            
                            # Re-assigning label_num to speed up label search process for next frame
                            label_num = opt_label_num
                            
                            print("Didn't try low res or already low res\n")
                            
                            
                        else: # Trying low res threshold to try to improve upon max_threshold
                        
                            x = dilation_disk_size
                            
                            print('Low_res_dil_disk_size being used to initiate low_res_dil_disk_sizes: {}'.format(x))
                        
                            low_res_dil_disk_sizes = [ max(1.05, x ** ( 1 - ( i/25 )**3 ) ) if i % 2 == 0 else min(30, ( x ** ( 1 + ( i/25 )**3 ) )) for i in range(20)]
                                
                            max_low_res_overlap = 0
                
                            opt_low_res_dil_disk_size = .1
                            
                            opt_low_res_label_num = .1
                                         
                            print('Trying new low_res disk sizes frame {}'.format(frame_num))
                            
                            low_res_dil_disk_count = 0
                            
                            for dil_disk_size in low_res_dil_disk_sizes:
                                
                                low_res_dil_disk_count += 1
                                    
                                hulls_curr_frame, thresh_img_curr_frame = hull_and_thresh_hist_eq(image, block_size_img_to_keep, dil_disk_size , median_disk_size, hist_eq = False, low_res_thresh_for_hull = True)
                                    
                                labels_curr_frame, num_labels_curr_frame = skimage.measure.label(hulls_curr_frame, return_num = True)
                                
                                
                                for curr_frame_label_num in range(1, num_labels_curr_frame + 1): # Check if hull_i+1 matches hull with diff label_num
                                                 
                                    hull_curr_frame = labels_curr_frame == curr_frame_label_num 
                                    
                                    test_overlap = hull_overlap(hull_prev_frame, hull_curr_frame) 
                                
            
                                    if (test_overlap - 0.001) > max_low_res_overlap: # Saving optimal disk size based on max overlap
                                        
                                        max_low_res_overlap = test_overlap
                                        
                                        opt_low_res_dil_disk_size = dil_disk_size
                                        
                                        opt_low_res_label_num = curr_frame_label_num
                                        
                                
                                if ( low_res_dil_disk_count == len(low_res_dil_disk_sizes) ): # Tested each low_res disk size...
                                
                                    print('Max normal overlap: {}'.format(max_overlap))
                                    
                                    print('Max low res overlap: {}'.format(max_low_res_overlap))
                                
                                    if max_overlap > max_low_res_overlap:
                                                                   
                                        print('Max overlap: {0} for new disk size: {1}\n'.format(max_overlap, opt_dil_disk_size))
                                                                                 
                                        hulls_curr_frame, thresh_img_curr_frame = hull_and_thresh_hist_eq(image, block_size_img_to_keep, opt_dil_disk_size , median_disk_size, hist_eq = False)                      
                                                   
                                        labels_curr_frame, num_labels_curr_frame = skimage.measure.label(hulls_curr_frame, return_num = True)
                                        
                                        hull_curr_frame = labels_curr_frame == opt_label_num
                                        
                                        hull_segmentation_curr_frame = hull_curr_frame * thresh_img_curr_frame
                            
                                        label_vid_array[frame_num] = hull_segmentation_curr_frame
                                        
                                        hull_prev_frame = hull_curr_frame 
                                        
                                        broken_frame_dict[frame_num] = (max_overlap, opt_dil_disk_size)
                                        
                                        dilation_disk_size = opt_dil_disk_size
                                        
                                        # Re-assigning label_num to speed up label search process for next frame
                                        label_num = opt_label_num
                                        
                                    else:
                                        
                                        print('Max low res overlap: {0} for disk size: {1}'.format(max_low_res_overlap, opt_low_res_dil_disk_size))
                                                                                 
                                        hulls_curr_frame, thresh_img_curr_frame = hull_and_thresh_hist_eq(image, block_size_img_to_keep, opt_low_res_dil_disk_size , median_disk_size, hist_eq = False, low_res_thresh_for_hull = True)                      
                                                   
                                        labels_curr_frame, num_labels_curr_frame = skimage.measure.label(hulls_curr_frame, return_num = True)
                                        
                                        hull_curr_frame = labels_curr_frame == opt_low_res_label_num
                                        
                                        hull_segmentation_curr_frame = hull_curr_frame * thresh_img_curr_frame
                            
                                        label_vid_array[frame_num] = hull_segmentation_curr_frame
                                        
                                        hull_prev_frame = hull_curr_frame 
                                        
                                        broken_frame_dict[frame_num] = (max_low_res_overlap, opt_low_res_dil_disk_size)
                                        
                                        dilation_disk_size = opt_low_res_dil_disk_size
                                        
                                        # Re-assigning label_num to speed up label search process for next frame
                                        label_num = opt_low_res_label_num
                                        
                                        low_res_label_list.append(label_count) # This label is now using low res thresholding
                                        
                                        print('\nLabel {} now in low_res_label_list\n'.format(label_count))
                                                                              
                    

    np.save(segmentation_vid_save_path + video + '/hull_{0}_segmentation_7_26_{1}.npy'.format(label_count, block_size_img_to_keep), label_vid_array)
    
    del label_vid_array
    
    np.save(dict_save_path + video + '/broken_frames_dict_label_{0}_7_26_{1}.npy'.format(label_count, block_size_img_to_keep), broken_frame_dict)
    
    label_count+=1


end = time.time()
print(end-start)

#####################################################################################################################################################################
#####################################################################################################################################################################

# Debugging

array_name = "/hull_1_segmentation_7_25_25.npy"


hull_1_array = np.load(segmentation_vid_save_path + video + array_name)


plt.imshow(hull_1_array[200], cmap='gray')

plt.imshow(label_vid_array[201], cmap='gray')

##########################################################


dilation_disk_size_list = [5, 5, 4, 2.5, 2.5]

median_disk_size_list = [1.25, 1.5, 1.5, 1.5, 1.5]

label_of_interest_list = [1, 4, 3, 10, 14] 

image = video_frames[200]

c1,t1 = hull_and_thresh_hist_eq(image,25, 1.9, 1, False, False)

plt.imshow(t1, cmap = 'gray')

plt.imshow(c1, cmap='gray')

plt.imshow(t1*c1, cmap='gray')

labels1, num1 = skimage.measure.label(c1, return_num = True)

num1

hull_prev_frame = labels1 == 1

hull_curr_frame = labels1 == 1

plt.imshow(hull_prev_frame, cmap='gray')

plt.imshow(hull_curr_frame, cmap='gray')


image = video_frames[201]

c1,t1 = hull_and_thresh_hist_eq(image,25, 1.9, 1, False, False)

plt.imshow(t1, cmap = 'gray')

plt.imshow(c1, cmap='gray')

plt.imshow(t1*c1, cmap='gray')

labels2, num1 = skimage.measure.label(c1, return_num = True)

hull_curr_frame = labels2 == 1


hull_overlap(hull_prev_frame, hull_curr_frame)
