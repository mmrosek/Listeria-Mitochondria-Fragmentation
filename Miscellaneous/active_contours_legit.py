#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:30:28 2017

@author: mmrosek
"""
from skimage.filters import threshold_otsu, rank, threshold_local
import imageio
import skimage.filters as filters
import skimage.morphology as morphology
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import scipy
import cv2
import skimage
from skimage import segmentation
from skimage.filters import gaussian


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

file_list = ['DsRed2-HeLa_2_21_LLO (Converted).mov','DsRed2-HeLa_3_15_LLO1part1 (Converted).mov','DsRed2-HeLa_3_15_LLO2part1 (Converted).mov','DsRed2-HeLa_3_1_LLO (Converted).mov',
'DsRed2-HeLa_3_23_LLO_1 (2) (Converted).mov','DsRed2-HeLa_3_31_LLO_1 (2) (Converted).mov',
'DsRed2-HeLa_3_31_LLO_2 (2) (Converted).mov','DsRed2-HeLa_3_31_LLO_3 (2) (Converted).mov',
'DsRed2-HeLa_3_8_LLO002 (Converted).mov',  'DsRed2-HeLa_3_9_LLO (Converted).mov',
'DsRed2-HeLa_4_5_LLO1 (Converted).mov','DsRed2-HeLa_4_5_LLO2 (Converted).mov']

plot_write_path = '/home/mmrosek/Documents/cca_images/llo/individual_video_plots/'  # Specifies path for plots to be written to

diagonal_connection = [[1,1,1],[1,1,1],[1,1,1]] # Structure that indicates pixels connected diagonally count as connected (used in loop below)

block_size = 35 # Higher number = larger window size/neighbordhood

offset = -0.2

video = file_list[0]

# Read in video file
video_frames = read_in_video("/home/mmrosek/Documents/shannon_vids/{}".format(video), "gray")

############################

# Active Contour example

image = video_frames[0]

adaptive_thresh = threshold_local(image, block_size, offset = offset) # Offset = 0 results in large portion of image being white
img = image > adaptive_thresh

plt.imshow(img, cmap = 'gray')

img = x

s = np.linspace(0, 2*np.pi, 400)
x = 450 + 100*np.cos(s)
y = 200 + 140*np.sin(s)
init = np.array([x, y]).T


snake = segmentation.active_contour(gaussian(img, 3),
                           init, alpha=0.015, beta=10, gamma=0.001)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)
plt.gray()
ax.imshow(img)
ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])


# Plot only snaked region
max_y = np.round(np.max(snake[:, 1]))
min_y = np.round(np.min(snake[:, 1]))
max_x = np.round(np.max(snake[:, 0]))
min_x = np.round(np.min(snake[:, 0]))

plt.imshow(img[ int(min_y)-1 : int(max_y)+1, int(min_x)-1 : int(max_x)+1])

###########################

## Shannon QBI test
#
## Dilation
#iterations = 1
#new_mask = img
#disk = morphology.disk(2)
#for i in range(iterations):
#    new_mask = morphology.binary_dilation(new_mask, selem = disk)
#    
## Median filtering
#disk = morphology.disk(3)
#filtered = np.array(filters.median(new_mask, selem = disk), dtype = np.bool)
#
#plt.imshow(filtered, cmap = "gray")
#
## Removing small objects
#new_mask = morphology.remove_small_objects(filtered, min_size = 512, connectivity = 8)
#plt.imshow(new_mask, cmap = "gray")
#
## Converts to float data type
##new_mask = new_mask.astype(float)
#
## Computing convex hull
#hull = morphology.convex_hull_object(new_mask)
#plt.imshow(hull, cmap = "gray")
#
## Generate labels for each convex hull
#labels, num_labels = skimage.measure.label(hull, return_num = True)
#
## Isolate a single hull
#hull_1 = labels == 1
#
#plt.imshow(hull_1, cmap = 'gray')
#
## Isolate part of original image corresponding to hull_1
#img_hull_1 = img * hull_1
#
#plt.imshow(img_hull_1, cmap = 'gray')
#
#
