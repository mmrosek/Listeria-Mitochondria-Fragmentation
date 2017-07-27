#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:10:58 2017

@author: mmrosek
"""
import time
import cv2
import pprint
import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank, threshold_local
from skimage.util import img_as_ubyte

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

# Read in video file
start = time.time()
video_frames = read_in_video("/home/mmrosek/Documents/shannon_vids/DsRed2-HeLa_4_5_LLO2 (Converted).mov", "gray")
end=time.time()
print(end-start)


############### Thresholding ####################

image = video_frames[0,:,:]

# Plots first frame of video #
plt.imshow(image, cmap='gray')


### Global Otsu ###

global_thresh = threshold_otsu(image)
global_otsu = image > global_thresh
plt.imshow(global_otsu,cmap='gray')


### Local Thresholding ###

block_size = 65 # Higher number = less granular
adaptive_thresh = threshold_local(image, block_size, offset=-0.000001) # Offset = 0 results in large portion of image being white
binary_adaptive = image > adaptive_thresh
plt.imshow(binary_adaptive,cmap='gray')


### Local Otsu ###

# Higher number = more of image being black and slower 
radius = 12
selem = disk(radius)
local_otsu = img > rank.otsu(img, selem)
plt.imshow(local_otsu,cmap='gray')




######### Alternatives ##############

th2 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
ret2,th2 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)





