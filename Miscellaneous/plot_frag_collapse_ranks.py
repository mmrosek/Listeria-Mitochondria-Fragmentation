#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 13:16:49 2017

@author: mmrosek
"""
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, scale
from matplotlib.backends.backend_pdf import PdfPages
import operator
from scipy.stats import spearmanr, pearsonr


llo_dict_read_path = '/home/mmrosek/Documents/llo_region_dicts/' 

plot_write_path = '/home/mmrosek/Documents/cca_images/'

#dict_list = ['large_regions', 'medium_regions', 
#             'all_regions','total_size']

threshold_type = 'adpative_15'
      
dict_type = 'large_regions'

llo_region_dict = np.load(llo_dict_read_path + 'llo_' + dict_type + '_dict_15.npy').item()


####################################################################################################################

def calc_window_max(list_name, window_size = 100, start_frame = 2500, median = True):
    
    max_window_value = 0
    
    if median == False:
        optima_function = np.mean
    else:
        optima_function = np.median
    
    for i in range(start_frame ,len(list_name)-window_size+1):
        
        if optima_function(list_name[i : i + window_size]) > max_window_value:
            
            max_window_value = optima_function(list_name[i : i + window_size])
            
            max_window_indices = (i, window_size + i)
            
    return(max_window_value,max_window_indices)
            
####################################################################################################################

def calc_window_min(list_name, window_size = 100, start_frame = 0, median = True):
    
    min_window_value = max(list_name)
    
    if median == False:
        optima_function = np.mean
    else:
        optima_function = np.median
    
    for i in range(start_frame,len(list_name)-window_size+1):
        
        if optima_function(list_name[i : i + window_size]) < min_window_value:
            
            min_window_value = optima_function(list_name[i : i + window_size])
            
            min_window_indices = (i, window_size + i)
            
    return(min_window_value,min_window_indices)

####################################################################################################################

#def create_range_dict_collapse(video_dictionary, window_size, start_frame = 2500):
#    
#    range_dict = {}
#    
#    for key, value in video_dictionary.items():
#        
#        print(key)
#        
#        value = value[start_frame:]
#        
#        max_median_window_value, max_median_window_index = calc_window_max(value, window_size) # max_median_window_index is on right side of window
#
#        frames_after_peak = np.array(value)[ max_median_window_index - window_size : ]
#        
#        print(frames_after_peak.shape)
#        
#        normalized_frames_after_peak = normalize(frames_after_peak.reshape(1,-1).astype(float))[0]
#    
#        max_norm_median_window_value,_ = calc_window_max(normalized_frames_after_peak, window_size)
#        
#        min_norm_median_window_value_after_peak , _ = calc_window_min(normalized_frames_after_peak, window_size)
#        
#        range_ = max_norm_median_window_value - min_norm_median_window_value_after_peak
#        
#        range_dict[key] = range_
#         
#    sorted_range_dict = sorted(range_dict.items(), key=operator.itemgetter(1))
#            
#    return(sorted_range_dict)

####################################################################################################################

def create_range_dict_fragmentation(video_dictionary, window_size, start_frame):
     
    range_dict = {}
    
    for key, value in video_dictionary.items():
        
        print(key)
        
        max_median_window_value, max_median_window_indices = calc_window_max(value, window_size) # max_median_window_index is on right side of window
    
        frames_before_peak_after_llo = np.array(value)[ start_frame : max_median_window_indices[1] + 1]
        
        normalized_frames_before_peak_after_llo = normalize(frames_before_peak_after_llo.reshape(1,-1).astype(float))[0]
    
        max_norm_median_window_value = np.median(normalized_frames_before_peak_after_llo[ normalized_frames_before_peak_after_llo.shape[0] - window_size : normalized_frames_before_peak_after_llo.shape[0]])
        
        min_norm_median_window_value_before_peak , min_indices = calc_window_min(normalized_frames_before_peak_after_llo, window_size)
        
        print(min_indices)
        
        range_ = max_norm_median_window_value / min_norm_median_window_value_before_peak
        
        range_dict[key] = range_
        
    sorted_range_dict = sorted(range_dict.items(), key=operator.itemgetter(1))
            
    return(sorted_range_dict)


####################################################################################################################

# Generate lists containing collapse/frag values

collapse_list = create_range_dict_collapse(llo_region_dict, 100, 2800)

fragmentation_list = create_range_dict_fragmentation(llo_region_dict, 100, 2800)


# Create dictionary containing values for each video as tuple
coordinate_dict = {}

for video in llo_region_dict.keys():
    
    collapse_index = [idx for idx in range(len(collapse_list)) if video in collapse_list[idx]]
    
    frag_index = [idx for idx in range(len(fragmentation_list)) if video in fragmentation_list[idx]]
    
    coordinate_dict[video] = (collapse_list[collapse_index[0]][1] , fragmentation_list[frag_index[0]][1])


# Create list of video names
video_names = [video[12:len(str(video))-16] for video in coordinate_dict.keys()]

# Create lists containing coordinates for scaling
collapse_coordinates = []
fragmentation_coordinates = []

for tup in coordinate_dict.values():
    collapse_coordinates.append(tup[0])
    fragmentation_coordinates.append(tup[1])
    
scaled_collapse_coordinates = scale(collapse_coordinates)
scaled_fragmentation_coordinates = scale(fragmentation_coordinates)
    

# Initialize list of colors 
jet = plt.cm.jet
colors = jet(np.linspace(0, 1, len(coordinate_dict)))

for collapse_coord, frag_coord, color in zip(scaled_collapse_coordinates, scaled_fragmentation_coordinates, colors):
   x = collapse_coord
   y = frag_coord
   plt.scatter(x,y,color=color)
plt.title('Fragmentation vs. Cell Collapse')
plt.xlabel('Cell Collapse')
plt.ylabel('Fragmentation')
plt.legend(video_names, loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(plot_write_path+'frag_vs_collapse.png',bbox_inches='tight')
plt.show()


# Unscaled plot

#jet = plt.cm.jet
#colors = jet(np.linspace(0, 1, len(coordinate_dict)))
#
#for tup, color in zip(coordinate_dict.values(), colors):
#   x = tup[0]
#   y = tup[1]
#   plt.scatter(x,y,color=color)
#
#plt.title('Fragmentation vs. Cell Collapse')
#plt.ylabel('Fragmentation')
#plt.xlabel('Cell Collapse')
#plt.legend(video_names, loc='center left', bbox_to_anchor=(1, 0.5))
#plt.show()