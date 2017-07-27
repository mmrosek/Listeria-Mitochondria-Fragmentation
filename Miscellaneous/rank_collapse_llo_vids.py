#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:12:43 2017

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

dict_list = ['large_regions', 'medium_regions', 
             'all_regions','total_size']

threshold_type = 'adpative_45'
      
dict_type = 'large_regions'

llo_region_dict = np.load(llo_dict_read_path + 'llo_' + dict_type + '_dict_15.npy').item()


optimal_rank = [('DsRed2-HeLa_3_31_LLO_1 (2) (Converted).mov', 4),
 ('DsRed2-HeLa_3_31_LLO_2 (2) (Converted).mov', 4.5),
 ('DsRed2-HeLa_3_15_LLO1part1 (Converted).mov', 5),
 ('DsRed2-HeLa_3_1_LLO (Converted).mov', 8.6),
 ('DsRed2-HeLa_4_5_LLO1 (Converted).mov', 7.9),
 ('DsRed2-HeLa_3_31_LLO_3 (2) (Converted).mov', 5),
 ('DsRed2-HeLa_3_23_LLO_1 (2) (Converted).mov', 8.9),
 ('DsRed2-HeLa_3_9_LLO (Converted).mov', 4),
 ('DsRed2-HeLa_2_21_LLO (Converted).mov', 9.2),
 ('DsRed2-HeLa_4_5_LLO2 (Converted).mov', 8.4),
 ('DsRed2-HeLa_3_15_LLO2part1 (Converted).mov', 8.9),
 ('DsRed2-HeLa_3_8_LLO002 (Converted).mov', 7.9)]


def eval_ranking(rankings):
    
    optimal_list = [rating[1] for rating in optimal_rank]

    ranking_list = []

    for opt_idx in range(len(optimal_rank)):
        for rank_idx in range(len(rankings)):
            if rankings[rank_idx][0] == optimal_rank[opt_idx][0]:
                ranking_list.append(rankings[rank_idx][1])
                continue
            
    return([spearmanr(optimal_list,ranking_list), pearsonr(optimal_list,ranking_list)])

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
####################################################################################################################

##### Different functions to generate ranking dictionaries #####

def create_range_dict_norm(video_dictionary):
    
    range_dict = {}
    for key, value in video_dictionary.items():
        video = key
        scaled_array = normalize(np.array(value).reshape(1,-1).astype(float))[0]
        peak_value = np.max(scaled_array)
        frames_after_peak = scaled_array[np.argmax(scaled_array) : ]
        min_value_after_peak = np.min(frames_after_peak)
        range_ = peak_value / min_value_after_peak
        range_dict[video] = range_
        sorted_range_dict = sorted(range_dict.items(), key=operator.itemgetter(1))
    
    return(sorted_range_dict)

####################################################################################################################
####################################################################################################################

### Same as create_range_dict_norm

#def create_range_dict_norm_after_peak(video_dictionary):
#    range_dict = {}
#    for key, value in video_dictionary.items():
#        video = key
#        array = np.array(value)
#        frames_after_peak = array[np.argmax(array) : ]
#        norm_frames_after_peak = normalize(frames_after_peak.reshape(1,-1).astype(float))[0]
#        peak_value = np.max(norm_frames_after_peak)
#        min_value_after_peak = np.min(norm_frames_after_peak)
#        range_ = peak_value / min_value_after_peak
#        range_dict[video] = range_
#    sorted_range_dict = sorted(range_dict.items(), key=operator.itemgetter(1))
#    return(sorted_range_dict)

####################################################################################################################
####################################################################################################################

### Same as create_range_dict_norm I believe

def create_range_dict_norm_after_peak_after_llo_(video_dictionary, start_frame = 2500):
    
    range_dict = {}
    
    for key, value in video_dictionary.items():
    
        video = key
    
        array = np.array(value[ start_frame : ])
        
        frames_after_peak_after_llo = array[ np.argmax(array) : ]
        
        norm_frames_after_peak_after_llo = normalize(frames_after_peak_after_llo.reshape(1,-1).astype(float))[0]
        
        peak_value = np.max(norm_frames_after_peak_after_llo)
        
        min_value_after_peak = np.min(norm_frames_after_peak_after_llo)
        
        range_ = peak_value / min_value_after_peak
        
        range_dict[video] = range_
        
    sorted_range_dict = sorted(range_dict.items(), key=operator.itemgetter(1))
    return(sorted_range_dict)

####################################################################################################################
####################################################################################################################

def create_range_dict_norm_after_peak_smoothed_optima(video_dictionary, window_size,  start_frame = 2500):
    
    range_dict = {}
    
    for key, value in video_dictionary.items():
        
        print(key)
        
        max_median_window_value, max_median_window_indices = calc_window_max(value, window_size) # max_median_window_index is on right side of window
        
        frames_after_peak = np.array(value)[ max_median_window_indices[0] : ]
        
        print(frames_after_peak.shape)
        
        normalized_frames_after_peak = normalize(frames_after_peak.reshape(1,-1).astype(float))[0]
        
        max_norm_median_window_value = np.median(normalized_frames_after_peak[ : window_size])
        
        min_norm_median_window_value_after_peak , _ = calc_window_min(normalized_frames_after_peak, window_size)
        
        range_ = max_norm_median_window_value / min_norm_median_window_value_after_peak
        
        range_dict[key] = range_
    
    sorted_range_dict = sorted(range_dict.items(), key=operator.itemgetter(1))
        
    return(sorted_range_dict)


####################################################################################################################
####################################################################################################################

def create_range_dict_norm_after_peak_after_llo_smoothed_optima(video_dictionary, window_size, start_frame = 2500):
    
    range_dict = {}
    
    for key, value in video_dictionary.items():
        
        print(key)
        
        value = value[start_frame:]
        
        max_median_window_value, max_median_window_indices = calc_window_max(value, window_size) # max_median_window_index is on right side of window

        frames_after_peak = np.array(value)[ max_median_window_indices[0] : ]
        
        print(frames_after_peak.shape)
        
        normalized_frames_after_peak = normalize(frames_after_peak.reshape(1,-1).astype(float))[0]
    
        max_norm_median_window_value = np.median(normalized_frames_after_peak[ : window_size])
        
        min_norm_median_window_value_after_peak , _ = calc_window_min(normalized_frames_after_peak, window_size)
        
        range_ = max_norm_median_window_value / min_norm_median_window_value_after_peak
        
        range_dict[key] = range_
         
    sorted_range_dict = sorted(range_dict.items(), key=operator.itemgetter(1))
            
    return(sorted_range_dict)


####################################################################################################################

z = create_range_dict_norm(llo_region_dict)

#x = create_range_dict_norm_after_peak(llo_region_dict)

p = create_range_dict_norm_after_peak_after_llo_(llo_region_dict)

s = create_range_dict_norm_after_peak_smoothed_optima(llo_region_dict, 150)

pl = create_range_dict_norm_after_peak_after_llo_smoothed_optima(llo_region_dict, 100, 2800)



eval_ranking(z)

eval_ranking(x)

eval_ranking(p)

eval_ranking(s)

eval_ranking(pl)








##### Other ideas #####

####################################################################################################################

#def max_over_min_shortly_after_peak_after_llo_smoothed_optima(video_dictionary, window_size, start_frame = 2500, num_frames_after_peak = 3000):
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
#        frames_after_peak = np.array(value)[ max_median_window_index - window_size : max_median_window_index - window_size + num_frames_after_peak]
#        
#        print(frames_after_peak.shape)
#        
#        normalized_frames_after_peak = normalize(frames_after_peak.reshape(1,-1).astype(float))[0]
#    
#        max_norm_median_window_value,_ = calc_window_max(normalized_frames_after_peak, window_size)
#        
#        min_norm_median_window_value_after_peak , _ = calc_window_min(normalized_frames_after_peak, window_size)
#        
#        range_ = max_norm_median_window_value / min_norm_median_window_value_after_peak
#        
#        range_dict[key] = range_
#         
#    sorted_range_dict = sorted(range_dict.items(), key=operator.itemgetter(1))
#            
#    return(sorted_range_dict)


####################################################################################################################

#def max_over_min_after_peak_after_llo_smoothed_optima(video_dictionary, window_size, start_frame = 2500):
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
#        range_ = max_norm_median_window_value / min_norm_median_window_value_after_peak
#        
#        range_dict[key] = range_
#         
#    sorted_range_dict = sorted(range_dict.items(), key=operator.itemgetter(1))
#            
#    return(sorted_range_dict)
#
#####################################################################################################################
#
#def median_value_after_smooth_peak_after_llo(video_dictionary, window_size, start_frame = 2500):
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
#        min_norm_median_value_after_peak = np.median(normalized_frames_after_peak[window_size : ])
#        
#        range_ = max_norm_median_window_value / min_norm_median_value_after_peak
#        
#        range_dict[key] = range_
#         
#    sorted_range_dict = sorted(range_dict.items(), key=operator.itemgetter(1))
#            
#    return(sorted_range_dict)
