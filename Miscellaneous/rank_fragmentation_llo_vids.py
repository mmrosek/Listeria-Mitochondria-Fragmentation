#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:00:19 2017

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
    
data_type = dict_type.split('_')[0].capitalize() + " " + dict_type.split('_')[1].capitalize()

llo_read_dict = np.load(llo_dict_read_path + 'llo_' + dict_type + '_dict_15.npy').item()


#optimal_rank = [('DsRed2-HeLa_3_31_LLO_2 (2) (Converted).mov', 5),
# ('DsRed2-HeLa_3_31_LLO_1 (2) (Converted).mov', 5),
# ('DsRed2-HeLa_3_31_LLO_3 (2) (Converted).mov', 5.5),
# ('DsRed2-HeLa_3_15_LLO1part1 (Converted).mov', 6),
# ('DsRed2-HeLa_3_1_LLO (Converted).mov', 7),
# ('DsRed2-HeLa_2_21_LLO (Converted).mov', 7),
# ('DsRed2-HeLa_4_5_LLO2 (Converted).mov', 7.5),
# ('DsRed2-HeLa_3_15_LLO2part1 (Converted).mov', 7.5),
# ('DsRed2-HeLa_4_5_LLO1 (Converted).mov', 7.5),
# ('DsRed2-HeLa_3_23_LLO_1 (2) (Converted).mov', 9),
# ('DsRed2-HeLa_3_8_LLO002 (Converted).mov', 9),
# ('DsRed2-HeLa_3_9_LLO (Converted).mov', 9.1)]


optimal_rank = [('DsRed2-HeLa_3_31_LLO_1 (2) (Converted).mov', 4),
 ('DsRed2-HeLa_3_31_LLO_2 (2) (Converted).mov', 5),
 ('DsRed2-HeLa_3_15_LLO1part1 (Converted).mov', 5),
 ('DsRed2-HeLa_3_1_LLO (Converted).mov', 6.6),
 ('DsRed2-HeLa_4_5_LLO1 (Converted).mov', 6.7),
 ('DsRed2-HeLa_3_31_LLO_3 (2) (Converted).mov', 6.8),
 ('DsRed2-HeLa_3_23_LLO_1 (2) (Converted).mov', 8.6),
 ('DsRed2-HeLa_3_9_LLO (Converted).mov', 8.6),
 ('DsRed2-HeLa_2_21_LLO (Converted).mov', 8.6),
 ('DsRed2-HeLa_4_5_LLO2 (Converted).mov', 9),
 ('DsRed2-HeLa_3_15_LLO2part1 (Converted).mov', 8.9),
 ('DsRed2-HeLa_3_8_LLO002 (Converted).mov', 9.2)]


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


def create_range_dict(video_dictionary):
    
#    videos = list(video_dictionary.keys())
#    arrays = list(video_dictionary.values())
    range_dict = {}
    for key, value in video_dictionary.items():
        video = key
        array = np.array(value)
        peak_value = np.max(array)
        frames_before_peak = array[:np.argmax(array)]
        min_value_before_peak = np.min(frames_before_peak)
        range_ = peak_value / min_value_before_peak
        range_dict[video] = range_
        sorted_range_dict = sorted(range_dict.items(), key=operator.itemgetter(1))
        
    return(sorted_range_dict)

####################################################################################################################

def create_norm_range_dict(video_dictionary):
    
    range_dict = {}
    for key, value in video_dictionary.items():
        scaled_array = normalize(np.array(value).reshape(1,-1).astype(float))[0]
        peak_value = np.max(scaled_array)
        frames_before_peak = scaled_array[:np.argmax(scaled_array)]
        min_value_before_peak = np.min(frames_before_peak)
        range_ = peak_value / min_value_before_peak
        range_dict[key] = range_
        sorted_range_dict = sorted(range_dict.items(), key=operator.itemgetter(1))
    
    return(sorted_range_dict)

####################################################################################################################

def create_norm_range_dict_pre_peak_post_llo(video_dictionary, start_frame):
    
    range_dict = {}
    
    for key, value in video_dictionary.items():
    
        video = key
    
        array = np.array(value)
        
        frames_before_peak_post_llo = array[ start_frame : np.argmax(array) + 1 ]
        
        scaled_frames_before_peak_post_llo = normalize(frames_before_peak_post_llo.reshape(1,-1).astype(float))[0]
        
        peak_value = np.max(scaled_frames_before_peak_post_llo)
        
        min_value_before_peak = np.min(scaled_frames_before_peak_post_llo)
        
        range_ = peak_value / min_value_before_peak
        
        range_dict[video] = range_
        
    sorted_range_dict = sorted(range_dict.items(), key=operator.itemgetter(1))
    
    return(sorted_range_dict)

####################################################################################################################

def create_norm_range_dict_smoothed_optima(video_dictionary, window_size):
    
    range_dict = {}
    
    for key, value in video_dictionary.items():
        
        print(key)
        
        norm_array = normalize(np.array(value).reshape(1,-1).astype(float))[0]
        
        max_norm_median_window_value, max_median_window_indices = calc_window_max(norm_array, window_size) # max_median_window_index is on right side of window
        
        frames_before_peak = norm_array[ : max_median_window_indices[0] ]
        
        min_norm_median_window_value_before_peak , _ = calc_window_min(frames_before_peak, window_size)
        
        range_ = max_norm_median_window_value / min_norm_median_window_value_before_peak
        
        range_dict[key] = range_
        
        print(range_)
    
    sorted_range_dict = sorted(range_dict.items(), key=operator.itemgetter(1))
        
    return(sorted_range_dict)

####################################################################################################################

def create_norm_range_dict_pre_peak_smoothed_optima_post_llo(video_dictionary, window_size, start_frame):
    
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


b = create_norm_range_dict(llo_read_dict)

d = create_norm_range_dict_pre_peak_post_llo(llo_read_dict , 2800)

e = create_norm_range_dict_smoothed_optima(llo_read_dict, 150)

f = create_norm_range_dict_pre_peak_smoothed_optima_post_llo(llo_read_dict, 150, 2800)


eval_ranking(b)

eval_ranking(d)

eval_ranking(e)

eval_ranking(f)







#####################################################################################################################
# 
#### Same as create_norm_range_dict
#
#
#def create_norm_range_dict_pre_peak(video_dictionary):
#    
#    range_dict = {}
#    
#    for key, value in video_dictionary.items():
#    
#        array = np.array(value)
#        
#        frames_before_peak = array[ : np.argmax(array) + 1 ]
#        
#        scaled_frames_before_peak = normalize(frames_before_peak.reshape(1,-1).astype(float))[0]
#        
#        peak_value = np.max(scaled_frames_before_peak)
#        
#        min_value_before_peak = np.min(scaled_frames_before_peak)
#        
#        range_ = peak_value / min_value_before_peak
#        
#        range_dict[key] = range_
#        
#    sorted_range_dict = sorted(range_dict.items(), key=operator.itemgetter(1))
#    
#    return(sorted_range_dict)

#####################################################################################################################
#
#### Same as create_norm_range_dict_smoothed_optima
#
#def create_norm_range_dict_pre_peak_smoothed_optima(video_dictionary, window_size):
#    
#    range_dict = {}
#    
#    for key, value in video_dictionary.items():
#        
#        print(key)
#        
#        max_median_window_value, max_median_window_indices = calc_window_max(value, window_size) # max_median_window_index is on right side of window
#        
#        frames_before_peak = np.array(value)[ : max_median_window_indices[1] + 1]
#        
#        normalized_frames_before_peak = normalize(frames_before_peak.reshape(1,-1).astype(float))[0]
#        
#        max_norm_median_window_value = np.median(normalized_frames_before_peak[ max_median_window_indices[0] : max_median_window_indices[1] ])
#        
#        min_norm_median_window_value_before_peak , _ = calc_window_min(normalized_frames_before_peak, window_size)
#        
#        range_ = max_norm_median_window_value / min_norm_median_window_value_before_peak
#        
#        print(range_)
#        
#        range_dict[key] = range_
#    
#    sorted_range_dict = sorted(range_dict.items(), key=operator.itemgetter(1))
#        
#    return(sorted_range_dict)


