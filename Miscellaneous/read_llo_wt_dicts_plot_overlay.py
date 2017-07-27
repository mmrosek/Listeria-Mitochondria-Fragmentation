#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:46:24 2017

@author: mmrosek
"""
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, scale
from matplotlib.backends.backend_pdf import PdfPages


wt_dict_read_path = '/home/mmrosek/Documents/wt_region_dicts/' 

llo_dict_read_path = '/home/mmrosek/Documents/llo_region_dicts/' 

plot_write_path = '/home/mmrosek/Documents/cca_images/llo_wt_overlayed/wt_29000-40000th_frames_no_long_llo/'

dict_list = ['large_regions', 'medium_regions', 
             'all_regions','total_size']

threshold_type = 'adpative_45'

for dict_type in dict_list: 
    
    data_type = dict_type.split('_')[0].capitalize() + " " + dict_type.split('_')[1].capitalize()
    
    print(data_type)
    
    pdf = PdfPages(plot_write_path + 'full_llo_mid_wt_{0}_{1}.pdf'.format(dict_type,threshold_type))

    llo_read_dict = np.load(llo_dict_read_path + 'llo_' + dict_type + '_dict_45.npy').item()
    
    wt_read_dict = np.load(wt_dict_read_path + 'wt_' + dict_type + '_dict_45.npy').item()
    
    
    # Calculate min frame num and number of llo vids
    min_frame_num = 10000000000000
    llo_video_count = 0
    for key , value in llo_read_dict.items():
        if len(value)  < min_frame_num:
            min_frame_num = np.array(value).shape[0]
        llo_video_count += 1
        
        
    # Calculate number of wt vids
    wt_video_count = 0
    num_slices_per_vid_list = [] # Indicates # of times current vid can be divided into slices of min_frame_num length
    for key , value in wt_read_dict.items():
        
        if 'LLO' in key:
                
            print(key)
                
            continue
        
        if len(value)  < min_frame_num:
            min_frame_num = np.array(value).shape[0]
        wt_video_count += 1
        
  
    # Initialize arrays
    llo_vids_array = np.zeros(shape=[llo_video_count,min_frame_num-1])
    wt_vids_array = np.zeros(shape=[wt_video_count,min_frame_num-1])
    
    print(wt_vids_array.shape)
    
    # Concatenate llo arrays into single matrix
    count = 0
    for key , value in llo_read_dict.items():
        llo_vids_array[count,:min_frame_num-1] = np.array(value[: min_frame_num-1])
        count += 1
    
    # Concatenate wt arrays into single matrix
    wt_frame_buffer = 29000 # Used to crop out the initial intensity increase that skews plots
    count = 0
    for key , value in wt_read_dict.items():
        
        if 'LLO' in key:
                
            print(key)
                
            continue
        
        wt_vids_array[count,:min_frame_num-1] = np.array(value[ wt_frame_buffer : min_frame_num-1 + wt_frame_buffer])
        count += 1
    
    # Scale arrays across rows/videos
    llo_array_scaled = scale(llo_vids_array, axis = 1)
    wt_array_scaled = scale(wt_vids_array, axis = 1)
            
    # Calculate median/mean values for each frame across all videos 
    mean_llo_array = np.sum(llo_vids_array, axis = 0)/llo_video_count # Axis = 0 takes mean down each column
    scaled_mean_llo_array = np.sum(llo_array_scaled, axis = 0)/llo_video_count 
    median_llo_array = np.median(llo_vids_array, axis = 0)
    scaled_median_llo_array = np.median(llo_array_scaled, axis = 0)
    
    mean_wt_array = np.sum(wt_vids_array, axis = 0)/wt_video_count # Axis = 0 takes mean down each column
    scaled_mean_wt_array = np.sum(wt_array_scaled, axis = 0)/wt_video_count 
    median_wt_array = np.median(wt_vids_array, axis = 0)
    scaled_median_wt_array = np.median(wt_array_scaled, axis = 0)
    
    
    # Determine range of x-axis
    x = range(llo_vids_array.shape[1])
    
    # Determine ylabel
    if data_type == 'Total Size':
        ylabel = 'Total Size (pixels)'
    else:
        ylabel = 'Number of Regions'
    
     # Plot mean array
    plt.ylabel(ylabel)
    plt.xlabel('Frame Number')
    plt.title('Average {} vs. Frame'.format(data_type))
    plt.plot(x, mean_llo_array, color = 'g', linewidth=0.075)
    plt.plot(x, mean_wt_array, color = 'b', linewidth=0.075)
    llo_patch = mpatches.Patch(color='g', label='LLO')
    wt_patch = mpatches.Patch(color='b', label='WT')
    plt.legend(handles=[llo_patch,wt_patch])
    plt.savefig(pdf, format='pdf')
    plt.savefig(plot_write_path+'Individual Plots/Average_{}_All_Vids.png'.format(data_type),bbox_inches='tight')
    plt.show()   
    
    # Plot mean scaled array
    plt.ylabel(ylabel)
    plt.xlabel('Frame Number')
    plt.title('Scaled Average {} vs. Frame'.format(data_type))
    plt.plot(x, scaled_mean_llo_array, color = 'g', linewidth=0.075)
    plt.plot(x, scaled_mean_wt_array, color = 'b', linewidth=0.075)
    llo_patch = mpatches.Patch(color='g', label='LLO')
    wt_patch = mpatches.Patch(color='b', label='WT')
    plt.legend(handles=[llo_patch,wt_patch])
    plt.savefig(pdf, format='pdf')
    plt.savefig(plot_write_path+'Individual Plots/Scaled_Average_{}_All_Vids.png'.format(data_type),bbox_inches='tight')
    plt.show()   
    
    # Plot median array
    plt.ylabel(ylabel)
    plt.xlabel('Frame Number')
    plt.title('Median {} vs. Frame'.format(data_type))
    plt.plot(x, median_llo_array, color = 'g', linewidth=0.075)
    plt.plot(x, median_wt_array, color = 'b', linewidth=0.075)
    llo_patch = mpatches.Patch(color='g', label='LLO')
    wt_patch = mpatches.Patch(color='b', label='WT')
    plt.legend(handles=[llo_patch,wt_patch])
    plt.savefig(pdf, format='pdf')
    plt.savefig(plot_write_path+'Individual Plots/Median_{}_All_Vids.png'.format(data_type),bbox_inches='tight')
    plt.show() 
    
    # Plot median array
    plt.ylabel(ylabel)
    plt.xlabel('Frame Number')
    plt.title('Scaled Median {} vs. Frame'.format(data_type))
    plt.plot(x, scaled_median_llo_array, color = 'g', linewidth=0.075)
    plt.plot(x, scaled_median_wt_array, color = 'b', linewidth=0.075)
    llo_patch = mpatches.Patch(color='g', label='LLO')
    wt_patch = mpatches.Patch(color='b', label='WT')
    plt.legend(handles=[llo_patch,wt_patch])
    plt.savefig(pdf, format='pdf')
    plt.savefig(plot_write_path+'Individual Plots/Scaled_{}_All_Vids.png'.format(data_type),bbox_inches='tight')
    plt.show() 
    
    # Close PDFs/finish writing process
    pdf.close()
