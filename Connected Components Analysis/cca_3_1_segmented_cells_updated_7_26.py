#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 14:24:52 2017

@author: mmrosek
"""
from skimage.filters import threshold_otsu, rank, threshold_local
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from matplotlib.backends.backend_pdf import PdfPages

# Zeros out regions with fewer than desired number of pixels
def remove_small_regions(region_size,label_matrix,sizes):
    mask_size = sizes <= region_size
    remove_pixel = mask_size[label_matrix]
    labels_copy = label_matrix.copy()
    labels_copy[remove_pixel]=0
    return(labels_copy)

plot_write_path = '/home/mmrosek/Documents/segmented_cell_vid_arrays/cca_segmented_cells/'  # Specifies path for plots to be written to

diagonal_connection = [[1,1,1],[1,1,1],[1,1,1]] # Structure that indicates pixels connected diagonally count as connected (used in loop below)

connection_type = 'diagonal' # Specifies connection type for file name upon writing

medium_region_threshold = 3

large_region_threshold = 10

offset = -0.2

# Read in array of segmented cell video
segmented_cell_array_path = '/home/mmrosek/Documents/segmented_cell_vid_arrays/'

video = "DsRed2-HeLa_3_1_LLO (Converted).mov/"

array_name_list = ["hull_1_segmentation_7_22_25.npy", "hull_2_segmentation_7_22_25.npy"]

block_size = array_name_list[0][-6:-4]

threshold_type = 'adaptive_{0}_{1}_offset'.format(block_size,offset) # Number indicates block size

# Opens pdfs to be written to
all_regions_pdf = PdfPages(plot_write_path + video + 'conn_regions_{0}_{1}.pdf'.format(threshold_type, connection_type))
medium_regions_pdf = PdfPages(plot_write_path + video + 'conn_regions_>{0}_{1}_{2}.pdf'.format(medium_region_threshold,threshold_type,connection_type))
large_regions_pdf = PdfPages(plot_write_path + video + 'conn_regions_>{0}_{1}_{2}.pdf'.format(large_region_threshold,threshold_type,connection_type))
total_sizes_pdf = PdfPages(plot_write_path + video  + 'total_size_{0}_{1}.pdf'.format(threshold_type, connection_type))


for array_name in array_name_list:
    
    block_size = array_name[-6:-4]
    
    cell_vid_array = np.load(segmented_cell_array_path + video + array_name)
    
    # Intialize lists to hold data for all videos
    all_vids_medium_regions_dict = {}
    all_vids_large_regions_dict = {}
    all_vids_all_regions_dict = {}
    all_vids_total_size_dict = {}
    
         
    medium_regions_list = []
    large_regions_list = []
    num_labels_list = []
    total_size_list = []

    for frame in range(cell_vid_array.shape[0]):
        
        if frame % 100 == 0:
            print(frame)
    
        image = cell_vid_array[frame,:,:]
        
        labels_diag, num_labels_diag = ndimage.label(image, structure=diagonal_connection)
        
        # Calculates size of each region
        sizes = ndimage.sum(image, labels_diag, range(num_labels_diag + 1))
        
        # Removes regions below thresholds
        medium_regions = remove_small_regions(medium_region_threshold,labels_diag,sizes)
        medium_labels = np.unique(medium_regions)
        num_labels_medium_regions = len(medium_labels)-1 # -1 to exclude 0 which is not truly a label
        
        large_regions = remove_small_regions(large_region_threshold,labels_diag,sizes)
        large_labels = np.unique(large_regions)
        num_labels_large_regions = len(large_labels)-1 # -1 to exclude 0 which is not truly a label
        
        num_labels_list.append(num_labels_diag)
        medium_regions_list.append(num_labels_medium_regions)
        large_regions_list.append(num_labels_large_regions)
        total_size_list.append(np.sum(sizes))
    
    
    all_vids_medium_regions_dict[video] = medium_regions_list
    all_vids_large_regions_dict[video] = large_regions_list
    all_vids_all_regions_dict[video] = num_labels_list
    all_vids_total_size_dict[video] = total_size_list
    
    x = range(cell_vid_array.shape[0])
    
    # Plots number of connected regions
    plt.figure(figsize=(14,9.5))
    plt.rcParams.update({'font.size':26})
    plt.ylabel('Number of regions')
    plt.xlabel('Frame Number')
    plt.title('Connected Regions, {0}, {1}'.format(video[:-17], 'Cell ' + array_name[5:6]))
    plt.plot(x, np.array(num_labels_list), linewidth=0.75)
    plt.savefig(all_regions_pdf, format='pdf')
    plt.savefig(plot_write_path+video+array_name +'_conn_regions_{0}.png'.format(threshold_type),bbox_inches='tight')
    
    # Plots number of connected regions greater than region_threshold
    plt.figure(figsize=(14,9.5))
    plt.rcParams.update({'font.size':26})
    plt.ylabel('Num regions > {} pixels'.format(large_region_threshold))
    plt.xlabel('Frame Number')
    plt.title('Connected Regions (>{0}), {1}, {2}'.format(large_region_threshold,video[:-17], 'Cell ' + array_name[5:6]))
    plt.plot(x, np.array(large_regions_list), linewidth=0.75)
    plt.savefig(large_regions_pdf, format='pdf')
    plt.savefig(plot_write_path+video+array_name +'_conn_regions_>{0}_{1}.png'.format(large_region_threshold,threshold_type),bbox_inches='tight')
    
    # Plots number of connected regions greater than region_threshold
    plt.figure(figsize=(14,9.5))
    plt.rcParams.update({'font.size':26})
    plt.ylabel('Num regions > {} pixels'.format(medium_region_threshold))
    plt.xlabel('Frame Number')
    plt.title('Connected Regions (>{0}), {1}, {2}'.format(medium_region_threshold,video[:-17], 'Cell ' + array_name[5:6]))
    plt.plot(x, np.array(medium_regions_list), linewidth=0.75)
    plt.savefig(medium_regions_pdf, format='pdf')
    plt.savefig(plot_write_path+video+array_name +'_conn_regions_>{0}_{1}.png'.format(medium_region_threshold,threshold_type),bbox_inches='tight')
     
    # Plots total region size
    plt.figure(figsize=(14,9.5))
    plt.rcParams.update({'font.size':26})
    plt.ylabel('Total Size')
    plt.xlabel('Frame Number')
    plt.title('Total Size of All Regions, {0}, {1}'.format(video[:-17], 'Cell ' + array_name[5:6]))
    plt.plot(x, np.array(total_size_list), color = 'g', linewidth=0.75)
    plt.savefig(total_sizes_pdf, format='pdf')
    plt.savefig(plot_write_path + video + array_name + '_total_size_{0}.png'.format(threshold_type),bbox_inches='tight')


# Close PDFs/finish writing process
all_regions_pdf.close()
large_regions_pdf.close()
total_sizes_pdf.close()  
medium_regions_pdf.close()

#
#dict_save_path = '/home/mmrosek/Documents/segmented_cell_vid_arrays/cca_segmented_cells/' 
#
#np.save(dict_save_path + video + array_name + ' llo_large_regions_dict.npy', all_vids_large_regions_dict)
#np.save(dict_save_path + video + array_name + ' llo_medium_regions_dict.npy', all_vids_medium_regions_dict)
#np.save(dict_save_path + video + array_name + ' llo_all_regions_dict.npy', all_vids_all_regions_dict)
#np.save(dict_save_path + video + array_name + ' llo_total_size_dict.npy', all_vids_total_size_dict)

