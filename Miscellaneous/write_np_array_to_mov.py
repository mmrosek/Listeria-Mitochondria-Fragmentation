#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 09:17:46 2017

@author: mmrosek
"""
import imageio
import numpy as np

##################################################################################################################################

# Grab every 10th frame and write to a new .mov file

file_list = ['DsRed2-HeLa_2_21_LLO (Converted).mov','DsRed2-HeLa_3_15_LLO1part1 (Converted).mov','DsRed2-HeLa_3_15_LLO2part1 (Converted).mov','DsRed2-HeLa_3_1_LLO (Converted).mov',
'DsRed2-HeLa_3_23_LLO_1 (2) (Converted).mov','DsRed2-HeLa_3_31_LLO_1 (2) (Converted).mov',
'DsRed2-HeLa_3_31_LLO_2 (2) (Converted).mov','DsRed2-HeLa_3_31_LLO_3 (2) (Converted).mov',
'DsRed2-HeLa_3_8_LLO002 (Converted).mov',  'DsRed2-HeLa_3_9_LLO (Converted).mov',
'DsRed2-HeLa_4_5_LLO1 (Converted).mov','DsRed2-HeLa_4_5_LLO2 (Converted).mov']

file_list = ['DsRed2-HeLa_3_23_LLO_1 (2) (Converted).mov']

#for video in file_list:
#
#    reader = imageio.get_reader("/home/mmrosek/Documents/shannon_vids/{}".format(video))
#    fps = reader.get_meta_data()['fps']
#    
#    writer = imageio.get_writer("/home/mmrosek/Documents/shannon_vids/{0}_{1}.mov".format(video,'every_10th_frame'), fps=fps)
#    
#    for idx,im in enumerate(reader):
#        
#        if idx % 10 == 0:
#        
#            writer.append_data(im)
#        
#    writer.close()


###################################################################################################################################

# Write binarized numpy array to .mov

def write_np_array_to_mov(array_read_path, vid_name, array_name):
    
    vid_date = vid_name[12:20]
       
    reader = np.load(array_read_path + vid_name + '/' + array_name)
    
    video_write_name = vid_date + '_' + array_name[:-3]
    
    real_value_array = np.where(reader==1, 255, 0) # added uint8 cus couldnt read the whole vid in, could do in pieces
    
    writer = imageio.get_writer("/home/mmrosek/Documents/shannon_vids/Single Cell Videos/{0}.mov".format(video_write_name), fps=100)
    
    for idx in range(real_value_array.shape[0]):
    
        writer.append_data(real_value_array[idx])
        
    writer.close()


for i in range(1,2):

    write_np_array_to_mov('/home/mmrosek/Documents/segmented_cell_vid_arrays/', 'DsRed2-HeLa_3_23_LLO_1 (2) (Converted).mov_every_10th_frame.mov', "hull_1_segmentation_7_25_25.npy")

#for i in range(1,2):
#
#    write_np_array_to_mov('/home/mmrosek/Documents/segmented_cell_vid_arrays/', 'DsRed2-HeLa_2_21_LLO (Converted).mov_every_10th_frame.mov', "hull_2_segmentation_7_25_25.npy")
#



##################################################################################################################################

#
#vid_name = 'DsRed2-HeLa_4_5_LLO1 (Converted).mov'
#
#array_name = "hull_2_segmentation_7_15_35.npy"


def write_full_length_np_array_to_mov(array_read_path, vid_name, array_name):
    
    vid_date = vid_name[12:20]
       
    reader = np.load(array_read_path + vid_name + '/' + array_name)
    
    video_write_name = vid_date + '_' + array_name[:-3]
    
    slice_frame_range = 4000
    
    for slice_num in range(int(np.ceil(reader.shape[0]/slice_frame_range))):
        
        print('Writing slice {}'.format(slice_num+1))
    
        real_value_array = np.where(reader[slice_num * slice_frame_range : (slice_num + 1) * slice_frame_range] == 1, 255, 0) 
          
        writer = imageio.get_writer("/home/mmrosek/Documents/shannon_vids/Single Cell Videos/{0}_slice_{1}.mov".format(video_write_name, slice_num+1), fps=100)
        
        for idx in range(real_value_array.shape[0]):
        
            writer.append_data(real_value_array[idx])
            
        writer.close()
    
        del real_value_array


write_full_length_np_array_to_mov('/home/mmrosek/Documents/segmented_cell_vid_arrays/', 'DsRed2-HeLa_3_1_LLO (Converted).mov', "hull_2_segmentation_7_22_25.npy")









#segmented_cell_array_path = '/home/mmrosek/Documents/segmented_cell_vid_arrays/'
#
#video = "DsRed2-HeLa_4_5_LLO2 (Converted).mov_every_10th_frame.mov/"
#
#vid_date = video[12:20]
#
#array_name = "hull_2_segmentation_7_13_35.npy"
#
#reader = np.load(segmented_cell_array_path + video + array_name)
#
#video_write_name = vid_date + '_' + array_name[:-5]
#
#x = np.where(reader==1, 255, 0)
#
#writer = imageio.get_writer("/home/mmrosek/Documents/shannon_vids/Single Cell Videos/{0}.mov".format(video_write_name), fps=100)
#
#for idx in range(x.shape[0]):
#
#    writer.append_data(x[idx])
#    
#writer.close()
