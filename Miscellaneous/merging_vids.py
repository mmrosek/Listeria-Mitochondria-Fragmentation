import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.filters import threshold_otsu, rank, threshold_local
import imageio
import skimage.filters as filters
import skimage.morphology as morphology
import matplotlib.pyplot as plt
import numpy as np
import skimage
import time
import cv2

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

video = 'DsRed2-HeLa_2_21_LLO (Converted).mov_every_10th_frame.mov'

# Read in video file
video1 = read_in_video("/home/mmrosek/Documents/shannon_vids/{}".format(video), "rgb")



video2 = read_in_video("/home/mmrosek/Documents/shannon_vids/Single Cell Videos/10th_frame_vids/2_21/2_21_7_25/2_21_LLO_hull_2_segmentation_7_25_25.mov", "gray")




#video1 = np.random.randint(0, 256, size=(nframes, 300, 300), dtype=np.uint8)
#video2 = np.random.randint(0, 256, size=(nframes, 300, 300), dtype=np.uint8)

#ax1.axis('off')
#ax2.axis('off')

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
fig.subplots_adjust(hspace=0, wspace=0)

ax1 = plt.Axes(fig, [0., 0., 0.5, 1])
ax2 = plt.Axes(fig, [0.5, 0., 0.5, 1])

ax1.set_axis_off()
ax2.set_axis_off()
fig.add_axes(ax1)
fig.add_axes(ax2)

im1 = ax1.imshow(video1[0], interpolation='none', aspect='auto', animated=True)
im2 = ax2.imshow(video2[0], interpolation='none', aspect='auto', cmap='gray', animated=True)

nframes = 800

ffmpegwriter = animation.writers['ffmpeg']
writer = ffmpegwriter(fps=100, bitrate=5000)

with writer.saving(fig, '/home/mmrosek/Documents/final_prez/2_21_vids_merged_shared_axes.mov', 100):
    for i in range(nframes):
        if i % 100 == 0:
            print('frame {}'.format(i))
        im1.set_array(video1[i])
        im2.set_array(video2[i])
        writer.grab_frame()