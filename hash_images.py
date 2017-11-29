import imageio
from skimage.measure import block_reduce
import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize
from math import ceil
from skimage import feature

"""
Function which, both convert image to numpy array and do 2 by 2 max-pooling and resizes the image
"""
def to_numpy_pooling(meta_vid):
	pooled	= block_reduce(np.asarray(meta_vid), block_size = (2,2,1), func = np.max)
	return pooled

"""
Function which averages a list of frames
"""
def average_frames(list_frames):

	added_frames = np.zeros(list_frames[0].shape)
	for arr in list_frames:
		added_frames += arr

	return added_frames / len(list_frames)
	# return reduce(np.add, list_frames) / len(list_frames)

"""
Function which devides an image into 10 chunks of frames - each frame represented by a numpy array
"""
def get_frame_chunks(filename, n_chunks = 5):
	video 			= imageio.get_reader(filename)
	n_frames		= video.get_length()
	frame_interval	= n_frames / n_chunks

	intervals 		= [xrange(frame_interval*i, frame_interval*(i + 1)) for i in range(0, n_chunks - 1)]
	last			= xrange(frame_interval*(n_chunks - 1 ), n_frames)
	intervals.append(last)

	return map(average_frames,[map(to_numpy_pooling, [video.get_data(idx) for idx in idx_list]) for idx_list in intervals])


"""
Function which provides a fixed number of frames, evenly spaced
"""
def get_frames(filename, n_frames = 20):
	video 			= imageio.get_reader(filename)
	length			= video.get_length()
	frame_interval	= length / n_frames
	trail_lead_idx	= int(round(length % float(n_frames) / 2))
	idx_list		= range(trail_lead_idx, length - trail_lead_idx, frame_interval)
	return [video.get_data(idx) for idx in idx_list]

"""
Resize function using Pillows - convert from numpy array to image and back to numpy array
"""
def resize(frame):
	#using LANCZOS for convolution "http://pillow.readthedocs.io/en/3.1.x/releasenotes/2.7.0.html"
	size 	= (20,20)
	i 		= Image.fromarray(frame)
	return np.asarray(i.resize(size, Image.LANCZOS))


"""
Function which average pixel values at distances from the center of the image. Then average all these averages.
"""
def rotation_invariant_feature(frame):
	w, h, _ 				= frame.shape
	cx, cy					= w/2, h/2
	max_dist				= min(w/2, h/2)
	avg_pixel_cirles		= np.zeros((max_dist,3))
	count_element_at_dist 	= np.zeros(max_dist)

	# add pixel values lying on same circle together
	for i, row in enumerate(frame):
		xdist	= abs((cx-i))
		if xdist > max_dist:
			continue
		for j, element in enumerate(row):
			ydist 	= abs((cy-j))
			if ydist > max_dist or ydist**2 + xdist**2 > max_dist**2:
				continue
			dist_index 							= int(np.sqrt(xdist**2 + ydist**2))-1
			avg_pixel_cirles[dist_index,] 		+= element
			count_element_at_dist[dist_index] 	+= 1

	# calculate average
	avg_pixel_cirles = avg_pixel_cirles.T / count_element_at_dist

	# return average of averages
	return avg_pixel_cirles.T

"""
Function which average pixel values along horizontal and vertical axis, relative to the center pixel
"""
def translation_invariant_feature(frame):
	w, h, _ 				= frame.shape
	cx, cy					= w/2, h/2

	# find average pixel value along x-axis and y-axis
	return np.mean(frame[cx,], axis = 0), np.mean(frame[:,cy], axis = 0)

def get_color_features(frames,frame_div=3):
	s = [int(ceil(i/frame_div)) for i in frames.shape]
	x = block_reduce(frames,block_size=(1, s[1], s[2], 1), func=np.mean)
	x = x.reshape(-1,3)
	x = normalize(x,axis=1)
	x = np.mean(x,axis=0)
	return x

def get_edge_features(frames,frame_div=3):
	frames 	= np.mean(frames,axis=3)
	edges 	= [feature.canny(i,sigma = 4) for i in frames]
	edges 	= np.stack(edges)
	s 		= [int(ceil(i/frame_div)) for i in frames.shape]
	x	 	= block_reduce(edges,block_size=(2,s[1], s[2]), func=np.mean)
	x 		= x.flatten()
	x 		= x/np.linalg.norm(x)
	return x

"""
Function used to visualize images from numpy array (used for debugging and testing)
"""
def visualize_singe_image(numpy_array):
	# convert numpy array to unsigned type uint8. Otherwise it cannot be shown
	Image.fromarray(numpy_array.astype('uint8')).show()


"""
Function which flattens a list of lists
"""
def flattener(alist):
	return [element for sublist in alist for element in sublist]


"""
This function takes a single video and transforms it using LSH
"""
def generate_video_representation(vid):
	#frames					= get_frame_chunks(vid)
	frames					= get_frames(vid)
	pooled					= map(to_numpy_pooling, frames)
	pooled_resized			= map(resize, pooled)
	# calculate total average
	tot_avg					= average_frames(pooled_resized)

	# get rotation invariant features averaged across all frames
	rotation_features		= flattener(normalize(rotation_invariant_feature(tot_avg)))
	
	# stacking
	pooled = np.stack(pooled)
	color_features			= get_color_features(pooled).tolist()
	edge_features			= get_edge_features(pooled).tolist()

	total_features 			= rotation_features + color_features + edge_features
	return rotation_features
