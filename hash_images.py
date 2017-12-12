import imageio
import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize
from math import floor
import imagehash as ih

"""
Resize function using Pillows - convert from numpy array to image and back to numpy array
"""
def resize(frame, w = 20, h = 20):
	#using LANCZOS for convolution "http://pillow.readthedocs.io/en/3.1.x/releasenotes/2.7.0.html"
	size 	= (w,h)
	i 		= Image.fromarray(frame)
	return np.asarray(i.resize(size, Image.LANCZOS))

"""
Function which averages a list of frames
"""
def average_frames_grey_scale(frames):
	frame 	= np.mean(frames, axis = 0)
	# make into an Image
	i 		= Image.fromarray(frame.astype('uint8'))
	# convert image to grey scale
	i		= i.convert(mode = "L")
	return np.asarray(i)

"""
Function which provides a fixed number of frames, evenly spaced
"""
def get_frames(filename, n_frames = 10):
	"""
	Function that crops frames in order to reduce effect of added margins
	"""
	def crop_center(img,portion):
	    x,y,c = img.shape
	    cropx = int(floor(x*portion))
	    cropy = int(floor(y*portion))
	    startx = x//2 - cropx//2
	    starty = y//2 - cropy//2
	    return img[startx:startx+cropx,starty:starty+cropy, :]
	video 			= imageio.get_reader(filename)
	length			= video.get_length()
	frame_interval	= length / n_frames
	if frame_interval !=0:
		trail_lead_idx	= int(round(length % float(n_frames) / 2))
		idx_list	= range(trail_lead_idx, length - trail_lead_idx, frame_interval)
	else:
		idx_list = range(length)
	return [crop_center(video.get_data(idx),0.92) for idx in idx_list]

"""
Function which calculates aspect-ratio
"""
def get_aspect_ratio(frame):
	w, h, _	= frame.shape
	return np.asarray([w/float(h)])

"""
Function which fills up bins from a list of hashes
"""
def get_wHash_features(frames):
	"""
	Function that hashes frames using LSH wHash function
	"""
	def image_hash(frames):
	    wHash = ""
	    for image in frames:
	        image = Image.fromarray(image)
	        wHash += str(ih.whash(image))

	    #Converting hexadecimal hash into hash with elements from 0-15
	    wHash = np.asarray([int(char,16) for char in wHash])
	    return wHash
	# Hash frames using Discrete Wavelet Transformation (DWT: wHash)
	hash_list							= image_hash(frames)
	fb = np.zeros(16)
	for Hash in hash_list:
		fb[Hash % 16] += 1
		return fb
"""
Function which average pixel values at distances from the center of the image. Then average all these averages.
"""
def get_rotation_invariant_features(frames):
	# Resize frames to be 20x20 and stack them
	pooled_resized_frames	= np.stack(map(resize, frames))

	# Average 30 frames element/pixel-wise and convert to gray-scale
	tot_avg_grey			= average_frames_grey_scale(pooled_resized_frames)

	w, h,					= tot_avg_grey.shape
	cx, cy					= w/2, h/2
	max_dist				= min(w/2, h/2)
	avg_pixel_cirles		= np.zeros(max_dist)
	count_element_at_dist 	= np.zeros(max_dist)

	# add pixel values lying on same circle together
	for i, row in enumerate(tot_avg_grey):
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
	return avg_pixel_cirles.reshape(1,-1)

"""
Function which gets normalized color buckets, by dividing a frame into squares
"""
def get_square_color_features(frames):
	"""
	Pooling function - which takes a block matrix of an image and maps all pixels using func
	"""
	def pooling(numpy_array, block_size = (2,2), func = np.max):
		# if numpy_array is not perfectly devisible by block_size, the last row/column will be cropped
		h, w, c = numpy_array.shape
		interval_h, interval_w = block_size
		new_array	= np.zeros((h/interval_h, w/interval_w,c))
		for i in range(h/interval_h):
			for j in range(w/interval_w):
				for k in range(c):
					new_array[i,j,k] = func(numpy_array[interval_h*i:interval_h*(i+1), interval_w*j:interval_w*(j+1), k])

		return new_array
	#  Stacking the original 30 frames
	frames				= np.stack(frames)
	# meaning over the frames
	avg_frame 			= np.mean(frames, axis = 0)
	# resize the frame to divide it by 9
	w, h, _ 			= [i/3 for i in avg_frame.shape]
	avg_frame			= resize(avg_frame.astype('uint8'), (w - w % 3), (h - h % 3))
	#divide it into 9 squares and mean the colors for each square
	w_block, h_block, _ = [i/3 for i in avg_frame.shape]
	x					= pooling(avg_frame, block_size = (w_block, h_block), func = np.mean)
	# normalize the color axis
	x					= normalize(x.reshape(-1, 3))
	return	x.ravel()
"""
Function which weights features
"""
def weight_features(do_weight, *args):
	# find number of features
	n_features 		= map(len, args)
	total_features	= float(sum(n_features))
	if do_weight:
		# calculated weights
		importance		= map(lambda x: x/total_features, n_features)
		norm 			= sum(map(lambda x: 1/x,importance))
		hash_weight		= map(lambda x: 1/(x*norm), importance)
		weights			= []
		for nf, hash_w in zip(n_features, hash_weight):
			weights += [hash_w]*int(nf)
		return np.asarray(weights), args
	else:
		return np.asarray([1] * int(total_features)), args


"""
This function takes a single video and transforms it using LSH
"""
def generate_video_representation(vid, do_weight):

    # Extract 30 frames from video
	frames							= get_frames(vid, n_frames = 30)

    # Store as numpy array
	numpy_frames					= map(np.asarray, frames)

	#  Get rotation invariant features averaged across all frames
	rotation_invariant_features		= normalize(get_rotation_invariant_features(numpy_frames)).ravel()

	# square colors
	square_color_features			= get_square_color_features(numpy_frames)

	# Get aspect ratio
	asp_ratio_feature				= get_aspect_ratio(frames[0])

    # Create a frequency bucket (16 bins) from the 30 hashes
	wHash_features					= get_wHash_features(frames)

	# Find weights for each set of features
	weights, featurelist	= weight_features(do_weight, wHash_features, asp_ratio_feature, rotation_invariant_features, square_color_features)
	total_features			= np.concatenate(featurelist)
	return total_features.tolist(), weights
