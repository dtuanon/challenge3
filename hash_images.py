import imageio
import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize
from math import ceil
from skimage import feature
import imagehash as ih

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





"""
Function which, both convert image to numpy array and do 2 by 2 max-pooling
"""
def to_numpy_pooling(meta_vid):
	pooled	= pooling(np.asarray(meta_vid), block_size = (2,2), func = np.max)
	return pooled.astype('uint8')

"""
Function which averages a list of frames
"""
def average_frames(list_frames):

	added_frames = np.zeros(list_frames[0].shape)
	for arr in list_frames:
		added_frames += arr

	return added_frames / len(list_frames)

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
def get_frames(filename, n_frames = 10):
	video 			= imageio.get_reader(filename)
	length			= video.get_length()
	frame_interval	= length / n_frames
	trail_lead_idx	= int(round(length % float(n_frames) / 2))
	idx_list		= range(trail_lead_idx, length - trail_lead_idx, frame_interval)
	return [video.get_data(idx) for idx in idx_list]

"""
Resize function using Pillows - convert from numpy array to image and back to numpy array
"""
def resize(frame, w = 20, h = 20):
	#using LANCZOS for convolution "http://pillow.readthedocs.io/en/3.1.x/releasenotes/2.7.0.html"
	size 	= (w,h)
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

#def get_color_features(frames,frame_div=3):
#	s = [int(ceil(i/frame_div)) for i in frames.shape]
#	x = pooling(frames,block_size=(1, s[1], s[2]), func=np.mean)
#	x = x.reshape(-1,3)
#	x = normalize(x,axis=1)
#	x = np.mean(x,axis=0)
#	return x

"""
Function which gets normalized color buckets, by dividing a frame into squares
"""
def square_color_bucket(frames):
	frame 				= np.mean(frames, axis = 0)
	w, h, _ 			= [i/3 for i in frame.shape]
	x					= resize(frame.astype('uint8'), (w - w % 3), (h - h % 3))
	w_block, h_block, _ = [i/3 for i in x.shape]
	x					= pooling(x, block_size = (w_block, h_block), func = np.mean)
	x					= normalize(x.reshape(-1, 3))
	return	x.ravel()


#def get_edge_features(frames,frame_div=3):
#	frames 	= np.mean(frames,axis=3)
#	edges 	= [feature.canny(i,sigma = 4) for i in frames]
#	edges 	= np.stack(edges)
#	s 		= [int(ceil(i/frame_div)) for i in frames.shape]
#	x	 	= pooling(edges,block_size=(2,s[1], s[2]), func=np.mean)
#	x 		= x.flatten()
#	x 		= x/np.linalg.norm(x)
#	return x

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
	return np.asarray([element for sublist in alist for element in sublist])


"""
Function which calculates aspect-ratio
"""
def aspect_ratio(frame):
	w, h, _	= frame.shape
	return np.asarray([w/float(h)])

"""
Function whichw weights features
"""
def weight_features(do_weight, *args):
	if do_weight:
		# find number of features
		n_features 		= map(len, args)
		total_features	= float(sum(n_features))
		# calculated weights
		importance		= map(lambda x: x/total_features, n_features)
		norm 			= sum(map(lambda x: 1/x,importance))
		return map(lambda x: 1/(x*norm), importance), args
	else:
		return [1] * len(args), args
	

def image_hash(frames):
    aHash = []
    pHash = []
    dHash = []
    wHash = []
    for image in frames:
        aHash.append(ih.average_hash(image))
        pHash.append(ih.perception_hash(image))
        dHash.append(ih.difference_hash(image))
        wHash.append(ih.wavelet_hash(image))
    return aHash, pHash, dHash, wHash
    

"""
This function takes a single video and transforms it using LSH
"""
def generate_video_representation(vid, do_weight):
    #frames					= get_frame_chunks(vid)
    frames					= get_frames(vid)
    """
    # without pooling
    pooled					= map(np.asarray, frames)
    
    # with pooling
    #pooled					= map(to_numpy_pooling, frames)
	
    pooled_resized			= map(resize, pooled)
    # calculate total average
    tot_avg					= average_frames(pooled_resized)

    # get rotation invariant features averaged across all frames
    rotation_features		= normalize(rotation_invariant_feature(tot_avg)).ravel()
	
    # stacking
    pooled = np.stack(pooled)
    color_features			= get_color_features(pooled)
    edge_features			= get_edge_features(pooled)
    
    #square colors
    square_color_feature	= square_color_bucket(pooled)
    
    # get aspect ratio 
    asp						= aspect_ratio(frames[0])
    """
    aHash, pHash, dHash, wHash   = image_hash(frames)                 

    # find weights for each set of features
    weights, featurelist	= weight_features(do_weight, aHash, pHash, dHash, wHash) #rotation_features, square_color_feature, asp)
    total_features			= np.concatenate(map(lambda x: x[1] / x[0], zip(weights, featurelist)))
    return total_features.tolist()
