from __future__ import division
import glob
import imageio
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score
import adjusted_rand_index
from multiprocessing import Pool
from skimage.measure import block_reduce
from math import ceil
from skimage import feature
from hash_images import generate_video_representation
def sub_rand_index(clusters):
	elems = list(set.union(*truth))

	# Index of Containing Set
	memory_truth = {}
	memory_clusters = {}
	def ics(element, set_list, set_list_name):
		if set_list_name == "truth":
			if element in memory_truth:
				return memory_truth[element]
		if set_list_name == "clusters":
			if element in memory_clusters:
				return memory_clusters[element]

		for c, s in enumerate(set_list):
			if element in s:
				if set_list_name == "truth":
					memory_truth[element] = c
				if set_list_name == "clusters":
					memory_clusters[element] = c
				return c

	x = map(lambda e: ics(e, clusters, 'clusters'), elems)
	y = map(lambda e: ics(e, truth, 'truth'), elems)

	return adjusted_rand_score(x,y)

def filenames_gen(n_clusters=None):
    if n_clusters == None:
        for i in glob.glob('./videos/*.mp4'):
            yield i
    truth = adjusted_rand_index.get_truth()
    filenames = []
    for i in range(n_clusters):
        set = truth[i]
        for name in set:
            filenames.append('./videos/'+name +'.mp4')
            yield './videos/'+name +'.mp4'
def path_to_name(v_path):
	return v_path.split('/')[-1].split('.')[0]

def get_frames(filename,n_frames=2,img_crop=1,video_crop=1):
    def crop_center(img,img_crop):
        x,y,c = img.shape
        cropx = x // img_crop
        cropy = y// img_crop
        startx = x//2 - cropx//2
        starty = y//2 - cropy//2
        return img[startx:startx+cropx,starty:starty+cropy, :]
    def crop_video(video,video_crop=1):
        length_video = video.get_length()
        crop = length_video//video_crop
        start = length_video//2 - crop//2
        return start, crop
    video = imageio.get_reader(filename)
    start,length_video = crop_video(video,video_crop)
    img = video.get_data(0)
    s = [int(img.shape[0]/img_crop), int(img.shape[1]/img_crop)]
    frames = np.zeros((n_frames,s[0],s[1],3))
    for n in range(n_frames):
        idx_frame = start + int((n/(n_frames-1))*(length_video-1))
        frames[n,] = crop_center(video.get_data(idx_frame),img_crop)
    return frames
def get_color_features(frames,frame_div=3):
	s = [int(ceil(i/frame_div)) for i in frames.shape]
	x = block_reduce(frames,block_size=(1, s[1], s[2], 1), func=np.mean)
	x = x.reshape(-1,3)
	x = normalize(x,axis=1)
	#x = np.mean(x,axis=0)
	x = x.flatten()
	return x

def get_average_edge_index(frames,frame_div=3):
	frames = np.mean(frames,axis=3)
	edges = [feature.canny(i,sigma = 4) for i in frames]
	edges = np.stack(edges)
	s = [int(ceil(i/frame_div)) for i in frames.shape]
	x = block_reduce(edges,block_size=(2,s[1], s[2]), func=np.mean)
	x = x.flatten()
	x = x/np.linalg.norm(x)
	return x
    #print np.mean(edges)
def create_LSH_vector(filename):
	# n_frames = 2
	# img_crop = 1
	# video_crop = 1
	# frames = get_frames(filename,n_frames,img_crop,video_crop)
	# frame_div = 3
	# x0 = get_color_features(frames,frame_div)
	# x1 = get_average_edge_index(frames,frame_div)
	# x = np.concatenate((x0, x1), axis=0)

	x = generate_video_representation(filename)
	return (filename,x)
def get_filenames_X(Results):
        X = []
        list_filenames = []
        for filename,x in Results:
            X.append(x)
            list_filenames.append(filename)
        return list_filenames,np.array(X)
def cluster(X,n_clusters):
    pred = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)
    return pred
def pred_to_clusters(list_filenames,pred,n_clusters):
    names = [path_to_name(path) for path in list_filenames]
    clusters = [set() for _ in range(n_clusters)]
    for idx , clus in enumerate(pred):
        clusters[clus].add(names[idx])
    return adjusted_rand_index.rand_index(clusters,n_clusters)

n_clusters = 2
print("Hashing")

p = Pool(4)
Results = p.imap(create_LSH_vector,filenames_gen(n_clusters),chunksize = 4)
list_filenames,X = get_filenames_X(Results)
print("Clustering")
pred = cluster(X,n_clusters)
print pred_to_clusters(list_filenames,pred,n_clusters)
