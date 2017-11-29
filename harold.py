from __future__ import division
import glob
import cv2
import imageio
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score
import adjusted_rand_index

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

def load_filenames(n_clusters=None):
    if n_clusters == None:
        return glob.glob('./videosub/*.mp4')
    truth = adjusted_rand_index.get_truth()
    filenames = []
    for i in range(n_clusters):
        set = truth[i]
        for name in set:
            filenames.append('./videos/'+name +'.mp4')
    return filenames

def path_to_name(v_path):
	return v_path.split('/')[-1].split('.')[0]
def get_frames(n_frames,filename):
    video = imageio.get_reader(filename)
    length_video = video.get_length()-1
    img = video.get_data(0)
    frames = np.zeros((n_frames,img.shape[0],img.shape[1],img.shape[2]))
    for n in range(n_frames):
        idx_frame = int((n/(n_frames-1))*length_video)
        frames[n,] = video.get_data(idx_frame)
    return frames
def create_LSH_matrix(list_filenames):
    N = len(list_filenames)
    n_frames = 2
    n_squares = 4
    X = np.zeros((N,n_frames,3))
    for i,filename in enumerate(list_filenames):
        if i%100 == 99:
            print i
        frames = get_frames(n_frames,filename)
        X[i,] += np.mean(frames,axis=(1,2))
        X[i,] = normalize(X[i,],axis=1)
    X = X.reshape((N,n_frames*3))
    return X

def cluster(X,n_clusters):
    pred = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)
    return pred
def pred_to_clusters(pred,n_clusters):
    names = [path_to_name(path) for path in list_filenames]
    clusters = [set() for _ in range(n_clusters)]
    for idx , clus in enumerate(pred):
        clusters[clus].add(names[idx])
    return adjusted_rand_index.rand_index(clusters,n_clusters)

n_clusters = 10
list_filenames = load_filenames(n_clusters)

print("Hashing")
X = create_LSH_matrix(list_filenames)
print("Clustering")
pred = cluster(X,n_clusters)
print pred_to_clusters(pred,n_clusters)
