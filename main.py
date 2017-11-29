import glob
import os
import imageio
from adjusted_rand_index import rand_index, get_truth
import argparse
from hash_images import generate_video_representation
import numpy as np
from cluster_images import cluster_videos_kmeans
# extract the video names from the paths
def path_to_name(v_path):
	return v_path.split('/')[-1].split('.')[0]

# testing code - getting only a subset of true clusters
def load_filenames(n_clusters=970):
    if n_clusters == 970:
        return glob.glob('./videosub/*.mp4')
    truth = get_truth()[0:n_clusters]
    filenames = ['./videos/'+name +'.mp4'
    			for set_of_names in truth 
    			for name in set_of_names]
    return filenames

# the main function, which solves the challenge
def main(video_files, n_clusters):
	video_files	= load_filenames(n_clusters = n_clusters)
	video_names = map(path_to_name, video_files)
	
	#generate_video_representation(video_files[0])
	videos		= np.asarray(map(generate_video_representation, video_files))
	
	clusters	= cluster_videos_kmeans(videos, video_names, n_clusters)
	score 		= rand_index(clusters, n_clusters)
	
	print score



if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--n_clusters", type = int, default = 970, choices = range(971), help = "Specify number of clusters to use for testing")

	args 		= parser.parse_args()
	n_clusters	= args.n_clusters

	# find paths to all videos
	video_files	= glob.glob('./videos/*.mp4')
	
	# run main function
	main(video_files, n_clusters)
