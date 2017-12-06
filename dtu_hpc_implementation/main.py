import glob
import os
import imageio
from adjusted_rand_index import rand_index, get_truth
import argparse
from hash_images import generate_video_representation
import numpy as np
import time
from cluster_images import cluster_videos_kmeans, cluster_videos_gmm, cluster_videos_ac
from mpi4py import MPI

"""
 set path to downloaded ffmpeg directory (downloaded by imageio - for correct version)
 the ffmpeg is located in .imageio in user home folder.

 HOWEVER FOR THE PROGRAM TO WORK THE PATH NEEDS TO BE ABSOLUTE PATH. "~" DOES NOT WORK

"""

path_to_ffmpeg = os.getcwd().split("Comp")[0] + '.imageio/ffmpeg/ffmpeg-linux64-v3.3.1'
os.environ['IMAGEIO_FFMPEG_EXE'] = path_to_ffmpeg

comm	= MPI.COMM_WORLD
size 	= comm.Get_size()
	# rank is zero-indexed
rank 	= comm.Get_rank()


# extract the video names from the paths
def path_to_name(v_path):
	return v_path.split('/')[-1].split('.')[0]

# testing code - getting only a subset of true clusters
def load_filenames(n_clusters=970):
    if n_clusters == 970:
    	# find paths to all videos
        return glob.glob('../videos/*.mp4')
    truth = get_truth()[0:n_clusters]
    filenames = ['../videos/'+name +'.mp4'
    			for set_of_names in truth 
    			for name in set_of_names]
    return filenames

# the main function, which solves the challenge
def main(n_clusters, do_weight, cluster):

	if rank == 0:
		start_time = time.time()
		video_files	= load_filenames(n_clusters = n_clusters)
		video_names = map(path_to_name, video_files)
		chunks	= [[] for _ in range(size)]
		for i, chunk in enumerate(zip(video_files,video_names)):
			chunks[i % size].append(chunk)
	else:
		video_files_and_names 	= None
		chunks 					= None
	comm.Barrier()
	
	# handle hashing in each process
	video_files_and_names 	= comm.scatter(chunks, root = 0)	
	video_files, names		= zip(*video_files_and_names)
	videos					= map(lambda x: generate_video_representation(x,do_weight), video_files)
	
	data = comm.gather(zip(videos,names), root = 0)
	if rank == 0:
		videos, video_names = zip(*[pair for paired_data in data for pair in paired_data ])
		videos 		= np.asarray(videos)
		if cluster == 'kmeans':
			clusters = cluster_videos_kmeans(videos, video_names, n_clusters)
		elif cluster == 'gmm':
			clusters = cluster_videos_gmm(videos, video_names, n_clusters)
		elif cluster == 'ac':
			clusters = cluster_videos_ac(videos, video_names, n_clusters)
		score 		= rand_index(clusters, n_clusters)
	
		print "Scores: ", np.round(score,2), "\nExecution time: %s" % (time.time() - start_time)



if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--n_clusters", type = int, default = 970, choices = range(971), help = "Specify number of clusters to use for testing")
	parser.add_argument("--do_weight", action = 'store_true', help = "Specify whether features should be weighted")
	parser.add_argument("--cluster", type = str, default = "kmeans", choices = ["kmeans", "gmm", "ac"], help = "Specify which cluster to use")
	
	
	args 		= parser.parse_args()
	n_clusters	= args.n_clusters
	do_weight	= args.do_weight
	cluster 	= args.cluster
	if rank == 0:
		print "Used parameters: \n\tn_clusters: {0}\n\tdo_weight: {1}\n\tcluster: {2}".format(n_clusters,do_weight,cluster)
	# run main function
	main(n_clusters, do_weight, cluster)
