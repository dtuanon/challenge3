import glob
import os

# extract the video names from the paths
def path_to_name(v_path):
	return v_path.split('/')[-1].split('.')[0]


# the main function, which solves the challenge
def main(video_files):
	video_names = map(path_to_name, video_files)





if __name__ == "__main__":

	# find paths to all videos
	video_files	= glob.glob('./videos/*.mp4')
	
	# run main function
	main(video_files)
