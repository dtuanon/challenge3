

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# DESCRIPTION:  The script below is for playing with one video (only for testing). 
#               Later it has to be adapted to process multiple videos.

# TASK:   Out of 1 thousand videos, another 9 thousand have been created which
#         are the same as the originals but with alterations in brightness (I 
#         have not yet seen color changes), speed, and some have also margins
#         added to the frame. Our task is to be able to recognize which videos
#         belong together. 

# IDEA:      The idea is to take a single video, hash or analyze every frame of that video,
#            such that each video is a collection of features/hash ids.

# If a video is a subclip of another, or simply a slower version of another,
# then the frames found in one video will also be found in the other videos. 
# This can allow us to detect which videos are the same but with slight alterations
# in speed, color or length. 



# Specify video file name here 
cap = cv2.VideoCapture('video_pairs/spoon1.mp4')

# Creating list to collect hash ids 
hash_collection = []
objects_collection = []

frame_count = 1
while(cap.isOpened()):
    # Reading frame by frame
    ret, frame = cap.read(); 
    
    # Displaying and Analyzing only every 30th frame
    if frame_count % 30 == 0:
        cv2.imshow("Original", frame)
        cv2.waitKey(0)
        
        #___ Perform the frame operations here__#
        
        # 1. Converting to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Gray Scale',gray)
        cv2.waitKey(0)

        # 2. Adding gaussian blur too aid edge detection
        blurred = cv2.GaussianBlur(gray, (7,7), 0)
        cv2.imshow('Gaussian Blur',blurred)
        cv2.waitKey(0)

        # 3. Finding Edges
        # (NOTE: Experiment with different edge detection thresholds. Crossvalidate?)
        canny = cv2.Canny(blurred, 10, 30) # 10, 30 will detect a lot! Try 50, 150
        # Uncomment line below to display the detected edges
        cv2.imshow("Detected Edges", canny)
        
        # 4. EXTRA: Based on the edges, you could additionally also detect the 
        #           number of objects in the frame as an extra feature before
        #           hashing the frame. 
        image, contours, hierarchy= cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects_collection.append(len(contours)) 
        print("Number of objects found = ", len(contours), " in frame ", frame_count)

        
        # Uncomment lines below if you want to display object/contour detection
        cv2.drawContours(frame, contours, -1, (0,255,0), 2)
        cv2.imshow("objects Found", frame)
        cv2.waitKey(0)
        
        # 5. [ ] Process canny image like in exercise 6?0
        
        # 6. [ ] Based on our features (num_objects and pixels from canny image) we
        #    hash (using LSH) each frame of the video.
        #hash_id = hash_frame(canny, num_objects)
        #hash_collection.append(hash_id)
    
    frame_count += 1
    
cap.release()
cv2.destroyAllWindows()
