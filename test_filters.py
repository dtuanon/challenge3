from __future__ import division

import imageio
from skimage import feature
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from PIL import Image
import numpy as np
import adjusted_rand_index
from math import ceil
from skimage.measure import block_reduce

f1 = './videos/7QIVELJDAP50.mp4'
f2 = './videos/LQ8YWJ3ZJK21.mp4'
f3 ='./videos/0TIBYZMOJD10.mp4'
files = [f1,f2,f3]
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

def get_average_edge_index(filename):
    video = imageio.get_reader(filename)

    frames = [video.get_data(0),video.get_data(1)]
    frames = np.stack(frames,axis = 0)
    frames_div = 3
    frames = np.mean(frames,axis=3)
    edges = [feature.canny(i,sigma = 4) for i in frames]
    edges = np.stack(edges)
    s = [int(ceil(i/frames_div)) for i in frames.shape]
    x = block_reduce(edges,block_size=(2,s[1], s[2]), func=np.mean)
    print x.shape
    x = x.flatten()
    x = x/np.linalg.norm(x)
    print np.round(x,decimals=3)
    #print np.mean(edges)

n_clusters = 1
for i,f in enumerate(filenames_gen(n_clusters)):
    get_average_edge_index(f)
    if i%9==0:
        print(i)
