'''
Notes:
Probably want to parallelize this code
refer to the open cv example files
you also Probably want a parameter for the size of image
and the resolution although you could downsample after

Bayesian update
markov chain state change?
'''
from sys import argv
import cv2
import numpy as np
import multiprocessing
from os import listdir
from os.path import join
import pdb

script,source,destination,POOL_SIZE, pct_frames = argv
POOL_SIZE = int(POOL_SIZE)
pct_frames = float(pct_frames)
file_lst = [ join(source,f) for f in listdir(source) ] #there should only be file, not folders

def get_frame_parallel(pool_size):
    pool = multiprocessing.Pool(pool_size)
    pool.map(get_frames,file_lst)


def write_path(file_str,count):
    name = file_str.split('/')[-1].split('.')[0]
    pdb.set_trace()
    return join(destination, name + '_' + str(count) + '.png' )


#should you assert pcnt_frames is a float <1
def get_frames(file_str):
    vid = cv2.VideoCapture(file_str)

    if vid.isOpened():
        frame_count = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        step_size = int(1/float(pct_frames))

        count = 0
        while count <= frame_count:
            pdb.set_trace()
            w_path = write_path(file_str,count)
            vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,count)
            ret, frame = vid.read()
            cv2.imwrite(w_path,frame)
            count+=step_size
        #Do you need to close the VideoCapture?
    else:
        return 'could not open file'



if __name__ == '__main__':
    get_frames('data/Tide.mp4')
