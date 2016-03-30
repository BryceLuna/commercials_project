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
#import pdb



def write_path(file_str,count):
    '''
    string,int => string
    This function takes in the video file path and returns
    the destination path a frame from that video will be written to
    '''
    name = file_str.split('/')[-1].split('.')[0]
    #pdb.set_trace()
    return join(destination, name + '_' + str(count) + '.jpeg' )


#should you assert pcnt_frames is a (float < 1)
def get_frames(file_str):
    '''
    string => None
    This function takes in the source of a video, samples from
    the video and writes those samples to a folder
    '''
    vid = cv2.VideoCapture(file_str)

    #Probably want a try except and issue an error if it doesn't work

    if vid.isOpened():
        frame_count = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        step_size = int(1/float(pct_frames))

        for count in xrange(0,frame_count,step_size):
            #pdb.set_trace()
            w_path = write_path(file_str,count)
            vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,count)
            ret, frame = vid.read()
            cv2.imwrite(w_path,frame)
            count+=step_size
        vid.release()
    else:
        return 'error'



if __name__ == '__main__':
    script, source, destination,POOL_SIZE, pct_frames = argv #look into a try except here
    pool_size = int(POOL_SIZE)
    pct_frames = float(pct_frames)
    #there should only be only files in here not folders or anything else
    #also was there was a hidden file in max .DS_ that I had to delete
    file_lst = [ join(source,f) for f in listdir(source) ]

    # for file_ in file_lst:
    #     get_frames(file_)

    pool = multiprocessing.Pool(pool_size)
    pool.map(get_frames,file_lst)
