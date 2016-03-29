'''
Notes:
Probably want to parallelize this code
refer to the open cv example files
you also Probably want a parameter for the size of image
and the resolution although you could downsample after

Bayesian update
markov chain state change?
'''

import cv2
import numpy as np

#should you assert pcnt_frames is a float <1
def get_frames(file_str,pct_frame):
    vid = cv2.VideoCapture(file)
    if vid.isOpened():
        frame_count = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        step_size = int(1/float(pct_frames))
        count = 0
        while count <= frame_count:
            vid.set(cv2.cv.CV_CAP_POS_FRAMES,count)
            ret, frame = vid.read()
            cv2.imwrite('/Users/datascientist/Desktop/images/' +  '_{}'.format(),frame)
        vid.set(cv2.cv.CV_CAP_POS_FRAMES,)
    else:
        return 'could not open file'



'''
vid = cv2.VideoCapture(file_path)
vid.isOpened()
vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
vid.set(cv2.cv.CV_CAP_POS_FRAMES)
cv.imshow('video','frame[1]')

'''

if __name__ == '__main__':
    pass
