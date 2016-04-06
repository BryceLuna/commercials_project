'''
To shrink an image, it will generally look best with CV_INTER_AREA interpolation
'''

import cv2
import argparse
#from sys import argv
import numpy as np
from os import listdir
from os.path import join
#import pdb

def write_path(file_str):
    '''
    string,tuple => string
    This function takes in an image file path and returns
    the path in which to write the image
    '''
    name = file_str.split('/')[-1].split('.')[0]
    #pdb.set_trace()
    return join(destination, name + '_' + dim_str + '.jpeg' )


def resize_img(file_str):
    '''
    string => None

    '''
    w_path = write_path(file_str)
    image = cv2.imread(file_str)
    #pdb.set_trace()
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(w_path,resized)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',type=str)
    parser.add_argument('-d',type=str)
    parser.add_argument('-dim',type=int,nargs=2)
    args = parser.parse_args()
    source = args.s
    destination = args.d
    dim = tuple(args.dim)

    dim_str = str(dim[0])+'x'+str(dim[1])
    file_lst = [ join(source,f) for f in listdir(source) ] #generator instead?

    for file_ in file_lst:
        resize_img(file_)
