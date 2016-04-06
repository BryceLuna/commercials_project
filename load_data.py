'''
might want to pickle the loaded data
'''
import argparse
import cv2
import numpy as np
from os import listdir
from os.path import join
import cPickle as pickle
#import pdb


def stack_images(image_lst,label):
    images = []          #might want to pre-allocate
    for filename in image_lst:
        img = cv2.imread(filename)
        #pdb.set_trace()
        if img is not None:
            images.append([img,label])
    return images


def get_data(commercials_path,shows_path):
    shows_lst = [join(shows_path,f) for f in listdir(shows_path)]
    commercials_lst = [ join(commercials_path,f) for f in listdir(commercials_path)]
    commercial_array = stack_images(commercials_lst,0)
    show_array = stack_images(shows_lst,1)
    mat = np.concatenate((commercial_array,show_array),axis=0)
    np.random.shuffle(mat)
    pickle.dump([mat[:,0],mat[:,1]],open("save.p","wb"))
    return mat[:,0],mat[:,1]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',type=str)
    parser.add_argument('-s',type=str)
    args = parser.parse_args()
    shows_path = args.s
    commercials_path = args.c
    get_data(commercials_path,shows_path)
