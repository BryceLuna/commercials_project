'''
take in all the files into an array along with
their labels
shuffle the array
then write each to a line in a
'''
import numpy as np
from os import listdir
from os.path import join


def process_line(path):
    img = cv2.imread(filename)
    if img is not None:
        return img


def generate_arrays_from_file(path):
    while 1:
        for path,label in np.load(path):
            # create numpy arrays of input data
            # and labels, from each line in the file
            x, y = process_line(path),label
            yield x, y

def write_path_label(commercials_path,shows_path,splt):
    shows_lst = [[join(shows_path,f),0] for f in listdir(shows_path)]
    commercials_lst = [[join(commercials_path,f),1] for f in listdir(commercials_path)]
    path_label = np.concatenate((commercials_lst,shows_lst),axis=0)

    indx = int(splt*len(path_label))
    np.random.shuffle(path_label)
    path_label_train = path_label[:indx]
    path_label_test = path_label[indx:]

    np.save('paths_train',path_label_train)
    np.save('paths_test',path_label_test)

    return None


if __name__ == '__main__':
    write_path_label('test_comm_images','test_show_images',.8)
