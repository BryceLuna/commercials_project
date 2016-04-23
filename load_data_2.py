import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from pyimage.pipeline import ImagePipeline


class Load_Data(ImagePipeline):
    """docstring for """
    def __init__(self,parent_dir):
        super(Load_Data,self).__init__(parent_dir)

        self.file_lst2 = []


    def _empty_variables(self):
        """
        """
        self.file_lst2 = []
        self.file_names2 = []
        self.features = None
        self.labels = None

    def get_frames(file_str):
        '''
        string => None
        This function takes in the source of a video, samples from
        the video and writes those samples to a folder
        '''
        vid = cv2.VideoCapture(file_str)

        if vid.isOpened():
            frame_count = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            step_size = int(1/float(pct_frames))

            for count in xrange(0,frame_count,step_size):
                w_path = write_path(file_str,count)
                vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,count)
                ret, frame = vid.read()
                count+=step_size
                return frame
            vid.release()
        else:
            print 'unable to open file: {}'.format(file_str)



    def read(self,file_type ,sub_dirs=tuple('all')):
        """

        """
        # Empty the variables containing the image arrays and image names, features and labels
        self._empty_variables()

        # Assign the sub dir names based on what is passed in
        self._assign_sub_dirs(sub_dirs=sub_dirs)
        if file_type in ['video','images']: #should this be a try accept?
            for sub_dir in self.sub_dirs:
                file_names = filter(self._accpeted_file_format, os.listdir(sub_dir))
                self.file_names2.append(file_names)

                file_lst = [self.get_frames(os.path.join(sub_dir, fname)) \
                            if file_type == 'video' else \
                            cv2.imread(os.path.join(sub_dir, fname)) \
                            for fname in file_names]
                self.file_lst2.append(file_lst)
        else:
            print "pleae enter valid file type"

    def transform(self, func, params, sub_dir=None, img_ind=None):
        """
        Takes a function and apply to every img_arr in self.img_arr.
        Have to option to transform one as  a test case

        :param sub_dir: The index for the image
        :param img_ind: The index of the category of images
        """
        # Apply to one test case
        if sub_dir is not None and img_ind is not None:
            sub_dir_ind = self.label_map[sub_dir]
            img_arr = self.file_lst2[sub_dir_ind][img_ind]
            img_arr = func(img_arr, **params).astype(float)
            cv2.imshow(img_arr)
            plt.show()
        # Apply the function and parameters to all the images
        else:
            new_img_lst2 = []
            for img_lst in self.img_lst2:
                new_img_lst2.append([func(img_arr, **params).astype(float) for img_arr in img_lst])
            self.img_lst2 = new_img_lst2

    def get_data(self,):
        if not self.file_lst2 and type(self.file_lst2[0][0]).__module__ == np.__name__:
            self._vectorize_labels()
            for pair in zip(self.file_lst2,self.labels):
                #TODO



    def _vectorize_labels(self):
        """
        Convert file names to a list of y labels (in the example it would be either cat or dog, 1 or 0)
        """
        # Get the labels with the dimensions of the number of image files
        self.labels = np.concatenate([np.repeat(i, len(img_names))
                                      for i, img_names in enumerate(self.file_names2)])


    @staticmethod
    def _accpeted_file_format(fname):
        """
        Return boolean of whether the file is of the accepted file format

        :param fname: Name of the file in question
        :return: True or False (if the file is accpeted or not)
        """
        formats = ['.png', '.jpg', '.jpeg','.mp4']
        for fmt in formats:
            if fname.endswith(fmt):
                return True
        return False



if __name__ == '__main__':
    data = Load_Data('images')
