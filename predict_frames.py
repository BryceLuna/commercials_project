import cv2
import numpy as np
from keras.models import model_from_json
from keras.models import Sequential

class ProcessFrames(object):
    """docstring for """
    def __init__(self):
        self.video_path = None
        self.model = None
        self.predictions = None

    def attach_video(self,video_path):
        self.video_path = video_path
    
    def attach_model(self,model_path,weights_path):
         self.model = self.load_model(model_path,weights_path)
         
         
    def load_model(self,model_path,weights_path):
        with open(model_path) as f:
            mod = f.read()
            model = model_from_json(mod)
            model.load_weights(weights_path)
            return model

    def get_predictions(self):
        return self.model.predict_classes(self.predictions)

    def read_video(self):

        vid = cv2.VideoCapture(self.video_path)
        
        if vid.isOpened():
        
            frame_count = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            self.predictions = np.zeros((frame_count,100,100,3))#need to know frame size
            for count in xrange(frame_count):
                ret,frame = vid.read() #probably don't want to get every frame
                processed_frame = self.process_frame(frame)
                self.predictions[count] = processed_frame
            vid.release()
        else:
            print 'unable to open file: {}'.format(file_str)


    #maybe should separate this algo, or somehow automatically detect what the model accepts
    #should probably convert to float32, divide by 255.
    def process_frame(self,frame):
        return cv2.resize(frame,(100,100),interpolation=cv2.INTER_AREA)/255.
