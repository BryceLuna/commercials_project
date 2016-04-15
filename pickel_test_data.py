import numpy as np
import load_data as ld
import cPickle as pickle
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils


def save_test_data(path1,path2):
    X, y = ld.get_data(path1,path2)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.1,random_state=42)

    X_test = np.array([img/255. for img in X_test]).astype('float32')

    Y_test = np_utils.to_categorical(y_test, 2)

    test_data = {'X_test':X_test,'Y_test':Y_test}
    pickle.dump(test_data,open('test_data.p','wb'))


if __name__ == '__main__':
    save_test_data('commercial_images_rescaled','show_images_rescaled')
