
import numpy as np
import cPickle as pickle
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
import load_data as ld

#loading commercial and tv images
#note you have to pass in commercials first and then shows
#X, y = ld.get_data('commercials_rescaled','shows_rescaled')
data = pickle.load(open("save.p",'rb'))
X,y = data[0],data[1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=42)

#reshaping input
X_train = np.array([img for img in X_train])
X_test = np.array([img for img in X_test])


n_train_obs = X_train.shape[0]
n_test_obs = X_test.shape[0]
input_size = X[0].shape
batch_size = 20
nb_classes = 2
nb_epoch = 5
# image_width = input_size[1]
# image_height = input_size[0]


Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


n_filters_1 = 16
n_filters_2 = 12
n_filters_3 = 8
filter_width = 3
filter_height = 3
pool_width = 2
pool_height = 2

model = Sequential()
model.add(Convolution2D(n_filters_1, filter_width, filter_height, border_mode='same',input_shape=input_size, dim_ordering='tf'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_width, pool_height)))
model.add(Convolution2D(n_filters_2, filter_width, filter_height, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_width, pool_height)))
model.add(Convolution2D(n_filters_3,filter_width, filter_height, border_mode='same'))
model.add(Activation('relu'))
model.add(Dropout(0.25))

#might want to add another dense layer
model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(2))
model.add(Activation('softmax'))


#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1,validation_data=(X_test, Y_test))
# score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=1)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c',type=str)
    # parser.add_argument('-s',type=str)
    # args = parser.parse_args()
    # tv_path = args.s
    # comm_path = args.c
    pass
