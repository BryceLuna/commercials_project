
import numpy as np
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras.layers.convolutional import Convolution2D, MaxPooling2D
# from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
import load_data as ld

#loading commercial and tv images
X, y = ld.get_data('test_images1','test_images2')


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=42)


n_train_obs = X_train.shape[0]
n_test_obs = X_test.shape[0]
input_size = X[0].shape[::-1]
batch_size = 250
nb_classes = 2
nb_epoch = 20
image_width = input_size[1]
image_height = input_size[2]


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
model.add(Convolution2D(n_filters_1, filter_width, filter_height, border_mode='same',input_shape=input_size)
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(pool_width, pool_height)))
model.add(Convolution2D(n_filters_2, n_filters_1, filter_width, filter_height, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(pool_width, pool_height)))
model.add(Convolution2D(n_filters_3, n_filters_2, filter_width, filter_height, border_mode='same'))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c',type=str)
    # parser.add_argument('-s',type=str)
    # args = parser.parse_args()
    # tv_path = args.s
    # comm_path = args.c
    pass
