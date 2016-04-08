
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import load_data_generator as ld

# look into ImageDataGenerator
#note you hard coded the input size

# n_train_obs = X_train.shape[0]
# n_test_obs = X_test.shape[0]
# input_size = X[0].shape
batch_size = 250
nb_classes = 2
nb_epoch = 10


# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)


n_filters_1 = 128
n_filters_2 = 64
n_filters_3 = 8
filter_width = 5
filter_height = 5
pool_width = 2
pool_height = 2

model = Sequential()
model.add(Convolution2D(n_filters_1, filter_width, filter_height, border_mode='same',input_shape=(3,100,100), dim_ordering='tf'))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(pool_width, pool_height)))
model.add(Convolution2D(n_filters_2, filter_width, filter_height, border_mode='same'))
model.add(Activation('relu'))
#model.add(Dropout(.25))
model.add(MaxPooling2D(pool_size=(pool_width, pool_height)))
model.add(Convolution2D(n_filters_3,filter_width, filter_height, border_mode='same'))
model.add(Activation('relu'))
#model.add(Dropout(0.25))

#might want to add another dense layer
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(2))
model.add(Activation('softmax'))


#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

model.fit_generator(ld.generate_arrays_from_file('paths_train.npy'),samples_per_epoch=batch_size, nb_epoch=nb_epoch,verbose=1,show_accuracy=True)
#,validation_data=ld.generate_arrays_from_file('paths_test.npy')

# model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1,validation_data=(X_test, Y_test))

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
