
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
import load_data as ld

# look into ImageDataGenerator
#note you hard coded the input size

X, y = ld.get_data('test_comm_images','test_show_images')

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.1,random_state=42)

#reshaping input
X_train = np.array([img for img in X_train])
X_test = np.array([img for img in X_test])


n_train_obs = X_train.shape[0]
n_test_obs = X_test.shape[0]
input_size = X[0].shape
batch_size = 250
nb_classes = 2
nb_epoch = 10


Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


n_filters_1 = 32
n_filters_2 = 12
n_filters_3 = 8
filter_width = 5
filter_height = 5
pool_width = 2
pool_height = 2

model = Sequential()
model.add(Convolution2D(n_filters_1, filter_width, filter_height, border_mode='same',input_shape=input_size, dim_ordering='tf'))
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

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

#don't think you need this
datagen.fit(X_train)

model.compile(loss='categorical_crossentropy', optimizer='adadelta')


model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        samples_per_epoch=n_train_obs,
                        nb_epoch=nb_epoch, show_accuracy=True,
                        validation_data=(X_test, Y_test),
                        nb_worker=1)

# model.fit_generator(ld.generate_arrays_from_file('paths_train.npy'),samples_per_epoch=batch_size, nb_epoch=nb_epoch,verbose=1,show_accuracy=True)
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
