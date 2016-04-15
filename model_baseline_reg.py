import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
import load_data as ld


def import_data(path1,path2):
    X, y = ld.get_data(path1,path2)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.1,random_state=42)

    #reshaping not working - so using this hack
    X_train = np.array([img/255. for img in X_train]).astype('float32')
    X_test = np.array([img/255. for img in X_test]).astype('float32')

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, X_test, Y_train, Y_test

def run_model(X_train,X_test,Y_train,Y_test):

    n_train_obs = X_train.shape[0]
    n_test_obs = X_test.shape[0]
    input_size = X_train[0].shape
    batch_size = 250
    nb_classes = 2
    nb_epoch = 12


    n_filters_1 = 64
    n_filters_2 = 32
    n_filters_3 = 32
    filter_width = 3
    filter_height = 3
    pool_width = 2
    pool_height = 2

    model = Sequential()

    model.add(Convolution2D(n_filters_1, filter_width, filter_height, border_mode='valid',input_shape=input_size, dim_ordering='tf'))
    model.add(LeakyReLU(alpha=0.01))

    model.add(Convolution2D(n_filters_2, filter_width, filter_height, border_mode='valid'))
    model.add(LeakyReLU(alpha=0.01))


    model.add(MaxPooling2D(pool_size=(pool_width, pool_height)))

    model.add(Convolution2D(n_filters_3,filter_width, filter_height, border_mode='valid'))
    model.add(LeakyReLU(alpha=0.01))

    model.add(MaxPooling2D(pool_size=(pool_width, pool_height)))

   #model.add(Dropout(.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(.3))

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

    es = EarlyStopping(patience=3)

    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            samples_per_epoch=n_train_obs,
                            nb_epoch=nb_epoch, show_accuracy=True,
                            validation_data=(X_test, Y_test),
                            nb_worker=1,callbacks=[es])

    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    return model

def save_model(model):
    json_string = model.to_json()
    with open('model_files/baseline_reg.json','w') as f:
        f.write(json_string)
    model.save_weights('model_files/baseline_reg.h5')


def score_model(model,X_test,y_test):
    y_pred = list(model.predict_classes(X_test))
    y_test = list(y_test)
    print classification_report(y_test,y_pred)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test= import_data('commercial_images_rescaled','show_images_rescaled')
    model = run_model(X_train, X_test, y_train, y_test)
    save_model(model)
    score_model(model,X_test,y_test)
