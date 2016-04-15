import numpy as np
import argparse
import json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import classification_report
import load_data as ld
from keras.models import model_from_json
import cPickle as pickle



def load_model(model_path,weights_path):
    with open(model_path) as f:
        mod = f.read()
        model = model_from_json(mod)
        model.load_weights(weights_path)
    return model

def load_test_data(path):
    data = pickle.load(open(path,'rb'))
    return data['X_test'],data['Y_test']

def score_model(model,X_test,y_test):
    y_pred = list(model.predict_classes(X_test))
    y_test = list(y_test)
    print classification_report(y_test,y_pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',type=str)
    parser.add_argument('-w',type=str)
    args = parser.parse_args()
    model_path = args.m
    weights_path = args.w
    model = load_model(model_path,weights_path)
    X_test,Y_test = load_test_data('test_data.p')
    score_model(model,X_test,Y_test[:,1])
