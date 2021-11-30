#author: YI DING
#email: dydifferent@gmail.com
#The University of Queensland 
#Oct 2021

from datetime import time
from threading import active_count
import tensorflow as tf
import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.python.keras import activations
from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.recurrent_v2 import lstm_with_backend_selection
import keras.backend as K


from ast import literal_eval
import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt


#==============================================LSTM many-to-one=============================================
#==============================================LSTM many-to-one=============================================





#csv file preprocess the csv file should include two columns.
#the first column is for timesteps list
#the second column is for relfection time
#reference in data_prepare.py

def csv_preprocess(signal_timesteps,reflection_time):
    '''
    return the converted input form to satisfy the learning_model module ask

    Parameters:
        signal_timesteps: the first column of csv
        reflection_time: the second column of csv
    '''
    signal_timesteps=signal_timesteps.to_list()
    reflection_time=reflection_time.to_list()

    data_timesteps=[]
    for index in range(len(signal_timesteps)):
        input=literal_eval(signal_timesteps[index])
        data_timesteps.append(input)

    data_timesteps=np.array(data_timesteps) # the input shound in form of array, but not list
    reflection_time_input=np.array(reflection_time)

    return data_timesteps, reflection_time_input






#reshape the input

def input_reshape(data_timesteps):
    '''
    reshape it into number of samples, time-steps and features

    parameters:
        data-timesteps: an array of numbers of timesteps data
    '''
    timesteps_reshape=np.reshape(data_timesteps,(data_timesteps.shape[0],data_timesteps.shape[1],1))

    return timesteps_reshape






def linear_regression_equality(y_true, y_pred):
    '''
    the method to return accuracy in LSTM

    parameters:
        y_ture:the ground truth
        y_test: the prediction
    '''
    diff = K.abs(y_true-y_pred)

    return K.mean(K.cast(diff < 0.1, tf.float32))






def lstm_learning(x,y):
    '''
    return the LSTM traing model

    parameters:
        x: the timesteps data
        y: the corresponding reflection time of each timesteps
    
    model = Sequential()
    #model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(x.shape[1:])))
    #model.add(Dropout(0.1))
    model.add(LSTM(10, activation='relu', return_sequences=True))
    model.add(Dropout(0.1))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1))

    opt = optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss='mse',metrics='acc')

    model.fit(x, y, epochs=10,validation_split=0.2, verbose=1)
    '''
    model = Sequential()
    #model.add(LSTM(128, activation='relu',return_sequences=True, input_shape=x.shape[1:]))
    #model.add(Dropout(0.2))
    #model.add(Dense(50, activation='relu'))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    opt = tf.keras.optimizers.Adam(lr=1e-3)#,decay=1e-6)
    model.compile(optimizer=opt, loss='mse')
    model.fit(x, y, epochs=10,batch_size=50,validation_split=0.2, verbose=1)
    return model






def lstm_predict(model,test_input):
    '''
    predict the output value of test input

    parameters:
        model: lstm trained model
        test_input: the timestamps data for test
    '''
    output = model.predict(test_input, verbose=0)
    
    return output










if __name__=='__main__':
    pass