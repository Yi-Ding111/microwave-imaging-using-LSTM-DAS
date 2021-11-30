#author: YI DING
#email: dydifferent@gmail.com
#The University of Queensland 
#Oct 2021

from google.protobuf import reflection
import numpy as np
import skrf as rf
import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
from ast import literal_eval

from tensorflow.python.keras.saving.save import load_model


import antenna_boundary_position as po
import signal_process as sp
import data_prepare as dp
import learning_model as lm

pos_file='./20200309-AS-CNC-Boundary-Movement/LogPeriodic/AntennaPositions.csv'
position_antennas=po.antenna_pos(pos_file)


#prepare dataset as input for LSTM
#signal_col=dp.signal_col_lstm() #the first column of df, reflection signal sequences
#reflection_col=dp.reflection_time_column_lstm() # the second column of df, reflection time from antenna position to boundary
#df_learning={'signal_timesteps':signal_col,'reflection_time':reflection_col}
#dataset=pd.DataFrame(df_learning)

dataset=pd.read_csv('data_for_lm_S11.csv')


signal_timesteps=dataset['signal_timesteps']
reflection_time=dataset['reflection_value']


#the final form of x (time_steps) and y (reflection_time_input) data for LSTM model learning 
data_timesteps, reflection_time_input=lm.csv_preprocess(signal_timesteps,reflection_time)
timesteps_input=lm.input_reshape(data_timesteps)

#prepare the test_input data
snp_file='./20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container1/Data/IF10k-CNC-Exp0123.s16p'
ntwk=rf.Network(snp_file)
real_reflection=sp.signal_complex_to_real(ntwk)#convert complex values to real one
keep_len_signal=sp.keep_len_signal(real_reflection)#keep one part of signals instead of whole as input

#S11
#normalization
keep_len_signal_s22=dp.normalization(keep_len_signal[1])
test_input=np.array(keep_len_signal_s22) 
#test_input=lm.input_reshape(test_input)#the array for testing
#print(test_input[0])


#traing lstm model
model=lm.lstm_learning(timesteps_input,reflection_time_input)
#model.save('lstm_disS11.h5')
#model=load_model('lstm_disS11.h5')
'''
#predict the test input
test_output=[]
for i in range(len(test_input)):
    test=np.reshape(test_input[i],(1,len(test_input[i]),1))#the array for testing
    outcome=lm.lstm_predict(model,test)#prediction
    test_output.append(outcome)
#the final ouput
reflection_output=[]
for i in range(len(test_output)):
    reflection_output.append(((test_output[i])[0])[0])
print(reflection_output)
'''

#S88
test=np.reshape(test_input,(1,len(test_input),1))
print(test.shape)
outcome=lm.lstm_predict(model,test)
print(outcome)


#original distance
ori_dis=dp.normalization_return(outcome[0],'S11')
print(ori_dis[0])


