import numpy as np
import skrf as rf
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft,fftfreq
import pandas as pd
from pylab import *
#from sklearn.decomposition import PCA
from math import tan,pi

import antenna_boundary_position as po
import target_data_pre as tp
import signal_process as sp
import learning_model as lm
from ast import literal_eval
from sklearn.preprocessing import MinMaxScaler
pos_file='./20200309-AS-CNC-Boundary-Movement/LogPeriodic/AntennaPositions.csv'
position=po.antenna_pos(pos_file)
a=position.iloc[0]
#print(a['degree'])


bound_file='./20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container1/GroundTruth/Exp0000.csv'
position2=po.boundary_pos(bound_file)
#print(sr.bound_ante_point(pos_file,bound_file))

closet_coordinate=pd.DataFrame(po.bound_ante_point(pos_file,bound_file))
#print(closet_coordinate)


#plt.scatter(position['x'],position['y'])
#plt.scatter(position2['x'],position2['y'],s=4.)
#plt.scatter(closet_coordinate[0],closet_coordinate[1],s=4.)
#plt.show()

#a=po.antenna_to_boundary_time(pos_file,bound_file)
#print(a)



plt.subplot(3,1,1)
ntwk=rf.Network('/Users/charles/Desktop/Target/HM/CalibratedData/IF10k-CNC-Exp0001-calq.s16p')
t=sp.signal_time(ntwk)
#a=sp.signal_complex_to_real(ntwk)
a=sp.scattering_td_signal(ntwk)
#a=tp.diff_method(ntwk)
plt.plot(t,a['td1,3'],label='t1,3,exp0001_signal')
plt.xlabel('(a)')
#plt.plot(t,a[1],label='t2,4,afterdiff_exp0001')
#plt.plot(t,a[2],label='t3,5,afterdiff_exp0001')
#plt.plot(t,a[3],label='t4,6,afterdiff_exp0001')
#plt.plot(t,a[4],label='t5,7,afterdiff_exp0001')
#plt.plot(t,a[5],label='t6,8,afterdiff_exp0001')
plt.legend()



plt.subplot(3,1,2)
ntwk=rf.Network('/Users/charles/Desktop/Target/HM/CalibratedData/empty-calq.s16p')
t=sp.signal_time(ntwk)
#a=sp.signal_complex_to_real(ntwk)
a=sp.scattering_td_signal(ntwk)
#a=tp.diff_method(ntwk)
plt.plot(t,a['td1,3'],label='t1,3,empty_signal')
plt.xlabel('(b)')
#plt.plot(t,a[1],label='t2,4,afterdiff_exp0001')
#plt.plot(t,a[2],label='t3,5,afterdiff_exp0001')
#plt.plot(t,a[3],label='t4,6,afterdiff_exp0001')
#plt.plot(t,a[4],label='t5,7,afterdiff_exp0001')
#plt.plot(t,a[5],label='t6,8,afterdiff_exp0001')
plt.legend()



plt.subplot(3,1,3)
ntwk=rf.Network('/Users/charles/Desktop/Target/HM/CalibratedData/IF10k-CNC-Exp0001-calq.s16p')
t=sp.signal_time(ntwk)
#a=sp.signal_complex_to_real(ntwk)
#a=sp.scattering_td_signal(ntwk)
a=tp.diff_method(ntwk)
plt.plot(t,a[0],label='t1,3,afterdiff_exp0001')
plt.xlabel('(c)')
#plt.plot(t,a[1],label='t2,4,afterdiff_exp0001')
#plt.plot(t,a[2],label='t3,5,afterdiff_exp0001')
#plt.plot(t,a[3],label='t4,6,afterdiff_exp0001')
#plt.plot(t,a[4],label='t5,7,afterdiff_exp0001')
#plt.plot(t,a[5],label='t6,8,afterdiff_exp0001')
plt.legend()



plt.show()


real_reflection_signal=sp.signal_complex_to_real(ntwk)
quarter_signal=sp.keep_len_signal(real_reflection_signal)
a=np.array(quarter_signal)
#print(a)



'''

dataset=pd.read_csv('data_for_lm.csv')

signal_timesteps=dataset['signal_timesteps']
reflection_time=dataset['reflection_time']
signal_timesteps=signal_timesteps.to_list()
reflection_time=reflection_time.to_list()

input=[]
input1=literal_eval(signal_timesteps[0]) #transfer string to list 
input2=literal_eval(signal_timesteps[1])
input.append(input1)
input.append(input2)
a=lm.input_reshape(np.array(input))
print(a[0].shape)


ntwk=rf.Network('./20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container1/Data/IF10k-CNC-Exp0030.s16p')
plt.subplot(2,1,1)
freq=ntwk.s[:,0,0]
t=sp.signal_time(ntwk)
plt.plot(np.arange(751),freq,label='fd:s11')
plt.xlabel('(a)')
plt.legend()
plt.subplot(2,1,2)
a=sp.signal_complex_to_real(ntwk)
plt.plot(np.arange(2001),a[0],label='td:s11')
plt.xlabel('(b)')
plt.legend()
plt.show()
'''