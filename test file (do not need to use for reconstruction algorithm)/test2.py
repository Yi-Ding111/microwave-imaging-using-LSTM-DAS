import numpy as np
import skrf as rf
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft,fftfreq
import pandas as pd
from pylab import *
#from sklearn.decomposition import PCA
from math import tan,pi

import antenna_boundary_position as po
import signal_process as sp
import learning_model as lm
from ast import literal_eval
from sklearn.preprocessing import MinMaxScaler



pos_file='./20200309-AS-CNC-Boundary-Movement/LogPeriodic/AntennaPositions.csv'
position=po.antenna_pos(pos_file)
a=position.iloc[0]
#print(a['degree'])


bound_file='./20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container1/GroundTruth/Exp0000.csv'
print(po.antenna_to_boundary_time(pos_file,bound_file))
ntwk=rf.Network('/Users/charles/Desktop/boundary_data/20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container1/Data/IF10k-CNC-Exp0030.s16p')
t=sp.signal_time(ntwk)
a=sp.signal_complex_to_real(ntwk)
#a=sp.scattering_td_signal(ntwk)
plt.plot(t,a[0],label='td1,1')
plt.plot(t,a[1],label='t2,2')
#plt.plot(t,a['td3,6'],label='t3,6')
#plt.plot(t,a['td4,7'],label='t4,7')
#plt.plot(t,a['td5,8'],label='t5,8')
#plt.plot(t,a['td6,9'],label='t6,9')
plt.legend()
plt.show()

fd=sp.signal_fd_real(ntwk)
plt.plot(np.arange(751),fd['fd1,1'],label='fd1,1')
plt.legend()
plt.show()














ntwk=rf.Network('/Users/charles/Desktop/microwavestrokesimulatedata/ich/Ich-12635-01-0003.s16p')
t=sp.signal_time(ntwk)
#a=sp.signal_complex_to_real(ntwk)
a=sp.scattering_td_signal(ntwk)
#plt.subplot(2,1,1)
#plt.plot(t,a['td1,4'],label='t1,4')
#plt.plot(t,a['td2,5'],label='t2,5')
#plt.plot(t,a['td3,6'],label='t3,6')
#plt.plot(t,a['td4,7'],label='t4,7')
#plt.plot(t,a['td5,8'],label='t5,8')
#plt.plot(t,a['td6,9'],label='t6,9')
#plt.legend()
#plt.show()


ntwk2=rf.Network('/Users/charles/Desktop/Target/HL/CalibratedData/IF10k-CNC-Exp0133-calq.s16p')
t=sp.signal_time(ntwk)
t=np.arange(0,181,1)
#a=sp.signal_complex_to_real(ntwk)
#b=sp.scattering_td_signal(ntwk2)
b=sp.scatter_array_truncated(ntwk2)
'''
#plt.subplot(2,1,2)
plt.plot(t,b['td1,4'],label='t1,4')
plt.plot(t,b['td2,5'],label='t2,5')
plt.plot(t,b['td3,6'],label='t3,6')
plt.plot(t,a['td4,7'],label='t4,7')
plt.plot(t,a['td5,8'],label='t5,8')
plt.plot(t,a['td6,9'],label='t6,9')
plt.plot(t,a['td7,10'],label='t7,10')
plt.plot(t,a['td8,11'],label='t8,11')
plt.legend()
plt.show()
'''

plt.plot(t,b[0],label='t1,4')
plt.plot(t,b[1],label='t2,5')
plt.plot(t,b[2],label='t3,6')
plt.plot(t,b[3],label='t4,7')
plt.plot(t,b[4],label='t5,8')
plt.plot(t,b[5],label='t6,9')
plt.plot(t,b[6],label='t7,10')
plt.plot(t,b[7],label='t8,11')
plt.legend()
plt.show()





scatter_dict=sp.signal_fd_real(ntwk)
scatter_dict=sp.scattering_td_signal(ntwk)
scatter_value_list=list(scatter_dict.values())
scatter_value_array=np.array(scatter_value_list)


t=sp.signal_time(ntwk)
#a=sp.signal_complex_to_real(ntwk)
#a=sp.scattering_td_signal(ntwk)
plt.plot(t,scatter_value_array[3],label='t1,4')
plt.plot(t,scatter_value_array[4],label='t1,5')
plt.legend()
#plt.show()

#create zero matrix
scatter_array_shape=scatter_value_array.shape
difference_signal=np.zeros(scatter_array_shape)

#Take difference and form 180 deg out-of-phase two element array

for i in range(scatter_value_array.shape[0]):
    if i != scatter_value_array.shape[0]-1:
        difference_signal[i]=scatter_value_array[i]-scatter_value_array[i+1]
    else:
        difference_signal[i]=scatter_value_array[i]-scatter_value_array[0]


t=sp.signal_time(ntwk)

plt.plot(t,difference_signal[3],label='t1,4')
plt.plot(t,difference_signal[4],label='t5,5')
plt.legend()
#plt.show()