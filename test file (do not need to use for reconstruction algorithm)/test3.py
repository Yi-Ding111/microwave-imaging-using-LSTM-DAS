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
import target_data_pre as tp
import beamformer as bf
import target_find as tf

pos_file='/Users/charles/Desktop/Target/AntennaPositions2.csv'
ntwk=rf.Network('/Users/charles/Desktop/Target/HL/CalibratedData/IF10k-CNC-Exp0001-calq.s16p') #65


#rerange inear array
#position_0=tp.linear_array_rerange(pos_file,0,1,2)
position2=po.antenna_pos(pos_file)
#print(po.antenma_pos_limit(pos_file))
#print(position)
#plt.scatter(position_0[0],position_0[1],c='r',s=4.)
#plt.scatter(position2['x'],position2['y'])
#plt.show()



#polar plot

#scatter_array=sp.linear_array_scatter(ntwk)[0:3]
#scatter_array=sp.scatter_array_truncated(ntwk)[5:8]
#scatter_array=tp.diff_method(ntwk)[2:5]
#print(scatter_array)
#y=bf.das_beamformer(pos_file,2,3,4,scatter_array,ntwk)
#print(y)
#tf.beamformer_polar(y)





#get all intersections of direction linear equation


coef_array=[]

#full_scatter_array=sp.linear_array_scatter(ntwk)
#full_scatter_array=sp.scatter_array_truncated(ntwk)
full_scatter_array=tp.diff_method(ntwk)

for scatter_index in range(len(full_scatter_array)): 
    if scatter_index<=13:
        #the scatter_array input (3,2001)
        scatter_array=full_scatter_array[scatter_index:scatter_index+3]
        #beamformer array
        beamformer_array=bf.das_beamformer(pos_file,scatter_index,scatter_index+1,scatter_index+2,scatter_array,ntwk)
        delay_array=bf.das_beamformer2(pos_file,scatter_index,scatter_index+1,scatter_index+2,scatter_array,ntwk)
        #direction linear equation coefficient list
        coef_list=tf.direction_linear_equation(pos_file,scatter_index,scatter_index+1,scatter_index+2,beamformer_array)
    elif scatter_index==14:
        scatter_array=np.array([full_scatter_array[14],full_scatter_array[15],full_scatter_array[0]])
        beamformer_array=bf.das_beamformer(pos_file,scatter_index,scatter_index+1,0,scatter_array,ntwk)
        coef_list=tf.direction_linear_equation(pos_file,scatter_index,scatter_index+1,0,beamformer_array)
    elif scatter_index==15:
        scatter_array=np.array([full_scatter_array[15],full_scatter_array[0],full_scatter_array[1]])
        beamformer_array=bf.das_beamformer(pos_file,scatter_index,0,1,scatter_array,ntwk)
        coef_list=tf.direction_linear_equation(pos_file,scatter_index,0,1,beamformer_array)

    print(delay_array)
    if scatter_index==0:
        plt.subplot(3,1,1)
        plt.plot(np.arange(2001),scatter_array[0],label='td_signal13')
        plt.plot(np.arange(2001),scatter_array[1],label='td_signal24')
        plt.plot(np.arange(2001),scatter_array[2],label='td_signal35')
        plt.xlabel('(a)')
        plt.legend()
        plt.subplot(3,1,2)
        plt.plot(np.arange(2001),delay_array[2],label='td_singal13_delay')
        plt.plot(np.arange(2001),delay_array[1],label='td_singal24_delay')
        plt.plot(np.arange(2001),delay_array[0],label='td_singal35_delay')
        plt.xlabel('(b)')
        plt.legend()
        plt.subplot(3,1,3)
        plt.plot(np.arange(2000),beamformer_array[45],label='beamformer')
        plt.xlabel('(c)')
        plt.legend()
        plt.show()


    


    
 