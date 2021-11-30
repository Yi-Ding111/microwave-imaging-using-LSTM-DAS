#author: YI DING
#email: dydifferent@gmail.com
#The University of Queensland 
#Oct 2021

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
ntwk=rf.Network('/Users/charles/Desktop/Target/HL/CalibratedData/IF10k-CNC-Exp0120-calq.s16p') #65


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
    
    coef_array.append(coef_list)

#tf.beamformer_polar(beamformer_array)


intersection_list=[]
for i in range(len(coef_array)):
    for j in range(i+1,len(coef_array)):
        inter_coor=tf.intersection_cal(coef_array[i],coef_array[j])
        intersection_list.append(inter_coor)






#filter coordinates out of boundary
max_x,min_x,max_y,min_y=po.antenma_pos_limit(pos_file)

intersection_filter=[]
for i in range(len(intersection_list)):
    intersection=intersection_list[i]
    if intersection[0]>min_x and intersection[0]<max_x:
        if intersection[1]>min_y and intersection[1]<max_y:
            if (intersection[0]**2+intersection[1]**2)<po.intersection_limit(pos_file):
                intersection_filter.append(intersection)

intersection_df=pd.DataFrame(np.array(intersection_filter))

#intersection after Kmeans
select_intersection=tf.kmeans_cluster(intersection_df,7) #21
select_df=pd.DataFrame(np.array(select_intersection))




plt.subplot(1,2,1)
plt.scatter(intersection_df[0],intersection_df[1])
plt.scatter(position2['x'],position2['y'])
plt.scatter(select_df[0],select_df[1])
plt.xlabel('EXP0120')
plt.legend()
#plt.show()



#target show
plt.subplot(1,2,2)
average_intersection=tf.average_intersection(select_intersection)
plt.scatter(position2['x'],position2['y'],s=4)
plt.scatter(average_intersection[0],average_intersection[1],s=100.)
plt.xlabel('EXP0120')
plt.legend()
plt.show()





