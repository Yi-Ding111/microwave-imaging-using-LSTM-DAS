#author: YI DING
#email: dydifferent@gmail.com
#The University of Queensland 
#Oct 2021

from math import pi, trunc
from matplotlib.pyplot import pcolor
import numpy as np
from numpy.lib.function_base import diff
import pandas as pd
import skrf as rf
import signal_process_2 as sp
import math
from math import pi
import matplotlib.pyplot as plt


def td_signal_target(ntwk, ntwk1):
    #scatter signal dictionary
    scatter_dict=sp.scattering_td_signal(ntwk)
    scatter_value_list=list(scatter_dict.values())
    scatter_value_array=np.array(scatter_value_list)

    scatter_dict_1=sp.scattering_td_signal(ntwk1)
    scatter_value_list_1=list(scatter_dict_1.values())
    scatter_value_array_1=np.array(scatter_value_list_1)

    #create zero matrix
    scatter_array_shape=scatter_value_array.shape
    difference_signal=np.zeros(scatter_array_shape)

    print(scatter_value_array.shape[0])
    #Take difference and form 180 deg out-of-phase two element array

    for i in range(scatter_value_array.shape[0]):
        difference_signal[i]=scatter_value_array[i]-scatter_value_array_1[i]
   #     if i != scatter_value_array.shape[0]-1:
#            difference_signal[i]=scatter_value_array[i]-scatter_value_array[i+1]
 #       else:
  #          difference_signal[i]=scatter_value_array[i]-scatter_value_array[0]

    #Generate SubSignals
    sub_pos=np.zeros(scatter_array_shape)
    sub_neg=np.zeros(scatter_array_shape)
    mean_zero=np.mean(difference_signal)
    print(mean_zero)

    for i in range(scatter_value_array.shape[0]):
        count=0
        index_list=[]
        for j in range(len(scatter_value_array[i])):
            if difference_signal[i][j]>mean_zero:
            #if difference_signal[i][j]>0:
                count +=1
                index_list.append(j)
        for replace_index in range(0,count):
            sub_pos[i][index_list[replace_index]]=difference_signal[i][index_list[replace_index]]

        count_2=0
        index_list_2=[]
        for m in range(len(scatter_value_array[i])):
            if difference_signal[i][m]<mean_zero:
            #if difference_signal[i][m]<0:
                count_2 +=1
                index_list_2.append(m)
        for replace_index_2 in range(0,count_2):
            sub_neg[i][replace_index_2] = -1 * difference_signal[i][index_list_2[replace_index_2]]

    
    #normalization
    for i in range(scatter_value_array.shape[0]):
        sub_pos[i]=sub_pos[i]/np.max(sub_pos[i])
        sub_neg[i]=sub_neg[i]/np.max(sub_neg[i])


    truncatedlen=300
    image_sub_pos=np.zeros((scatter_value_array.shape[0],truncatedlen))
    image_sub_neg=np.zeros((scatter_value_array.shape[0],truncatedlen))
    for i in range(scatter_value_array.shape[0]):
        image_sub_pos[i]=sub_pos[i][:truncatedlen]
        image_sub_neg[i]=sub_neg[i][:truncatedlen]


    a=np.arange(0,truncatedlen)
    a_transpose=a.reshape(a.shape[0],1)
    r=a_transpose/truncatedlen
    theta=np.linspace(-pi,pi,scatter_value_array.shape[0])
    theta_list_x, theta_list_y=[],[]
    for i in theta:
        theta_cos=math.cos(i)
        theta_list_x.append(theta_cos)
    for j in theta:
        theta_sin=math.sin(j)
        theta_list_y.append(theta_sin)

    X=r*theta_list_x
    Y=r*theta_list_y

    
    newmap=np.zeros((scatter_value_array.shape[0],truncatedlen))
    indexlist_x=[]
    indexlist_y=[]
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if Y[i][j]>=0:
                indexlist_x.append(i)
                indexlist_y.append(j)
    for index in range(len(indexlist_x)):
        newmap[indexlist_y[index]][indexlist_x[index]]=image_sub_pos[indexlist_y[index]][indexlist_x[index]]
    
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if Y[i][j]<0:
                indexlist_x.append(i)
                indexlist_y.append(j)
    for index in range(len(indexlist_x)):
        newmap[indexlist_y[index]][indexlist_x[index]]=image_sub_neg[indexlist_y[index]][indexlist_x[index]]

    #transpose
    newmap=newmap.T
    
    fig, ax = plt.subplots()
    c = ax.pcolormesh(X,Y,newmap)
    fig.colorbar(c, ax=ax)
    plt.show()






    
    


    
    










    return 

























if __name__=='__main__':
    ntwk=rf.Network('/Users/charles/Desktop/Target/HM/CalibratedData/IF10k-CNC-Exp0078-calq.s16p')
    ntwk1=rf.Network('/Users/charles/Desktop/Target/HM/CalibratedData/IF10k-CNC-Exp0000-calq.s16p')
    print(td_signal_target(ntwk, ntwk1))