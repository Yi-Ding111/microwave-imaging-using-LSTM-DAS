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
import sympy
import beamformer as bf
import math
from math import pi,tan
import matplotlib.pyplot as plt
import heapq
from sklearn.cluster import KMeans


import antenna_boundary_position as po
import signal_process as sp
import target_data_pre as tp

'''
def td_signal_target(ntwk):
    #scatter signal dictionary
    #scatter_dict=sp.signal_fd_real(ntwk)
    scatter_dict=sp.scattering_td_signal(ntwk)
    scatter_signal_keys={'td1,4','td2,5','td3,6','td4,7','td5,8','td6,9','td7,10','td8,11',
                       'td9,12','td10,13','td11,14','td12,15','td13,16','td14,1','td15,2','td16,3'}
    #the dict of all reflection signals in time domain
    scatter_signal_dict={ key:value for key,value in scatter_dict.items() if key in scatter_signal_keys} 
    scatter_value_list=list(scatter_signal_dict.values())
    #scatter_value_list=list(scatter_dict.values())
    scatter_value_array=np.array(scatter_value_list)
    #scatter_value_array=np.array(sp.signal_complex_to_real(ntwk))


    #create zero matrix
    scatter_array_shape=scatter_value_array.shape
    difference_signal=np.zeros(scatter_array_shape)

    #Take difference and form 180 deg out-of-phase two element array

    for i in range(scatter_value_array.shape[0]):
        if i != scatter_value_array.shape[0]-1:
            difference_signal[i]=scatter_value_array[i]-scatter_value_array[i+1]
        else:
            difference_signal[i]=scatter_value_array[i]-scatter_value_array[0]

    #Generate SubSignals
    sub_pos=np.zeros(scatter_array_shape)
    sub_neg=np.zeros(scatter_array_shape)

    for i in range(scatter_value_array.shape[0]):
        count=0
        index_list=[]
        for j in range(len(scatter_value_array[i])):
            if scatter_value_array[i][j]>0:
                count +=1
                index_list.append(j)
        for replace_index in range(0,count):
            sub_pos[i][index_list[replace_index]]=difference_signal[i][index_list[replace_index]]

        count_2=0
        index_list_2=[]
        for m in range(len(scatter_value_array[i])):
            if scatter_value_array[i][m]<0:
                count_2 +=1
                index_list_2.append(m)
        for replace_index_2 in range(0,count_2):
            sub_neg[i][replace_index_2] = -1 * difference_signal[i][index_list_2[replace_index_2]]

    
    #normalization
    for i in range(scatter_value_array.shape[0]):
        sub_pos[i]=sub_pos[i]/np.max(sub_pos[i])
        sub_neg[i]=sub_neg[i]/np.max(sub_neg[i])


    truncatedlen=100
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
    #print(newmap[1])
    fig, ax = plt.subplots()
    c = ax.pcolormesh(X,Y,newmap)
    fig.colorbar(c, ax=ax)
    plt.show()
    #plt.imshow(newmap)
    #plt.show()

    return 

    #ntwk=rf.Network('/Users/charles/Desktop/microwavestrokesimulatedata/ich/Ich-12635-01-0003.s16p')
    #td_signal_target(ntwk)
    #ntwk=rf.Network('/Users/charles/Desktop/microwavestrokesimulatedata/ich/Ich-12635-01-0013.s16p')
    #td_signal_target(ntwk)
'''





#============================================polar plot=============================================
#============================================polar plot=============================================






def beamformer_polar(beamformer_array):
    '''
    '''
    y_beamf_polar = np.sum(beamformer_array**2, axis=1)
    #print(y_beamf_polar)
    y_beamf_polar_dB = 10*np.log10(y_beamf_polar)
    #print(y_beamf_polar_dB)
    dB_max = y_beamf_polar_dB.max()


    # Polar plot dynamic range
    dyn_range = 5#dB

    N_theta = 181
    theta = np.linspace(0, np.pi, N_theta)

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(theta, y_beamf_polar_dB)

    ax.set_rmax(dB_max)
    ax.set_rmin(dB_max - dyn_range)

    ax.set_thetamin(0)
    ax.set_thetamax(180)
    fig.set_tight_layout(True)
    plt.show()






#============================================intersections============================================
#============================================intersections============================================






def beamformer_degree(beamformer_array):
    '''
    cal the degree of beamformer array in linear array

    parameters:
        beamformer_array: the array of beamformer
    '''
    y_beamf_polar = (np.sum(beamformer_array**2, axis=1)).tolist()
    #y_beamf_polar=(10*np.log10(y_beamf_polar)).tolist()

    N_theta = 181
    theta = np.linspace(0, 180, N_theta)

    #find the direction degree with the largest value in list index
    distance_max_index=list(map(y_beamf_polar.index, heapq.nlargest(3,y_beamf_polar))) #list length: 1
    
    #signal_degree=theta[distance_max_index[0]]

    # cal average degree 
    degree=[]
    for i in distance_max_index:
        degree_single=theta[i]
        degree.append(degree_single)
    
    signal_degree=np.sum(degree)/len(distance_max_index)

    return signal_degree







def direction_linear_equation(pos_file,left_index,rerange_index,right_index,beamformer_array):
    '''
    cal the linear equation corresponding to the direction of each antenna
        
    parameters:
        pos_file:antenna position file
        left_index: the left antenna postion of rerange position
        rerange_index: the index of antenna postion which need moving
        right_index: the right antenna position of rerange position
        beamformer_array: the array of beamformer
    '''
    #the intersection 
    intersection=tp.linear_array_rerange(pos_file,left_index,rerange_index,right_index)
    #read antenna postion dataframe
    antenna_position=po.antenna_pos(pos_file=pos_file)
    #cal linear expression coefficients of two-side antenna positions,Ax+By+C=0
    A_s,B_s,C_s=tp.two_point_equation(antenna_df=antenna_position,index_1=left_index,index_2=right_index)

    #cal two-side tilt angle
    if A_s==0:
        angle=0 #degree
    elif B_s==0:
        angle=90
    else:
        slope=-(A_s/B_s) #y=-(A/B)X-(C/B)
        angle=math.atan(slope)/(pi/180) #range(-90,0) union(0,90)

    beam_angle=beamformer_degree(beamformer_array)


    #new linear tilt angle 
    if angle<=0:
        new_degree=360-(-angle+beam_angle)
    else:
        if angle>=beam_angle:
            new_degree=angle-beam_angle
        else:
            new_degree=360-(beam_angle-angle)
    

    if new_degree==90 or new_degree==270:
        A_new=1
        B_new=0
        C_new=0
    else:
        A_new=tan(new_degree*(pi/180))
        B_new=-1
        C_new=intersection[1]-tan(new_degree*(pi/180))*intersection[0] #y=tan(alpha)x+b  C=b

    return [A_new,B_new,C_new]







def intersection_cal(direction_equation_1,direction_equation_2):
    '''
    cal the intersection of both new direction linear equations

    parameters:
        pos_file:antenna position file
        left_index: the left antenna postion of rerange position
        rerange_index: the index of antenna postion which need moving
        right_index: the right antenna position of rerange position
        beamformer_array: the array of beamformer
    '''
    #Ax+By+C=0
    A_1,B_1,C_1=direction_equation_1[0],direction_equation_1[1],direction_equation_1[2]
    A_2,B_2,C_2=direction_equation_2[0],direction_equation_2[1],direction_equation_2[2]

    x=sympy.symbols('x')
    ans=sympy.solve([((A_2/B_2)-(A_1/B_1))*x+((C_2/B_2)-(C_1/B_1))],[x])

    if len(ans)!=0:
        x_ans=ans[x]
        y_ans=(-A_1/B_1)*x_ans-(C_1/B_1)
    
    else:
        x_ans=1000
        y_ans=1000
    
    return [x_ans,y_ans]






#============================================k-means=====================================
#============================================k-means=====================================

def kmeans_cluster(intersection_group,cluster_num):
    '''
    use kmeans ML to find the densest intersection set

    parameters:
        intersection_group: (df) the intersection set
        cluster_num: (int) the number of clusters
    '''
    km=KMeans(n_clusters=cluster_num)

    #predict 
    km_predict=(km.fit_predict(intersection_group[[0,1]])).tolist()#list

    #find the value which most frequent
    frequent_cluster=max(km_predict,key=km_predict.count)

    select_intersection=[]
    for i in range(len(km_predict)):
        if km_predict[i]==frequent_cluster:
            intersection=intersection_group.iloc[i]
            inter_x=intersection[0]
            inter_y=intersection[1]
            select_intersection.append([inter_x,inter_y])
    
    return select_intersection




    

def average_intersection(select_intersection):
    '''
    cal the average value of intersections

    parameters:
        selection_intersection: (list) the intersections after kmeans
    '''
    x_set=[]
    y_set=[]
    for i in range(len(select_intersection)):
        x_set.append(select_intersection[i][0])
        y_set.append(select_intersection[i][1])
    
    x_average=np.mean(x_set)
    y_average=np.mean(y_set)

    return [x_average,y_average]












if __name__=='__main__':
    pass