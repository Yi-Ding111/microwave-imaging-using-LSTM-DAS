#author: YI DING
#email: dydifferent@gmail.com
#The University of Queensland 
#Oct 2021

import numpy as np
import pandas as pd
from pylab import *
from math import tan,pi
import heapq
import sympy
import skrf as rf

import signal_process as sp
import antenna_boundary_position as po





#===================================================equation function===================================================
#===================================================equation function===================================================






def degree_binary_linear_equation(antenna_df,index):
    '''
    according to the degree of antenna transmitting, cal the equation (y=tan alpha x +b)

    parameters:
        antenna_df: (dataframe) antenna position
        index: the index of dataframe from 0 to 15
    
    return: the coefficients of equation: Ax+By+C=0
    '''
    point_record=antenna_df.iloc[index] #select the index row of df

    #Ax+By+C=0
    if point_record['degree']==90 or point_record['degree']==270:
        A,B,C=1,0,0
    elif point_record['degree']==0 or point_record['degree']==180:
        A,B,C=0,1,0
    else:
        x, y, degree=point_record['x'], point_record['y'], point_record['degree']
        A=tan(degree*(pi/180))
        B=-1
        C=y-tan(degree*(pi/180))*x  #y=tan(alpha)x+b  C=b

    return A,B,C # Ax+By+C=0






def two_point_equation(antenna_df,index_1,index_2):
    '''
    according two coordinates, cal linear equation

    parameters:
        antenna_df: (dataframe) antenna position
        index_1: the index of antenna position
        index_2: the other index of antenna position
    
    return: the coefficients of equation: Ax+By+C=0
    '''
    #coordinate (x,y)
    coor_1=antenna_df.iloc[index_1]
    coor_2=antenna_df.iloc[index_2]

    #coefficient
    A=coor_2['y']-coor_1['y']
    B=coor_1['x']-coor_2['x']
    C=(coor_2['x']*coor_1['y'])-(coor_1['x']*coor_2['y'])

    return A,B,C






def intersection_cal(A1,B1,C1,A2,B2,C2):
    '''
    cal the intersection of two linear equations

    parameters:
        A1,B1,C1: the coefficients of one linear equation
        A2,B2,C2: the coefficients of the other linear equation
    
    return: the coordinate (x,y) of intersection
    '''
    m=A1*B2-A2*B1
    if m==0:
        intersection=None
    else:
        x=(C2*B1-C1*B2)/m
        y=(C1*A2-C2*A1)/m
        intersection=(x,y)

    return intersection






#==============================================================rerange position=======================================================
#==============================================================rerange position=======================================================





def linear_array_rerange(pos_file,left_index,rerange_index,right_index):
    '''
    group 16 antennas as 3 a group, like A1,A2,A3; A2,A3,A4....
    rerange 3 positions into one group, move the middle one into same line with other two antennas

    parameters:
        pos_file:antenna position file
        left_index: the left antenna postion of rerange position
        rerange_index: the index of antenna postion which need moving
        right_index: the right antenna position of rerange position
    
    return: the rerange coordinate(x,y) of rerange antenna position
    '''
    #read antenna postion dataframe
    antenna_position=po.antenna_pos(pos_file=pos_file)

    #cal linear expression coefficient of antenna position (x,y,degree), formula is Ax+By+C=0
    A_r,B_r,C_r=degree_binary_linear_equation(antenna_df=antenna_position,index=rerange_index)

    #cal linear expression coefficients of two-side antenna positions,Ax+By+C=0
    A_s,B_s,C_s=two_point_equation(antenna_df=antenna_position,index_1=left_index,index_2=right_index)

    #get the intersection of two equations
    intersection=intersection_cal(A_r,B_r,C_r,A_s,B_s,C_s)

    return intersection






def rerange_distance(pos_file,intersection,rerange_index):
    '''
    cal the distance(mm) between the rerange position and initial position

    parameters:
        pos_file: antenna position file
        intersection: the output of function linear_array_rerange(pos_file,left_index,rerange_index,right_index)
        rerange_index: the index of antenna postion which need moving
    '''
    init_record=po.antenna_pos(pos_file=pos_file).iloc[rerange_index]
    init_position=(init_record['x'],init_record['y'])
    rerange_dis=(po.euclidence_distance(intersection,init_position))

    return rerange_dis






def linear_array_distance(pos_file,left_index,rerange_index,right_index):
    '''
    cal the distance among left antenna, rerange antenna, right antenna

    parameters:
        pos_file:antenna position file
        left_index: the left antenna postion of rerange position
        rerange_index: the index of antenna postion which need moving
        right_index: the right antenna position of rerange position
    
    return the list of distance
    '''
    intersection=linear_array_rerange(pos_file,left_index,rerange_index,right_index)
    #left
    init_record_left=po.antenna_pos(pos_file=pos_file).iloc[left_index]
    init_position_left=(init_record_left['x'],init_record_left['y'])
    #right
    init_record_right=po.antenna_pos(pos_file=pos_file).iloc[right_index]
    init_position_right=(init_record_right['x'],init_record_right['y'])
    #distance
    dis_left=(po.euclidence_distance(intersection,init_position_left))
    dis_right=po.euclidence_distance(intersection,init_position_right)

    distance_list=[dis_right,0,dis_left]

    return distance_list






#=======================================================difference method=========================================================
#=======================================================difference method=========================================================





def diff_method(ntwk):
    '''
    use test signals minus empty signals to get a new signal

    parameters:
        the signals for difference
    '''
    empty_ntwk=rf.Network('/Users/charles/Desktop/Target/HM/CalibratedData/empty-calq.s16p')
    empty_scatter_array=sp.linear_array_scatter(empty_ntwk)
    #the scatters signals for detecting target
    test_scatter_array=sp.linear_array_scatter(ntwk)

    diff_scatter_array=[]
    for index in range(len(empty_scatter_array)):
        diff_scatter=test_scatter_array[index]-empty_scatter_array[index]
        diff_scatter_array.append(diff_scatter.tolist())
    
    return np.array(diff_scatter_array)








































if __name__=='__main__':
    pass