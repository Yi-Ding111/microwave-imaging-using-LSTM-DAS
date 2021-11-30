#author: YI DING
#email: dydifferent@gmail.com
#The University of Queensland 
#Oct 2021

import numpy as np
import pandas as pd
import skrf as rf
import target_data_pre as tp
import signal_process as sp
import math 
from math import pi
port_num=16
c0=2.24e11





def delay_signal(ntwk,antenna_index,x,t0,degree,m):
    '''
    cal the signal after delay

    parameters:
        ntwk: s16p file, use rf.Network()
        antenna_index: the index of antenna
        x: the scatter signal input
        t0: the delay time
        degree: incident angle
        m: the m th antenna
    '''

    #get the array of frequency
    freq=ntwk.f

    #get the step of between two frequencies
    delta_freq=freq[1]-freq[0]
    num_zero_freq=len(np.arange(0,freq[0],delta_freq))

    #create zero frequency array
    zero_freq_array=np.zeros(num_zero_freq)
    
    #put zero frequency and Sij together
    freq_half=np.hstack((zero_freq_array,ntwk.s[:,antenna_index,(antenna_index+3)%16]))

    #do conjugation and reversation to get the other half
    conj_reverse_freq_half=(((np.delete(freq_half,0)).conjugate()))[::-1]
    freq_full=np.hstack((freq_half,conj_reverse_freq_half))

    '''
    X_f = np.fft.rfft(x)
    
    N_dft = x.shape[0]
    #f = freq_full[:N_dft//2+1]
    #f=np.linspace(0.5,2,1001)
    f=np.linspace(0.5e9,2e9,26)
    
    X_f_delayed = X_f*np.exp(-1j*2*np.pi*f*t0)
    
    return np.fft.irfft(X_f_delayed)
    '''

    t=sp.signal_time(ntwk)
    if m !=-2:
        if degree<=90:
            for i in range(len(t)):
                if t[i]>t0:
                    delay_index=i
                    for j in range(delay_index):
                        if m==-1:
                            x.insert(j,0)
                            x.pop()
                        else:
                            x.remove(x[j])
                            x.append(0)
                break
        
        else:
            for i in range(len(t)):
                if t[i]>t0:
                    delay_index=i
                    for j in range(delay_index):
                        if m==-3:
                            x.insert(j,0)
                            x.pop()
                        else:
                            x.remove(x[j])
                            x.append(0)
                break
    #return x
    
    X_f = np.fft.rfft(x)
    
    N_dft = x.shape[0]
    #f = freq_full[:N_dft//2+1]
    f=np.linspace(0.5e10,2e10,1001)
    #f=np.linspace(0.5e9,2e9,26)
    
    X_f_delayed = X_f*np.exp(-1j*2*np.pi*f*t0)
    
    return np.fft.irfft(X_f_delayed)






def das_beamformer(pos_file,left_index,rerange_index,right_index,scatter_array,ntwk):
    '''
    delay and sum beamformer

    parameters:
        pos_file:antenna position file
        left_index: the left antenna postion of rerange position
        rerange_index: the index of antenna postion which need moving
        right_index: the right antenna position of rerange position
        scatter_array: the array of scatters [S14,S25,S36]
        ntwk: ntwk: s16p file, use rf.Network()
    '''
    #antenna_index=np.array([left_index,rerange_index,right_index])
    
    # 180 degree direction
    N_theta = 181
    theta = np.linspace(0, np.pi, N_theta)
    
    #define the weights of each signal in delay-and-sum beamformer
    weight=np.array([1,1,1])

    beamformer_array=np.zeros((N_theta,scatter_array.shape[1]-1))
    #beamformer_array=np.zeros((N_theta,scatter_array.shape[1]))

    #vector of antenna indices
    antenna_index=[1,0,1]

    #inter-antenna spacing
    inter_dis=tp.linear_array_distance(pos_file,left_index,rerange_index,right_index)

    antenna_dis=[]
    for i in range(3):
        value=antenna_index[i]*inter_dis[i]
        antenna_dis.append(value)

    #cal delay-and-sum beamformer for each partition of 180 degree
    for theta_i in range(N_theta):
        degree=math.degrees(theta_i)
        # calculate candidate time delays
        time_delays = np.array(antenna_dis)*np.cos(theta[theta_i])/c0
        #print(time_delays)
    
        #each antenna flashback
        for m in [-1,-2,-3]:
            # delay and sum signals
            beamformer_array[theta_i,:] += weight[m]*delay_signal(ntwk,antenna_index[m],scatter_array[m, :], time_delays[m],degree,m)
    
    # compensate for No. of sensors in array
    beamformer_array *= 1./len(scatter_array)

    return beamformer_array





















if __name__=='__main__':
    pass