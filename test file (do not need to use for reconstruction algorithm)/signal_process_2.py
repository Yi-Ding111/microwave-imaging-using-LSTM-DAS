import numpy as np
import skrf as rf
from numpy.fft import fft,ifft,fftfreq
from pylab import *


#the number of antennas
port_num=16
#the number of dimensions of signal in time domain
N=2001






def freq_domain_to_time_domain(ntwk):
    '''
    tranform frequency domain signal to time domain signal by IFFT

    parameters:
        ntwk: s16p file, use rf.Network() 
    '''
    td_signal=locals()
    #get the array of frequency
    freq=ntwk.f
    #get the step of between two frequencies
    delta_freq=freq[1]-freq[0]
    num_zero_freq=len(np.arange(0,freq[0],delta_freq))
    #create zero frequency array
    zero_freq_array=np.zeros(num_zero_freq)
    
    for i in range(0,port_num):
        for j in range(0,port_num):
            #put zero frequency and Sij together
            freq_half=np.hstack((zero_freq_array,ntwk.s[:,i,j]))
            #do conjugation and reversation to get the other half
            conj_reverse_freq_half=(((np.delete(freq_half,0)).conjugate()))[::-1]
            freq_full=np.hstack((freq_half,conj_reverse_freq_half))
            #time domain
            #td_signal['td'+str(i+1)+','+str(j+1)]=np.fft.ifft(freq_full)*len(ntwk.s[:,i,j])
            #print (len(abs(ntwk.s[:,i,j])))
            td_signal['td'+str(i+1)+','+str(j+1)]=   ntwk.s_db[:,i,j]  #np.fft.ifft(freq_full)*len(ntwk.s[:,i,j])
    td_signal.pop('ntwk')
    return td_signal






#=============================================reflection signals=========================================
#=============================================reflection signals=========================================






def reflection_td_signal(ntwk):
    '''
    filter all reflection time domain sognals : Si,i

    parameters:
        ntwk: s16p file, use rf.Network()
    '''
    td_signal=freq_domain_to_time_domain(ntwk)
    #all reflection signal keys in td_signal dictonary
    reflection_signal_keys={'td1,1','td2,2','td3,3','td4,4','td5,5','td6,6','td7,7','td8,8',
                       'td9,9','td10,10','td11,11','td12,12','td13,13','td14,14','td15,15','td16,16'}
    #the dict of all reflection signals in time domain
    reflection_signal_dict={ key:value for key,value in td_signal.items() if key in reflection_signal_keys} 
    #the list of all reflection signals in time domain
    reflection_signal_list=list(reflection_signal_dict.values())

    return reflection_signal_list




def signal_complex_to_real(ntwk):
    '''
    convert complex value signal lists to real values list

    parameters: s16p file, use rf.Network()
    '''
    reflection_signal_list=reflection_td_signal(ntwk)

    real_reflection_signal=[]
    for i in range(len(reflection_signal_list)):
        real_single_reflection=[]
        single_reflection=reflection_signal_list[i]
        for j in range(len(single_reflection)):
            single_reflection_index=single_reflection[j].real  #only keep real part
            real_single_reflection.append(single_reflection_index)
        real_reflection_signal.append(real_single_reflection)
    
    return real_reflection_signal



        
def keep_len_signal(signal_matrix):
    '''
    for cal boundary
    cut the signal length, keep one certain length of the front signal
    decrease cal time

    parameters:
        ntwk: s16p file, use rf.Network()
        signal_matrix: the matrix of corresponding matrix 
    '''
    quarter_signal_mat=[]
    for i in range(len(signal_matrix)):
        single_signal_list=signal_matrix[i]
        quarter_single_signal=single_signal_list[:200]   #adjust the keeping length 
        quarter_signal_mat.append(quarter_single_signal)
    
    return quarter_signal_mat

#ntwk=rf.Network('./20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container1/Data/IF10k-CNC-Exp0000.s16p')
#real_reflection_signal=sp.signal_complex_to_real(ntwk)
#quarter_signal=sp.keep_len_signal(real_reflection_signal)
#a=np.array(quarter_signal)
#print(a)






#===============================================scatter signals==========================================
#===============================================scatter signals==========================================






def scattering_td_signal(ntwk):
    '''
    filter all scatter time domain signals : Si,j
    return Sij signals in real form

    parameters:
        ntwk: s16p file, use rf.Network()
    '''
    scatter_signal_dict=freq_domain_to_time_domain(ntwk)
    for i in ['td1,1','td2,2','td3,3','td4,4','td5,5','td6,6','td7,7','td8,8',
                'td9,9','td10,10','td11,11','td12,12','td13,13','td14,14','td15,15','td16,16']:
        scatter_signal_dict.pop(i)

    #convert complex value signal lists to real values
    scatter_keys=list(scatter_signal_dict.keys())
    scatter_values=list(scatter_signal_dict.values())

    real_scatter_signal=[]
    for i in range(len(scatter_values)):
        real_single_scatter=[]
        single_scatter=scatter_values[i]
        for j in range(len(single_scatter)):
            single_scatter_index=single_scatter[j].real  #only keep real part
            real_single_scatter.append(single_scatter_index)
        real_scatter_signal.append(real_single_scatter)
    
    scatter_dict=dict(zip(scatter_keys,real_scatter_signal))


    return scatter_dict







def linear_array_scatter(ntwk):
    '''
    filtering scatter signals like Si,i+3

    parameters:
        ntwk: s16p file, use rf.Network()
    '''
    scatter_dict=scattering_td_signal(ntwk)
    scatter_signal_keys={'td1,4','td2,5','td3,6','td4,7','td5,8','td6,9','td7,10','td8,11',
                       'td9,12','td10,13','td11,14','td12,15','td13,16','td14,1','td15,2','td16,3'}
    #the dict of all reflection signals in time domain
    scatter_signal_dict={ key:value for key,value in scatter_dict.items() if key in scatter_signal_keys} 
    scatter_value_list=list(scatter_signal_dict.values())
    #scatter_value_list=list(scatter_dict.values())
    scatter_value_array=np.array(scatter_value_list)

    return scatter_value_array







def scatter_array_truncated(ntwk):
    '''
    truncate scatter signals to keep front part

    parameters:
        ntwk: s16p file, use rf.Network()
    '''
    scatter_array=linear_array_scatter(ntwk)

    #truncate
    scatter_truncated=[]
    
    for i in range(len(scatter_array)):
        scatter=np.abs(scatter_array[i][45:86])
        #scatter=scatter_array[i][20:61]
        #scatter=np.abs(scatter_array[i])
        scatter_truncated.append(scatter)
    
    return np.array(scatter_truncated)










#===============================================frequency domain=========================================
#===============================================frequency domain=========================================







def signal_fd_real(ntwk):
    '''
    return signals in freqiency domain (dict)

    Parameters:
        ntwk:s16p file, use rf.Network()
    '''
    signal_fd_dict={}
    for i in range(0,port_num):
        for j in range(0,port_num):
            single_signal_fd=ntwk.s[:,i,j]
            signal_real=[]
            for m in range(len(single_signal_fd)):
                signal_real_value=single_signal_fd[m].real
                signal_real.append(signal_real_value)
            signal_fd_dict['fd'+str(i+1)+','+str(j+1)]=signal_real
    
    return signal_fd_dict








def signal_time(ntwk):
    '''
    calculate the time period of signal

    parameters:
        ntwk: s16p file, use rf.Network()
    '''
    freq=ntwk.f
    delta_freq=freq[1]-freq[0]
    #time=(1/delta_freq)
    time=(1/20000000)
    t=np.arange(0,time,time/N)
    return t





def inc_difference(time_sequence,times):
    '''
    for each signal list, multiply signal sequence by times to increase difference among timesteps
    
    Parameters:
        time_sequences: the list of signal time sequence
        times: int. the number multiplied by signal
    '''
    sequence_array=np.array(time_sequence)
    sequence_by_times=sequence_array*times
    sequence_by_times=sequence_by_times.tolist()

    return sequence_by_times

#a=[2,3,4,5]
#a=np.array(a)
#b=a*5
#b=b.tolist()
#[10, 15, 20, 25]










if __name__=='__main__':
    pass
