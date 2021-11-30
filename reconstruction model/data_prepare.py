#author: YI DING
#email: dydifferent@gmail.com
#The University of Queensland 
#Oct 2021

import os
from numpy.ma.core import maximum_fill_value
import pandas as pd
import numpy as np
import skrf as rf
from numpy.fft import fft,ifft,fftfreq
import pandas as pd
from pylab import *
from modulefinder import *
from tensorflow.python.keras.saving.save import load_model

from signal_process import inc_difference
import learning_model as lm
import antenna_boundary_position as po
import signal_process as sp




##==================================dataset prepare===========================================
##==================================dataset prepare===========================================





def readall_file_name_sort(abs_path):
    '''
    return the list of all snp file names (eg:IF10k-CNC-Exp0000.s16p in Data or Exp0000.csv in GroundTruth) sorted

    Parameters:
        abs_path:the path of the folder including targeting files 
    '''
    file_name=os.listdir(abs_path)

    if '.DS_Store' in file_name:
        file_name.remove('.DS_Store')
    
    file_name.sort(key=lambda x:int((x.split('.')[0])[-4:])) # sort the file name according to the last 4 digits in name (eg:IF10k-CNC-Exp0000.s16p)
    
    # in list eg:['IF10k-CNC-Exp0000.s16p', 'IF10k-CNC-Exp0001.s16p', 'IF10k-CNC-Exp0002.s16p', 'IF10k-CNC-Exp0003.s16p'...]
    return file_name 

#print(readall_file_name_sort('./20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container3/GroundTruth'))





def all_filepath_list(abs_path,file_paths):
    '''
    return all file paths sorted list
    
    Parameters:
        abs_path: the path of the folder including targeting files
        file_paths: []
    '''
    domain=os.path.abspath(abs_path) #the folder including targeting data files abspath: os.path.abspath(r'path')
    file_name_list=readall_file_name_sort(abs_path)
    #file_paths=[]
    for files in file_name_list:
        data_file=os.path.join(domain,files)
        file_paths.append(data_file)
    
    return file_paths
    
'''
domain=os.path.abspath(r'./20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container1/GroundTruth')
file_name=readall_file_name_sort('./20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container1/GroundTruth')
file_paths=[]
for files in file_name:
    snp_file=os.path.join(domain,files)
    file_paths.append(snp_file)

#print(file_paths)


#ntwk=rf.Network(file_paths[0])
#a=sp.signal_complex_to_real(ntwk)
#print(a)
'''





def file_path_integration():
    '''
    integrate all data file paths (snp or groundtruth data) from different folders into one same list
    return two list: first list is s16p file paths integration, second list is groundtruth file paths integration
    '''
    #all training data file paths
    
    #s16p file absolute path
    snp_abspath_list=['./20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container1/Data',
                      './20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container2/Data',
                      './20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container3/Data']

    #ground truth file absolute path
    gt_abspath_list=['./20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container1/GroundTruth',
                     './20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container2/GroundTruth',
                     './20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container3/GroundTruth']


    #integrate all file paths into list
    snp_file_paths=[] #as one variable of all_filepath_list() this is for s16p file
    gt_file_paths=[] #this is for groundtruth file
    for i in range(len(snp_abspath_list)):
        snp_path=snp_abspath_list[i]
        gt_path=gt_abspath_list[i]
        all_filepath_list(snp_path,snp_file_paths)
        all_filepath_list(gt_path,gt_file_paths)
    
    #gt_file_paths=[] #this is for groundtruth file
    #for i in range(len(gt_abspath_list)):
       # gt_path=gt_abspath_list[i]
        #all_filepath_list(gt_path,gt_file_paths)

    return snp_file_paths , gt_file_paths






#===================================dataset generation================================
#===================================dataset generation================================






'''
the dataframe putting into LSTM as input should has two columns, one column for signal data in time domain,
the other column is the time for each signal (reflection signal like s11) from antenna position to boundary.
the LSTM model is assumned in the category of MANY-TO-ONE,
so each row of the dataframe shoule look like ([7.3234534,4.23472839042,7.234790423....],0.5e-8)
'''





def normalization(input_list):
    '''
    return list after implementing minmaxscalar normalization
    
    parameters:
        input_list: the list of signal or reflection value
    '''
    min_value=min(input_list)
    max_value=max(input_list)
    scale_list=[((i-min_value)/(max_value-min_value)) for i in input_list]
    
    return scale_list





def normalization_return(reflection_value,class_S):
    '''
    return the original reflection value

     parameters:
        reflection_value: the minmaxscallar value 
        class_S: the class of reflection_value
    '''
    s={'S11': [12.150000000000006, 28.549999999999997], 
        'S22': [13.383870665842526, 27.22349169375597], 
        'S33': [13.1285233366133, 26.055901634754463], 
        'S44': [13.39927598790323, 26.806559141374333], 
        'S55': [12.299999999999997, 26.1], 
        'S66': [11.7419634218473, 25.402310485465684], 
        'S77': [11.594311018771235, 24.27203567894544], 
        'S88': [11.004989095860102, 27.273655053916045], 
        'S99': [9.049999999999997, 29.400000000000006], 
        'S1010': [7.674782863377963, 25.052306400808693], 
        'S1111': [6.183902974659294, 19.27311791070662], 
        'S1212': [6.311918884142925, 19.833529690904736], 
        'S1313': [6.700000000000003, 20.5], 
        'S1414': [8.501034760545334, 21.919403846820295], 
        'S1515': [9.361608408815238, 21.999417901390036], 
        'S1616': [10.867067911815044, 24.853503616190626]}

    min_value=s[class_S][0]
    max_value=s[class_S][1]
    original=(reflection_value*(max_value-min_value))+min_value

    return original






def signal_col_lstm():
    '''
    return the list including all reflection signal sequences.
    output as the first colume of learning machine input dataframe
    '''
    snp_file_paths=file_path_integration()[0]

    signal_col=[]
    for i in range(len(snp_file_paths)):
        ntwk=rf.Network(snp_file_paths[i])
        real_reflection=sp.signal_complex_to_real(ntwk)
        keep_len_signal=sp.keep_len_signal(real_reflection)

        for j in range(len(keep_len_signal)):
            signal_col.append(keep_len_signal[j])

    return signal_col


#a=signal_col_lstm()
#a=np.array(a)
#print(a.shape)







def signal_col_lstm_16():
    '''
    return 16 list including Sii signal sequences sepreately
    output as the first colume of learning machine input dataframe
    '''
    snp_file_paths=file_path_integration()[0]
    
    signal_col_s11,signal_col_s22,signal_col_s33,signal_col_s44,signal_col_s55=[],[],[],[],[]
    signal_col_s66,signal_col_s77,signal_col_s88,signal_col_s99,signal_col_s1010=[],[],[],[],[]
    signal_col_s1111,signal_col_s1212,signal_col_s1313,signal_col_s1414,signal_col_s1515,signal_col_s1616=[],[],[],[],[],[]
    for i in range(len(snp_file_paths)):
        ntwk=rf.Network(snp_file_paths[i])
        real_reflection=sp.signal_complex_to_real(ntwk)
        keep_len_signal=sp.keep_len_signal(real_reflection)

        #each signal list needs normalization 
        #append all different signal sequences to different list
        signal_col_s11.append(normalization(keep_len_signal[0]))
        signal_col_s22.append(normalization(keep_len_signal[1]))
        signal_col_s33.append(normalization(keep_len_signal[2]))
        signal_col_s44.append(normalization(keep_len_signal[3]))
        signal_col_s55.append(normalization(keep_len_signal[4]))
        signal_col_s66.append(normalization(keep_len_signal[5]))
        signal_col_s77.append(normalization(keep_len_signal[6]))
        signal_col_s88.append(normalization(keep_len_signal[7]))
        signal_col_s99.append(normalization(keep_len_signal[8]))
        signal_col_s1010.append(normalization(keep_len_signal[9]))
        signal_col_s1111.append(normalization(keep_len_signal[10]))
        signal_col_s1212.append(normalization(keep_len_signal[11]))
        signal_col_s1313.append(normalization(keep_len_signal[12]))
        signal_col_s1414.append(normalization(keep_len_signal[13]))
        signal_col_s1515.append(normalization(keep_len_signal[14]))
        signal_col_s1616.append(normalization(keep_len_signal[15]))

    return signal_col_s11,signal_col_s22,signal_col_s33,signal_col_s44,signal_col_s55,signal_col_s66,signal_col_s77,signal_col_s88,signal_col_s99,signal_col_s1010,signal_col_s1111,signal_col_s1212,signal_col_s1313,signal_col_s1414,signal_col_s1515,signal_col_s1616






def reflection_time_column_lstm():
    '''
    return the list including all reflection time from antenna position to boundary.
    output as the second colume of learning machine input dataframe
    '''
    gt_file_paths=file_path_integration()[1]

    #antenna position coordinates
    pos_file='./20200309-AS-CNC-Boundary-Movement/LogPeriodic/AntennaPositions.csv'

    reflection_col=[]
    for i in range(len(gt_file_paths)):
        bound_file=gt_file_paths[i]
        reflection_time=po.antenna_to_boundary_time(pos_file,bound_file)

        for j in range(len(reflection_time)):
            reflection_col.append(reflection_time[j])

    return reflection_col


#a=reflection_time_column_lstm()
#print(a)






def reflection_col_lstm_16():
    '''
    return the list including all reflection values from antenna position to boundary
    store in lists by different Sii seperately
    output as the second colume of learning machine input dataframe
    '''
    gt_file_paths=file_path_integration()[1]

    #antenna position coordinates
    pos_file='./20200309-AS-CNC-Boundary-Movement/LogPeriodic/AntennaPositions.csv'

    ref_col_s11,ref_col_s22,ref_col_s33,ref_col_s44,ref_col_s55=[],[],[],[],[]
    ref_col_s66,ref_col_s77,ref_col_s88,ref_col_s99,ref_col_s1010=[],[],[],[],[]
    ref_col_s1111,ref_col_s1212,ref_col_s1313,ref_col_s1414,ref_col_s1515,ref_col_s1616=[],[],[],[],[],[]
    for i in range(len(gt_file_paths)):
        bound_file=gt_file_paths[i]
        reflection_value=po.antenna_to_boundary_time(pos_file,bound_file)


        ref_col_s11.append(round(reflection_value[0],15))#keep 15 digits after decimal point
        ref_col_s22.append(round(reflection_value[1],15))
        ref_col_s33.append(round(reflection_value[2],15))
        ref_col_s44.append(round(reflection_value[3],15))
        ref_col_s55.append(round(reflection_value[4],15))
        ref_col_s66.append(round(reflection_value[5],15))
        ref_col_s77.append(round(reflection_value[6],15))
        ref_col_s88.append(round(reflection_value[7],15))
        ref_col_s99.append(round(reflection_value[8],15))
        ref_col_s1010.append(round(reflection_value[9],15))
        ref_col_s1111.append(round(reflection_value[10],15))
        ref_col_s1212.append(round(reflection_value[11],15))
        ref_col_s1313.append(round(reflection_value[12],15))
        ref_col_s1414.append(round(reflection_value[13],15))
        ref_col_s1515.append(round(reflection_value[14],15))
        ref_col_s1616.append(round(reflection_value[15],15))

    return ref_col_s11,ref_col_s22,ref_col_s33,ref_col_s44,ref_col_s55,ref_col_s66,ref_col_s77,ref_col_s88,ref_col_s99,ref_col_s1010,ref_col_s1111,ref_col_s1212,ref_col_s1313,ref_col_s1414,ref_col_s1515,ref_col_s1616

    



def reflection_col_lstm_16_nor():
    '''
    store the min and max value of each reflection list into dic
    using for recovering normalization 
    '''
    reflection_list=reflection_col_lstm_16()

    #the min and max value of each reflection list to store in dic, using for recovering normalization 
    recover_nor={}
    for i in range(16):
        min_value, max_value=min(reflection_list[i]), max(reflection_list[i])
        recover_nor['S{}{}'.format(i+1,i+1)]=[min_value,max_value]
    
    return recover_nor

# {'S11': [12.150000000000006, 28.549999999999997], 
   #'S22': [13.383870665842526, 27.22349169375597], 
   #'S33': [13.1285233366133, 26.055901634754463], 
   #'S44': [13.39927598790323, 26.806559141374333], 
   #'S55': [12.299999999999997, 26.1], 
   #'S66': [11.7419634218473, 25.402310485465684], 
   #'S77': [11.594311018771235, 24.27203567894544], 
   #'S88': [11.004989095860102, 27.273655053916045], 
   #'S99': [9.049999999999997, 29.400000000000006], 
   #'S1010': [7.674782863377963, 25.052306400808693], 
   #'S1111': [6.183902974659294, 19.27311791070662], 
   #'S1212': [6.311918884142925, 19.833529690904736], 
   #'S1313': [6.700000000000003, 20.5], 
   #'S1414': [8.501034760545334, 21.919403846820295], 
   #'S1515': [9.361608408815238, 21.999417901390036], 
   #'S1616': [10.867067911815044, 24.853503616190626]}





#=========================================store data=========================================
#=========================================store data=========================================






#transfer df to csv, saving time for following cal
'''
signal_col=signal_col_lstm() #the first column of df, reflection signal sequences
reflection_col=reflection_time_column_lstm()# the second column of df, reflection time from antenna position to boundary

df_learning={'signal_timesteps':signal_col,'reflection_time':reflection_col}
df_learning_file=pd.DataFrame(df_learning)
df_learning_file.to_csv('/Users/charles/Desktop/boundary_data/data_for_lm_300.csv')
'''


'''
signal_ret=signal_col_lstm_16()
reflection_ret=reflection_col_lstm_16()

for index in range(16):
    signal_col=signal_ret[index] #the first column of df, reflection signal sequences

    #do normalization for each reflection list
    #the second column of df, reflection time from antenna position to boundary
    reflection_col=normalization(reflection_ret[index])  

    df_learning={'signal_timesteps':signal_col,'reflection_value':reflection_col}
    df_learning_file=pd.DataFrame(df_learning)
    df_learning_file.to_csv('/Users/charles/Desktop/boundary_data/data_for_lm_S{i}{i}.csv'.format(i=index+1))
'''





#==========================================prepare test array=========================================
#==========================================prepare test array=========================================






def test_output(test_file_path,model_name):
    '''
    prepare the array for the test input and return the prediction value list

    parameters:
        test_file_path: the file for prediction
        model_name: the list of the corresponding learning model
    '''
    ntwk=rf.Network(test_file_path)
    real_reflection=sp.signal_complex_to_real(ntwk)#convert complex values to real one
    keep_len_signal=sp.keep_len_signal(real_reflection)#keep one part of signals instead of whole as input

    reflection_list=[]
    for index in range(len(keep_len_signal)):
        keep_len_signal_Sii=normalization(keep_len_signal[index])
        test_input=np.array(keep_len_signal_Sii)
        test=np.reshape(test_input,(1,len(test_input),1))
        model=load_model(model_name[index])
        outcome=lm.lstm_predict(model,test)
        ori_dis=normalization_return(outcome[0],'S{}{}'.format(index+1,index+1))
        reflection_list.append(ori_dis[0])
    
    return reflection_list





















if __name__=='__main__':
    pass