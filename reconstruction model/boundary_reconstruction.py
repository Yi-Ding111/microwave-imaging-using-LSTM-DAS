#author: YI DING
#email: dydifferent@gmail.com
#The University of Queensland 
#Oct 2021

import numpy as np
import skrf as rf
import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
from ast import literal_eval

from tensorflow.python.keras.saving.save import load_model


import antenna_boundary_position as po
import signal_process as sp
import data_prepare as dp
import learning_model as lm

pos_file='./20200309-AS-CNC-Boundary-Movement/LogPeriodic/AntennaPositions.csv'



'''
#prepare dataset as input for LSTM
#signal_col=dp.signal_col_lstm() #the first column of df, reflection signal sequences
#reflection_col=dp.reflection_time_column_lstm() # the second column of df, reflection time from antenna position to boundary
signal_ret=dp.signal_col_lstm_16()
reflection_ret=dp.reflection_col_lstm_16()

for index in range(16):
    signal_col=signal_ret[index] #the first column of df, reflection signal sequences

    #do normalization for each reflection list
    #the second column of df, reflection time from antenna position to boundary
    reflection_col=dp.normalization(reflection_ret[index])  

    df_learning={'signal_timesteps':signal_col,'reflection_value':reflection_col}
    df_learning_file=pd.DataFrame(df_learning)
'''


dataset_name=['data_for_lm_S11.csv','data_for_lm_S22.csv','data_for_lm_S33.csv','data_for_lm_S44.csv','data_for_lm_S55.csv',
              'data_for_lm_S66.csv','data_for_lm_S77.csv','data_for_lm_S88.csv','data_for_lm_S99.csv','data_for_lm_S1010.csv',
              'data_for_lm_S1111.csv','data_for_lm_S1212.csv','data_for_lm_S1313.csv','data_for_lm_S1414.csv','data_for_lm_S1515.csv','data_for_lm_S1616.csv']

#lstm trained model
model_name=['lstm_disS11.h5','lstm_disS22.h5','lstm_disS33.h5','lstm_disS44.h5','lstm_disS55.h5',
            'lstm_disS66.h5','lstm_disS77.h5','lstm_disS88.h5','lstm_disS99.h5','lstm_disS1010.h5',
            'lstm_disS1111.h5','lstm_disS1212.h5','lstm_disS1313.h5','lstm_disS1414.h5','lstm_disS1515.h5','lstm_disS1616.h5']

#test file path
test_file_name=['./20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container1/Data/IF10k-CNC-Exp0055.s16p',
                './20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container1/Data/IF10k-CNC-Exp0111.s16p',
                './20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container1/Data/IF10k-CNC-Exp0155.s16p',
                './20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container1/Data/IF10k-CNC-Exp0200.s16p',
                './20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container1/Data/IF10k-CNC-Exp0237.s16p']


#prediction
prediction_list=dp.test_output(test_file_name[4],model_name)
#print("prediction distances:",prediction_list)

#converting to coordinate
coor_prediction_list=po.reflection_to_bound_point(pos_file,prediction_list)
#print(coor_prediction_list)
#plot the prediction coordinates
position=po.antenna_pos(pos_file)
prediction_df=pd.DataFrame(coor_prediction_list)
#print('the transformation points:',prediction_df)

#the ground truth of boundary
bound_file='./20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container1/GroundTruth/Exp0237.csv'
position2=po.boundary_pos(bound_file)


closet_coordinate=pd.DataFrame(po.bound_ante_point(pos_file,bound_file))
#print(po.bound_ante_point(pos_file,bound_file))


#plot
plt.scatter(position['x'],position['y'],label='antenna position')
#plt.scatter(position2['x'],position2['y'],s=4.)
plt.scatter(closet_coordinate[0],closet_coordinate[1],s=4.,label='closet boundary points (ground truth)')
plt.scatter(prediction_df[0],prediction_df[1],s=8.,label='reconstructed boundary points')
plt.legend()
plt.show()





# the points array for bezier curve recunstruct
coor_prediction_list.append(po.reflection_to_bound_point(pos_file,prediction_list)[0])
#coor_prediction_array=np.array(coor_prediction_list)
#print(coor_prediction_array)
plt.scatter(position['x'],position['y'])
plt.scatter(position2['x'],position2['y'],s=4.)
plt.scatter(closet_coordinate[0],closet_coordinate[1],s=4.)
plt.scatter(prediction_df[0],prediction_df[1],s=4.)
po.curve_draw(coor_prediction_list)
plt.xlabel('EXP0237')
plt.legend()
plt.show()






#rmse
'''
#all file path
snp_file_paths=dp.file_path_integration()[0]
gt_file_paths=dp.file_path_integration()[1]

single_rmse=[]
for i in range(len(snp_file_paths)):
    ntwk=rf.Network(snp_file_paths[i])
    closet_coordinate=po.bound_ante_point(pos_file,gt_file_paths[i])
    prediction_dis=dp.test_output(snp_file_paths[i],model_name)
    coor_prediction=po.reflection_to_bound_point(pos_file,prediction_dis)
    accuracy=po.rmse(closet_coordinate,coor_prediction)
    single_rmse.append(accuracy)

print('the RMSE of boundary reconstruction is:', np.sum(single_rmse)/(i+1))
'''


'''
ntwk=rf.Network('./20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container1/Data/IF10k-CNC-Exp0237.s16p')
closet_coordinate=po.bound_ante_point(pos_file,'./20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container1/GroundTruth/Exp0237.csv')
prediction_dis=dp.test_output('./20200309-AS-CNC-Boundary-Movement/LogPeriodic/Container1/Data/IF10k-CNC-Exp0237.s16p',model_name)
coor_prediction=po.reflection_to_bound_point(pos_file,prediction_dis)
accuracy=po.rmse(closet_coordinate,coor_prediction)
print(accuracy)
'''


