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
import math


v=3e8



#============================================cal distance and time between antenna position and boundary========================================
#============================================cal distance and time between antenna position and boundary========================================





def antenna_pos(pos_file):
    '''
    read all antenna positions:(x,y,degree)

    parameters: 
        pos_file: antenna position file
    '''
    col_name=['x','y','degree']
    antenna_df=pd.read_csv(pos_file,names=col_name)

    return antenna_df





def antenma_pos_limit(pos_file):
    '''
    Find the maximum and minimum coordinates

    parameters:
         pos_file: antenna position file
    '''
    antenna_df=antenna_pos(pos_file)
    max_x=antenna_df['x'].max()
    min_x=antenna_df['x'].min()
    max_y=antenna_df['y'].max()
    min_y=antenna_df['y'].min()

    return max_x,min_x,max_y,min_y






def intersection_limit(pos_file):
    '''
    return the value of X_square + Y_square

    parameters:
         pos_file: antenna position file
    '''
    antenna_df=antenna_pos(pos_file)
    x=(antenna_df.iloc[1])['x']
    y=(antenna_df.iloc[1])['y']

    limit=x**2+y**2

    return limit






def boundary_pos(bound_file):
    '''
    read ground truth boundary positions:(x,y)

    parameters:
        bound_file: boundary ground truth file
    '''
    col_name=['x','y']
    boundary_df=pd.read_csv(bound_file,names=col_name)

    return boundary_df





def degree_binary_linear_equation(pos_file,index):
    '''
    according to the degree of antenna transmitting, cal the equation (y=tan alpha x +b)

    parameters:
        pos_file: antenna position file
        index: the index of dataframe from 0 to 15
    '''
    antenna_df=antenna_pos(pos_file)
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





def point_equation_distance(p,pos_file,index_antena):
    '''
    cal the distance of one point coordinate to the binary linear equation

    paranmeters:
        p: the point coordinate (x,y)
        pos_file: antenna position file
        index_antenna: the index of dataframe from 0 to 15
    '''
    # formula: | Ax0 + By0 + C | / sqrt(A**2 + B**2) --- y=tan alpha x +b --- tan alpha x -y +b =0
    A,B,C=degree_binary_linear_equation(pos_file,index_antena)
    #distance
    point_intercept=abs(A*p[0]+B*p[1]+C)/math.sqrt(A**2+B**2)

    return point_intercept





def euclidence_distance(p1,p2):
    '''
    cal the euclidence distance between two points

    parameters:
        p1: first point coordinate (x1,y1)
        p2: second point coordinate (x2,y2)
    '''
    points_dis=np.sqrt(((p1[0]-p2[0])**2) + ((p1[1]-p2[1])**2))
    return points_dis





def bound_ante_point(pos_file,bound_file):
    '''
    considering the antenna degree
    grab the boundary point which is closest to one antenna coordinate. 

    parameters:
        pos_file: antenna position file
        bound_file: boundary ground truth file
    '''
    antenna_df=antenna_pos(pos_file) # df of antenna position
    boundary_df=boundary_pos(bound_file) # df of ground truth points

    #calculate 45 points which are the closest boundary points to the antenna and let them be candidates
    distance_can=[]#the candidate boundary points of each antenna. array [[(x,y),(x,y)...45...(xy)],[(x,y),(x,y)...45...(xy)]...16]
    for i in range(len(antenna_df)):
        all_distance=[]
        for j in range(len(boundary_df)):
            #grab antenna position p1(p1_x,p1_y) and boundary point p2(p2_x,p2_y)
            p1=antenna_df.iloc[i]
            p2=boundary_df.iloc[j]
            points_dis=euclidence_distance(p1,p2) # cal the distance between p1 and p2
            all_distance.append(points_dis)
        distance_min_index=list(map(all_distance.index, heapq.nsmallest(45,all_distance))) # find the top 45 smallest distance indexes in list

        #grab top 45 cloest distance boundary points coordinates (x,y)
        each_45_points_tuple=[]
        for index in distance_min_index:
            p_x,p_y=(boundary_df.iloc[index])[0], (boundary_df.iloc[index])[1]
            p_coor=(p_x,p_y)
            each_45_points_tuple.append(p_coor)

        distance_can.append(each_45_points_tuple)
    
    #find the point from diatance_can has the cloest distance with the antenna
    closet_coordinate=[]
    for m in range(len(antenna_df)):
        point_distance_list=[]
        points_cluster=distance_can[m]
        for n in range(len(points_cluster)):
            point_single=points_cluster[n]
            point_intercept=point_equation_distance(point_single,pos_file,m)
            point_distance_list.append(point_intercept) # have 45 values of distance
        closet_distance_index=point_distance_list.index(min(point_distance_list))
        closet_distance_coor=points_cluster[closet_distance_index] # the closet point coordinate (x,y)
        closet_coordinate.append(closet_distance_coor) # one to one correspondence between 16 tuples and antenna positions
    
    return closet_coordinate






def antenna_to_boundary_time(pos_file,bound_file):
    '''
    return 16 reflection time between 16 antennas and the corresponding boundary points from bound_ante_point()

    parameters:
        pos_file: antenna position file
        bound_file: boundary ground truth file
    '''
    antenna_df=antenna_pos(pos_file)
    closet_coordinate=bound_ante_point(pos_file,bound_file)
    time_closet_list=[]
    for index in range(len(antenna_df)):
        antenna_point=antenna_df.iloc[index]
        closet_bound_point=closet_coordinate[index]
        #cal two points distance
        dis_to_closet_point=euclidence_distance(antenna_point,closet_bound_point)
        time_to_closet_point=(dis_to_closet_point*0.01)/v   #time=distance/velocity
        time_closet_list.append(time_to_closet_point)
        #time_closet_list.append(dis_to_closet_point)
    
    return time_closet_list






#==============================================convert time to distance and coordinate===========================================
#==============================================convert time to distance and coordinate===========================================





def reflection_to_bound_point(pos_file,prediction_list):
    '''
    return 16 points of boundary calculating by prediction

    parameters:
        pos_file: pos_file: antenna position file
        prediction_list: the 16 prediction value in form of list
    '''
    antenna_df=antenna_pos(pos_file)

    predict_boundary_position=[]

    for index in range(len(prediction_list)):
        reflection_value=prediction_list[index]
        position_ant=antenna_df.iloc[index]
        A,B,C=degree_binary_linear_equation(pos_file,index)
        a_x,a_y=position_ant['x'],position_ant['y']
        
        if index==0:#degree=90
            y=sympy.symbols('y')
            ans=sympy.solve([(a_x)**2+(y-a_y)**2-reflection_value**2],[y])
            y_1,y_2=ans[0][0].evalf(),ans[1][0].evalf()
            if a_y>y_1:
                predict_boundary_position.append((0,y_1))
            else:
                predict_boundary_position.append((0,y_2))
        
        elif index in range(1,4) or index in range(5,8): #degree in (90,270) except 0
            x=sympy.symbols('x')
            ans=sympy.solve([(x-a_x)**2+(((-C-A*x)/B)-a_y)**2-reflection_value**2],[x])
            x_1,x_2=ans[0][0].evalf(),ans[1][0].evalf()
            if a_x>x_1:
                pos_y=(-C-A*x_1)/B
                predict_boundary_position.append((x_1,pos_y))
            else:
                pos_y=(-C-A*x_2)/B
                predict_boundary_position.append((x_2,pos_y))

        elif index==4: #degree=0
            x=sympy.symbols('x')
            ans=sympy.solve([(x-a_x)**2+(a_y)**2-reflection_value**2],[x])
            x_1,x_2=ans[0][0].evalf(),ans[1][0].evalf()
            if a_x>x_1:
                predict_boundary_position.append((x_1,0))
            else:
                predict_boundary_position.append((x_2,0))
        
        elif index==8: #degree=270
            y=sympy.symbols('y')
            ans=sympy.solve([(a_x)**2+(y-a_y)**2-reflection_value**2],[y])
            y_1,y_2=ans[0][0].evalf(),ans[1][0].evalf()
            if a_y<y_1:
                predict_boundary_position.append((0,y_1))
            else:
                predict_boundary_position.append((0,y_2))
            
        elif index in range(9,12) or index in range(13,16):
            x=sympy.symbols('x')
            ans=sympy.solve([(x-a_x)**2+(((-C-A*x)/B)-a_y)**2-reflection_value**2],[x])
            x_1,x_2=ans[0][0].evalf(),ans[1][0].evalf()
            if a_x<x_1:
                pos_y=(-C-A*x_1)/B
                predict_boundary_position.append((x_1,pos_y))
            else:
                pos_y=(-C-A*x_2)/B
                predict_boundary_position.append((x_2,pos_y))

        elif index==12: #degree=360
            x=sympy.symbols('x')
            ans=sympy.solve([(x-a_x)**2+(a_y)**2-reflection_value**2],[x])
            x_1,x_2=ans[0][0].evalf(),ans[1][0].evalf()
            if a_x<x_1:
                predict_boundary_position.append((x_1,0))
            else:
                predict_boundary_position.append((x_2,0))
        
    return predict_boundary_position








#=============================================bezier curve===========================================
#=============================================bezier curve===========================================






# 
def get_bezier_coef(points):
    '''
    find the interpolation: a & b points
    '''
    n = len(points) - 1

    # build coefficents matrix
    C = 4 * np.identity(n)
    np.fill_diagonal(C[1:], 1)
    np.fill_diagonal(C[:, 1:], 1)
    C[0, 0] = 2
    C[n - 1, n - 1] = 7
    C[n - 1, n - 2] = 2

    # build points vector
    P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
    P[0] = points[0] + 2 * points[1]
    P[n - 1] = 8 * points[n - 1] + points[n]

    A = np.linalg.solve(C, P)
    B = [0] * n
    for i in range(n - 1):
        B[i] = 2 * points[i + 1] - A[i + 1]
    B[n - 1] = (A[n - 1] + points[n]) / 2

    return A, B







def get_cubic(a, b, c, d):
    '''
    returns the general Bezier cubic formula given 4 control points
    '''
    return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d







def get_bezier_cubic(points):
    '''
    return one cubic curve for each consecutive points
    '''
    A, B = get_bezier_coef(points)

    return [
        get_cubic(points[i], A[i], B[i], points[i + 1])
        for i in range(len(points) - 1)
    ]







def evaluate_bezier(points, n):
    '''
    evalute each cubic curve on the range [0, 1] sliced in n points
    '''
    curves = get_bezier_cubic(points)
    
    return np.array([fun(t) for fun in curves for t in np.linspace(0, 1, n)])






def curve_draw(coor_predict):
    '''
    draw the bezier curve

    parameters:
        coor_predict: the coordinates array for bezier curve
    '''
    points=(np.array(coor_predict)).astype('float64')
    # fit the points with Bezier interpolation
    # use 50 points between each consecutive points to draw the curve
    path = evaluate_bezier(points, 50)
    # extract x & y coordinates of points
    x, y = points[:,0], points[:,1]
    px, py = path[:,0], path[:,1]
    #print(px)
    #print(py)
    # plot
    #plt.figure(figsize=(11, 8))
    plt.plot(px, py, 'b-')
    #plt.plot(x, y, 'ro')
    #plt.show()







#============================================accuracy for boundary============================================
#============================================accuracy for boundary============================================






def rmse(initial_point_array,prediction_point_array):
    '''
    cal rmse of intial point and boundary point

    parameters:
        initial_point_array: the closest points to the antenna on the boundary
        prediction_point_array: the prediction points for reconstruction
    '''
    single_diff=[]
    for index in range(len(initial_point_array)):
        initx,inity=initial_point_array[index][0],initial_point_array[index][1]
        predx,predy=prediction_point_array[index][0],prediction_point_array[index][1]
        difference=math.sqrt((initx-predx)**2+(inity-predy)**2)
        single_diff.append(difference)
    
    return np.sum(single_diff)/len(initial_point_array)









    


if __name__=="__main__":
    pass