# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 00:08:28 2018

@author: heera
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from numpy import genfromtxt
import math
from math import copysign
import operator
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import svm
from sklearn.linear_model import LinearRegression
import csv


train_data=[]
test_label=[]
TrainX=[]
TrainY=[]
TestX=[]
TestY=[]
c = []
header = []

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []  
    for x in range(k):
        neighbors.append(distances[x][0])     
    return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def main():
    k = 3
    for x in range(len(TestX)):
        neighbors = getNeighbors(train_data, TestX[x], k)
        result = getResponse(neighbors)
        test_label.append(int(result))
    return test_label

def pick_data():
    my_train_data = genfromtxt('C:\\Datamining project1\\GenomeTrainXY.csv', delimiter=',')
    a = np.transpose(my_train_data)
    np.savetxt('genome_train.csv', a, fmt='%f', delimiter=",")
    data = np.roll(a,-1,-1)
    for i in range(len(data)):
        train_data.append(data[i])
    np.savetxt('train_data.csv', train_data, fmt='%f', delimiter=",")
    my_test_data = genfromtxt('C:\\Datamining project1\\GenomeTestX.csv', delimiter=',')
    c = np.transpose(my_test_data)
    for i in range (0, len(c)):
        TestX.append(c[i][:])
    np.savetxt('TestX.csv', TestX, fmt='%f', delimiter=",")
    for i in range (0,len(a)):
        TrainY.append(a[i][0])
        TrainX.append(a[i][1:])            
    np.savetxt('TrainX.csv', TrainX, fmt='%f', delimiter=",") 
    np.savetxt('TrainY.csv', TrainY, fmt='%f', delimiter=",")
  
def class_separation():
    my_data = genfromtxt('C:\\Datamining project1\\GenomeTrainXY.csv', delimiter=',')
    z = np.transpose(my_data)
    df1 = pd.DataFrame(z[0:11])
    d1 = df1.drop(df1.columns[0], axis = 1)
    np.savetxt('1_train_data.csv', d1, fmt='%f', delimiter=",")
    df2 = pd.DataFrame(z[11:17])
    d2 = df2.drop(df2.columns[0], axis = 1)
    np.savetxt('2_train_data.csv', d2, fmt='%f', delimiter=",")
    df3 = pd.DataFrame( z[17:28])
    d3 = df3.drop(df3.columns[0], axis = 1)
    np.savetxt('3_train_data.csv',d3, fmt='%f', delimiter=",")
    df4 = pd.DataFrame(z[28:])
    d4 = df4.drop(df4.columns[0], axis = 1)
    np.savetxt('4_train_data.csv',d4 , fmt='%f', delimiter=",")

def avg_of_each_class_for_a_label():
    class_1_sum = []
    data = genfromtxt('C:\\Users\\heera\\Desktop\\1_train_data.csv', delimiter=',')
    df = pd.DataFrame(data)
    class_1_sum.append(df.sum(axis = 0)/(len(data)))
    np.savetxt('class_1_avg.csv',class_1_sum , fmt='%f', delimiter=",")
    
    class_2_sum = []
    data1 = genfromtxt('C:\\Users\\heera\\Desktop\\2_train_data.csv', delimiter=',')
    df1 = pd.DataFrame(data1)
    class_2_sum.append(df1.sum(axis = 0)/(len(data1)))
    np.savetxt('class_2_avg.csv',class_2_sum , fmt='%f', delimiter=",")
    
    class_3_sum = []
    data2 = genfromtxt('C:\\Users\\heera\\Desktop\\3_train_data.csv', delimiter=',')
    df2 = pd.DataFrame(data2)
    class_3_sum.append(df2.sum(axis = 0)/(len(data2)))
    np.savetxt('class_3_avg.csv',class_3_sum , fmt='%f', delimiter=",")
    
    class_4_sum = []
    data3 = genfromtxt('C:\\Users\\heera\\Desktop\\4_train_data.csv', delimiter=',')
    df3 = pd.DataFrame(data3)
    class_4_sum.append(df3.sum(axis = 0)/(len(data3)))
    np.savetxt('class_4_avg.csv',class_4_sum , fmt='%f', delimiter=",")
    
    total_sum = []
    data4 = genfromtxt('C:\\Users\\heera\\Desktop\\TrainX.csv', delimiter=',')
    df4 = pd.DataFrame(data4)
    total_sum.append(df4.sum(axis = 0)/len(data4))
    np.savetxt('total_avg.csv',total_sum , fmt='%f', delimiter=",")

def variance():
    val = []
    var = []
    first = genfromtxt('C:\\Users\\heera\\Desktop\\1_train_data.csv', delimiter=",")
    second = genfromtxt('C:\\Users\\heera\\Desktop\\class_1_avg.csv', delimiter=",")
    for i in range(len(first)):
        for j in range(len(second)):                
            val.append(((first[i][j]-second[j])**2))
    writer1=csv.writer(open('class_1_b_division.csv','w',newline = ""))
    writer1.writerow(val[0:4434]) 
    writer1.writerow(val[4434:8868])
    writer1.writerow(val[8868:13302])
    writer1.writerow(val[13302:17736])
    writer1.writerow(val[17736:22170])
    writer1.writerow(val[22170:26604])
    writer1.writerow(val[26604:31038])
    writer1.writerow(val[31038:35472])
    writer1.writerow(val[35472:39906])
    writer1.writerow(val[39906:44340])
    writer1.writerow(val[44340:48774])
    data = genfromtxt('C:\\Users\\heera\\Desktop\\class_1_b_division.csv', delimiter=',')
    df = pd.DataFrame(data)
    var.append(df.sum(axis = 0)/(len(data)-1))
    np.savetxt('class_1_var.csv',var , fmt='%f', delimiter=",")

    val1 = []
    var1 = []
    first1 = genfromtxt('C:\\Users\\heera\\Desktop\\2_train_data.csv', delimiter=",")
    second1 = genfromtxt('C:\\Users\\heera\\Desktop\\class_2_avg.csv', delimiter=",")
    for i in range(len(first1)):
        for j in range(len(second1)):                
            val1.append(((first1[i][j]-second1[j])**2))
    writer2=csv.writer(open('class_2_b_division.csv','w',newline = ""))
    writer2.writerow(val1[0:4434]) 
    writer2.writerow(val1[4434:8868])
    writer2.writerow(val1[8868:13302])
    writer2.writerow(val1[13302:17736])
    writer2.writerow(val1[17736:22170])
    writer2.writerow(val1[22170:26604])
    data1 = genfromtxt('C:\\Users\\heera\\Desktop\\class_2_b_division.csv', delimiter=',')
    df1 = pd.DataFrame(data1)
    var1.append(df1.sum(axis = 0)/(len(data1)-1))
    np.savetxt('class_2_var.csv',var1 , fmt='%f', delimiter=",")
    
    val2 = []
    var2 = []
    first2 = genfromtxt('C:\\Users\\heera\\Desktop\\3_train_data.csv', delimiter=",")
    second2 = genfromtxt('C:\\Users\\heera\\Desktop\\class_3_avg.csv', delimiter=",")
    for i in range(len(first2)):
        for j in range(len(second2)):                
            val2.append(((first2[i][j]-second2[j])**2))
    writer3=csv.writer(open('class_3_b_division.csv','w',newline = ""))
    writer3.writerow(val2[0:4434]) 
    writer3.writerow(val2[4434:8868])
    writer3.writerow(val2[8868:13302])
    writer3.writerow(val2[13302:17736])
    writer3.writerow(val2[17736:22170])
    writer3.writerow(val2[22170:26604])
    writer3.writerow(val2[26604:31038])
    writer3.writerow(val2[31038:35472])
    writer3.writerow(val2[35472:39906])
    writer3.writerow(val2[39906:44340])
    writer3.writerow(val2[44340:48774])
    data2 = genfromtxt('C:\\Users\\heera\\Desktop\\class_3_b_division.csv', delimiter=',')
    df2 = pd.DataFrame(data2)
    var2.append(df2.sum(axis = 0)/(len(data2)-1))
    np.savetxt('class_3_var.csv',var2 , fmt='%f', delimiter=",")
    
    val3 = []
    var3 = []
    first3 = genfromtxt('C:\\Users\\heera\\Desktop\\4_train_data.csv', delimiter=",")
    second3 = genfromtxt('C:\\Users\\heera\\Desktop\\class_4_avg.csv', delimiter=",")
    for i in range(len(first3)):
        for j in range(len(second3)):                
            val3.append(((first3[i][j]-second3[j])**2))
    writer4=csv.writer(open('class_4_b_division.csv','w',newline = ""))
    writer4.writerow(val3[0:4434]) 
    writer4.writerow(val3[4434:8868])
    writer4.writerow(val3[8868:13302])
    writer4.writerow(val3[13302:17736])
    writer4.writerow(val3[17736:22170])
    writer4.writerow(val3[22170:26604])
    writer4.writerow(val3[26604:31038])
    writer4.writerow(val3[31038:35472])
    writer4.writerow(val3[35472:39906])
    writer4.writerow(val3[39906:44340])
    writer4.writerow(val3[44340:48774])
    writer4.writerow(val3[48774:53208])
    data3 = genfromtxt('C:\\Users\\heera\\Desktop\\class_4_b_division.csv', delimiter=',')
    df3 = pd.DataFrame(data3)
    var3.append(df3.sum(axis = 0)/(len(data3)-1))
    np.savetxt('class_4_var.csv',var3 , fmt='%f', delimiter=",")
    
f_score_val = []
def f_score():
    den = []
    num = []
    var1 = genfromtxt('C:\\Users\\heera\\Desktop\\class_1_var.csv', delimiter=",")
    var2 = genfromtxt('C:\\Users\\heera\\Desktop\\class_2_var.csv', delimiter=",")
    var3 = genfromtxt('C:\\Users\\heera\\Desktop\\class_3_var.csv', delimiter=",")
    var4 = genfromtxt('C:\\Users\\heera\\Desktop\\class_4_var.csv', delimiter=",")
    for i in range(len(var1)):
        den.append(((10*var1[i])+(5*var2[i])+(10*var3[i])+(11*var4[i]))/(36))
    sum1 = genfromtxt('C:\\Users\\heera\\Desktop\\class_1_avg.csv', delimiter=",")
    sum2 = genfromtxt('C:\\Users\\heera\\Desktop\\class_2_avg.csv', delimiter=",")
    sum3 = genfromtxt('C:\\Users\\heera\\Desktop\\class_3_avg.csv', delimiter=",")
    sum4 = genfromtxt('C:\\Users\\heera\\Desktop\\class_4_avg.csv', delimiter=",")
    sum_all = genfromtxt('C:\\Users\\heera\\Desktop\\total_avg.csv', delimiter=",")
    for i in range(len(sum1)):
        num.append(((11*((sum1[i]-sum_all[i])**2))+(6*((sum2[i]-sum_all[i])**2))+(11*((sum3[i]-sum_all[i])**2))+(12*((sum4[i]-sum_all[i])**2)))/(3))
    np.savetxt('denominator.csv',den , fmt='%f', delimiter=",")
    np.savetxt('numerator.csv',num , fmt='%f', delimiter=",")
    for i in range(len(num)):
        if (den[i] == 0 or num[i] == 0):
            f_score_val.append(float('inf'))
        else:
            f_score_val.append((num[i] / den[i]))
    np.savetxt('f_score.csv',f_score_val , fmt='%s', delimiter=",")
    df = pd.DataFrame(f_score_val)
    df.to_csv('f_score_headers.csv', header = False)
    
    
def top_100_features():
    c = []
    feat = []
    dat = []
    lab = []
    dat1 = []
    val = genfromtxt('C:\\Users\\heera\\Desktop\\f_score_headers.csv', delimiter=",")
    data = sorted(val, key = lambda x: x[1], reverse = True)
    df = pd.DataFrame(data)
    df.to_csv('f_score_sorted.csv', header = False)
    val1 = genfromtxt('C:\\Users\\heera\\Desktop\\f_score_sorted.csv', delimiter=",")
    for i in range(len(val1)):
       c.append(int(val1[i][1]))
    feat = (c[0:100])
    f = genfromtxt("C:\\Users\\heera\\Desktop\\train_data.csv", delimiter = ",")
    df2 = pd.DataFrame(f)
    for i in range (0,len(f)):
        lab.append(int(((f[i][4434]))))
    for i in range(len(feat)):
        dat.append((df2[df2.columns[feat[i]-1]]))
    df1 = pd.DataFrame(dat)
    df1.to_csv('f_score_train_data.csv', header = False)
    with open('TrainY_F_Score.csv', 'w', newline = "") as file:
        writer = csv.writer(file)
        for value in lab:
            writer.writerow([value])         
            
    f1 = genfromtxt("C:\\Users\\heera\\Desktop\\TestX.csv", delimiter = ",")
    df3 = pd.DataFrame(f1)
    for j in range(len(feat)):
        dat1.append(df3[df3.columns[feat[j]-1]])
    df4 = pd.DataFrame(dat1)
    df4.to_csv('f_testdata.csv', header = False)    

   
def pick_data_f_score():
    my_train_data = genfromtxt('C:\\Users\\heera\\Desktop\\TrainX_F.csv', delimiter=',')
    for i in range (0,len(my_train_data)):
        TrainX_F.append(my_train_data[i])
        
    my_tst_data = genfromtxt('C:\\Users\\heera\\Desktop\\TestX_F.csv', delimiter=',')
    for i in range (0,len(my_tst_data)):
        TestX_F.append(my_tst_data[i]) 
        
    my_train_label = genfromtxt('C:\\Users\\heera\\Desktop\\TrainY_F_Score.csv', delimiter=',')
    for i in range (0,len(my_train_label)):
        TrainY_F.append((my_train_label[i]))
    df = pd.DataFrame(TrainX_F)
    df1 = pd.DataFrame(TrainY_F)
    horizontalStack = pd.concat([df, df1], axis=1)    
    horizontalStack.to_csv('out.csv', header = False)
    data = genfromtxt('C:\\Users\\heera\\Desktop\\out.csv', delimiter=',')
    a = np.transpose(data)
    b = a[1:]
    train_data = np.transpose(b)
    np.savetxt('f_score_train_data.csv', train_data, fmt='%f', delimiter=",")
    
def predictor(TrainX_F,TrainY_F,TestX): 
    cen=NearestCentroid()    
    SVM=svm.SVC()   
    regr = LinearRegression()
    cen.fit(TrainX_F,TrainY_F)
    SVM.fit(TrainX_F, TrainY_F)
    regr.fit(TrainX_F, TrainY_F)
    print ("Centroid Predicted Labels: ", end = '')
    print (cen.predict(TestX))
    print ("SVM Predicted Labels: ", end = '')
    print (SVM.predict(TestX))
    print ("LR Predicted Labels: ", end = '')
    print (regr.predict(TestX))    

TrainX_F = []  
TrainY_F = [] 
TestX_F = []
train_data = []         
pick_data()
class_separation()
avg_of_each_class_for_a_label()
variance()
f_score()
top_100_features()
my_data = genfromtxt('C:\\Users\\heera\\Desktop\\f_score_train_data.csv', delimiter=',')
z = np.transpose(my_data)
df10 = pd.DataFrame(z[1:100])
np.savetxt('TrainX_F.csv',df10 , fmt='%f', delimiter=",")
my_dat = genfromtxt('C:\\Users\\heera\\Desktop\\f_testdata.csv', delimiter=',')
z1 = np.transpose(my_dat)
df11 = pd.DataFrame(z1[1:100])
np.savetxt('TestX_F.csv',df11 , fmt='%f', delimiter=",")
pick_data_f_score()
print ("KNN Predicted Labels: ", end = '')
print (main())
predictor(TrainX_F,TrainY_F,TestX_F) 