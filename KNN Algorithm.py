# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 02:49:35 2016

Name: Prasanna Lalingkar
ID: 800936073

Contains the code for the hw1. 

@author: pdlalingkar
"""
import numpy as np
from scipy import stats
from collections import Counter
import timeit
import random
from __future__ import print_function
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from random import randint

'''
KNN Algorithm.
testY = testknn(trainX, trainY, testX, k)
where trainX is a (nTrain *D) data matrix, testX is a (nTest * D) data matrix, trainY is a (nTrain * 1) label vector, and testY is a (nTest * 1) label vector, and k is the number of nearest neighbors for classification.
'''

def testknn(trainX, trainY, testX, k):
    testY=[]                                                    # initialize
    for i in range(0,(len(testX))):                             # 1 test sample at a time
        if(len(trainX)==1):                                     # only one training sample
            testY=np.array([trainY,]*(len(testX)))              # 1D array for all test labels
            break
        new = np.array([testX[i,],]*(len(trainX)))              # 2D array of 1 test sample at a time              
        sqDiff = np.sqrt(np.sum(np.square(trainX-new),axis=1))  # Euclidean distance of test point from all training points
        indList = list(np.argpartition(sqDiff,k))[:k]           # Choosing index of nearest K points
        listOfLabels = list(trainY[indList])                    # Labels of the nearest K points
        if k != 1:                                              
            forMode = Counter(listOfLabels)                     # Take mode of labels
            testY.append(forMode.most_common(1)[0][0])          # Assign to the test point
            del forMode
        else:
            testY.append(listOfLabels[0])                       # Assign the label to the test point 
        del new                                                 # A good practice, had read this somewhere
        del sqDiff
        del indList
        del listOfLabels
           
    return testY

ans = testknn(trainX, trainY, testX, k)                         # Sample function call



'''
Condensed 1-NN function
condensedIdx = condensedata(trainX, trainY)
where condensedIdx is a vector of indicies of the condensed training set.
'''

def condensedata(trainX, trainY):

    condensedIdx=[]                                             # Initialize 3 arrays for condensed sets
    condensedX=np.empty((0,16),int)                             # For label, index and predictors
    condensedY=[]
    index=randint(0,int(len(trainX)-1))                         # Choose any number at random from trainX
    condensedX=np.append(condensedX,np.array([trainX[index,]]),axis=0)
    condensedY=trainY[index,]
    condensedIdx=np.append(condensedIdx,index)                  # Assign values to 3 condensed sets
    indices=[0]                                                 # Dummy initialization to enter the loop
    
    while not(not(indices)):                                    # Till there are incorrectly classified elements
        if(len(condensedX) == len(trainX)):                     # Break if condensed set contains all of training set (worst case)
            break
        indices = []                                            # No incorrectly classified samples yet
        temp = testknn(condensedX,condensedY,trainX, 1)         # Classify the training set using condensed sets for training
        incorrect=((temp==trainY).astype(int)==0)               # Identify incorrectly classified data points
        indices=[i for i, elem in enumerate(incorrect.tolist(), 0) if elem]   # Get indices for incorrectly classified points
        if len(indices)>0:        
            pick = random.sample(indices,1)[0]                  # Pick randomly any incorrectly classified point
        condensedIdx = np.append(condensedIdx,pick)
        condensedX = np.append(condensedX,np.array([trainX[pick,]]),axis=0)
        condensedY = np.append(condensedY,trainY[pick,])        # Update all 3 condensed set arrays and repeat process
        del incorrect
        del temp

    condensedIdx = condensedIdx[:len(condensedIdx)-1]           
    return condensedIdx

condensedIdx = condensedata(trainX, trainY)                     # Sample function call



''' Helper code to read input file and create data sets '''

# Change input path here
dataFile = "C:\Users\pdlalingkar\Documents\Masters in CS\Spring 2016\Machine Learning\Data.csv";
data = np.loadtxt(dataFile, delimiter=',', usecols=range(1,17))
labels = np.loadtxt(dataFile, delimiter=',', usecols=range(0,1), dtype=str)

nTrain = 15000
nTest = 5000
D = 16

# Testing parameters were used to test the basic implementations
trainX = data[0:10,]
trainY = labels[0:10,]
testX = data[10:15,]
testYActual = labels[10:15,]
k=1

# Full parameters
trainX = data[0:nTrain,]
trainY = labels[0:nTrain,]
testX = data[nTrain:,]
testYActual = labels[nTrain:,]
k=9

# Random Sampling as per required N
randomList = random.sample(xrange(15000), 100)
trainX = data[randomList,]
trainY = labels[randomList,]
testX = data[nTrain:,]
testYActual = labels[nTrain:,]
kvals = [9,7,5,3,1]

# Code for generating log files for testknn()
# Change input dirrectory and filepath
log = open("C:\\Users\\pdlalingkar\\Documents\\Masters in CS\\Spring 2016\\Machine Learning\\KNN\\Logs\\val_100_5000.txt", "w")
print("test", file = log)

for k in kvals:
    start = timeit.default_timer()
    ans = testknn(trainX, trainY, testX, k)
    stop = timeit.default_timer()
    print("K: "+str(k)+" .. nTrain: "+str(len(trainX))+" .. nTest "+str(len(testX))+" ", file = log)
    print("Execution time: " + str(round((stop - start),2)) + " seconds", file = log)
    print("Accuracy Score: " + str(accuracy_score(testYActual, ans)), file = log)
    print("Precision: " + str(precision_score(testYActual, ans, average='weighted')), file = log)
    print("Recall: " + str(recall_score(testYActual, ans, average='weighted')), file = log)
    print((confusion_matrix(testYActual, ans)), file = log)
    print(classification_report(testYActual, ans), file = log)
    print("...........................",file = log)
    print(" ",file = log)

log.close()



   

# Code for generating log files for condensedata()
# Change input dirrectory and filepath
log = open("C:\\Users\\pdlalingkar\\Documents\\Masters in CS\\Spring 2016\\Machine Learning\\KNN\\LogsC\\new_15000_5000.txt", "w")
print("test", file = log)
start = timeit.default_timer()
print("K: "+str(k)+" .. Training Set N: "+str(len(trainX))+" .. Test Set N "+str(len(testX))+" ", file = log)    
condensedIdx = condensedata(trainX, trainY)
condensed = timeit.default_timer()    
print("Condensed Set creation time: " + str(round((condensed - start),2)) + " seconds", file = log)
inList = list([int(float(x)) for x in condensedIdx.tolist()])
trainY[inList,]
trainX[inList,]    
print("Condensed Set N: " + str(len(inList)) + " ", file = log)


for k in kvals:
    print("Nth time: " + str(k) + " ", file = log)
    condensed = timeit.default_timer()
    ans = testknn(trainX, trainY, testX, k)
    stop = timeit.default_timer()
    print("KNN Execution time: " + str(round((stop - condensed),2)) + " seconds", file = log)
    print("Accuracy Score: " + str(accuracy_score(testYActual, ans)), file = log)
    print("Precision: " + str(precision_score(testYActual, ans, average='weighted')), file = log)
    print("Recall: " + str(recall_score(testYActual, ans, average='weighted')), file = log)
    print((confusion_matrix(testYActual, ans)), file = log)
    print(classification_report(testYActual, ans), file = log)
    print("...........................",file = log)
    print(" ",file = log)


log.close()
