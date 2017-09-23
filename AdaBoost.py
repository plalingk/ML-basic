# -*- coding: utf-8 -*-
"""
Created on Thu Apr 07 12:53:59 2016

@author: pdlalingkar
"""

#from __future__ import print_function
import numpy as np
from random import random
import random
import math


''' 
val = dataTrainTest(data,labels,operation)
Takes as input the data, corresponding labels and a character input 'operation'
(T = Train/Test split, V = Train/Validation split) and divides the data accordingly.
returns a tuple of values accordingly.
'''
def dataTrainTest(data,labels,operation='t'):
    if operation == 't' or operation == 'T':
        nTrain = int(0.75*labels.size)
        nTest = labels.size-nTrain
    if operation == 'v' or operation == 'V':
        nTest = labels.size/9
        nTrain = labels.size-nTest
    # Generating indices for splits
    indexAll = np.array(range(0,labels.size))
    indexTrain = np.random.choice(indexAll,nTrain,replace=False)    
    indexTest = np.setdiff1d(indexAll,indexTrain) 
    # Subsetting data
    XTrain = np.array(data[indexTrain])
    YTrain = np.array(labels[indexTrain])
    XTest = np.array(data[indexTest])
    YTest = np.array(labels[indexTest])
    
    retVal = (XTrain,YTrain,XTest,YTest)
    return retVal


'''
[w] = pseudoinverse(X, Y) returns the learned weight vector for the pseudoinverse algorithm for 
linear regression.
'''
def pseudoinverse(XTrain, YTrain):
    # Adding intercepts
    X = np.ones((YTrain.size,16+1))   
    X[:,1:] = XTrain
    XTrain = X
    # Computing weight vector
    Xt = np.transpose(XTrain)
    Xdag = np.dot(np.linalg.inv(np.dot(Xt,XTrain)),Xt)
    pred = np.array([-1 if x=='republican' else 1 for x in YTrain])
    YTrain = pred    
    w = np.dot(Xdag,YTrain)
    return w


'''
[w, iters] = pla(X, Y, w0)
X - Input data, independent variables.
Y - Output vector, class labels.
w0 - If present, gives initial set of weights to be used, else 0.
Returns a learned weight vector and number of iterations for the 
perceptron learning algorithm. w0 is the (optional) initial set of weights.
*Uncomment the lines of code to plot the data and view values*
'''
def pla(XTrain, YTrain, w0=np.zeros(17)):
    # Adding intercepts
    X = np.ones((YTrain.size,16+1))
    X[:,1:] = XTrain
    XTrain = X
    # converting YTrain to -1 and +1
    pred = np.array([-1 if x=='republican' else 1 for x in YTrain])
    YTrain = pred
    indices=[0]
    iters = 0
    w = w0

    while(iters<500):
        indices = []   
        iters=iters+1
        Xt = np.transpose(XTrain)
        prod = np.dot(w,Xt)
        pred = np.array([-1 if x<0 else 1 for x in prod])
        mis = ((pred==YTrain).astype(int))
        indices = [i for i, e in enumerate(mis.tolist()) if e == 0]
        if len(indices)>0:    
            pick = random.choice(indices)   
        else:
            retVal = [w,iters]
            return retVal
        if(iters==1):
            minError = len(indices)
            weights = w
        if(len(indices) < minError):
            minError = len(indices)
            weights = w
            
        XTrain[pick]
        YTrain[pick]    
        w=w+np.dot(YTrain[pick],XTrain[pick])
    retVal = [weights,minError]
    return retVal
        
'''
model = adaTrain(XTrain, YTrain, version) which takes as input:
XTrain, the training examples, a (N, D) numpy ndarray where N is the number of training examples and D is the dimensionality
YTrain, a 1-D array of labels
version, a string which can be 'stump' 'perceptron' or 'both'
This method returns a model object containing the parameters of the trained model.
'''

def adaTrain(XTrain, YTrain, version):    
    if(version=='perceptron'):
        splitData = dataTrainTest(XTrain,YTrain,operation='v') 
        TrainX = splitData[0]
        TrainY = splitData[1]
        ValidationX = splitData[2]
        ValidationY = splitData[3]

        # Intial Distribution D    
        D = np.repeat(float(1)/TrainY.size,TrainY.size)   
        # Adding intercepts to Validation
        X = np.ones((ValidationY.size,16+1))
        X[:,1:] = ValidationX
        ValidationX = X
        # 1st pocket with initial guessed w0
        w0 = pseudoinverse(TrainX,TrainY)
        ans = pla(TrainX, TrainY,w0)
        pocketWeights = ans[0]
        # Adding intercepts to Train
        X = np.ones((TrainY.size,16+1))
        X[:,1:] = TrainX
        TrainX1 = X    
    
        # First weak learner
        prod = np.dot(TrainX1,pocketWeights)
        pred1 = np.array([-1 if x<0 else 1 for x in prod])
        pred = np.array(['republican' if x<0 else 'democrat' for x in prod])
        incorrect = ((pred==TrainY).astype(int))
        indices = [i for i, e in enumerate(incorrect.tolist()) if e == 0]       
        np.put(prod,np.setdiff1d(np.array(range(0,prod.size)),np.array(indices)),0)     
        np.put(incorrect,indices,5)
        np.put(incorrect,np.setdiff1d(np.array(range(0,prod.size)),np.array(indices)),0)    
        np.put(incorrect,indices,1)
        weightedError = sum(D*incorrect)    
        # Calculating alpha and z, avoid divide by zero error
        alpha = 0.5 * math.log(((1-weightedError)/(weightedError)))
        z = 2*math.pow((weightedError*(1-weightedError)),0.5)    
        if(z==0):
            z=1
        # Calculating new Distribution
        TrainYconverted = np.array([-1 if x=='republican' else 1 for x in TrainY])    
        Dt = (D*np.exp(-alpha*TrainYconverted*pred1))/z    
    
        # Validation error
        prod = np.dot(ValidationX,pocketWeights)
        pred = np.array(['republican' if x<0 else 'democrat' for x in prod])
        incorrect = ((pred==ValidationY).astype(int))
        indices = [i for i, e in enumerate(incorrect.tolist()) if e == 0]
        validationError = float(len(indices))/pred.size
        #print("Validation Error 1st and set to min error: "+str(validationError))
        prevValidationError = validationError 
        t=0
    
        model = []
        model.insert(t,pocketWeights.tolist())
        Alpha = []
        Alpha.insert(t,alpha)
    
        #print("\n starting for loop")
        while(prevValidationError>=validationError and t<=10):
            #print(" ")
            #print("new iteration ")
            prevValidationError = validationError
            t=t+1
            indicesAll = np.array(range(0,Dt.size))
            indicesTrain = np.random.choice(TrainY.size, TrainY.size, p=Dt)
            
            TrainXT = np.array(TrainX[indicesTrain])
            TrainYT = np.array(TrainY[indicesTrain])
            w0 = pseudoinverse(TrainXT,TrainYT)
            ans = pla(TrainXT, TrainYT,w0)
            pocketWeights = ans[0]
            
            prod = np.dot(TrainX1,pocketWeights)
            pred1 = np.array([-1 if x<0 else 1 for x in prod])
            pred = np.array(['republican' if x<0 else 'democrat' for x in prod])
            incorrect = ((pred==TrainY).astype(int))
            indices = [i for i, e in enumerate(incorrect.tolist()) if e == 0]       
            np.put(prod,np.setdiff1d(np.array(range(0,prod.size)),np.array(indices)),0)     

            np.put(incorrect,indices,5)
            np.put(incorrect,np.setdiff1d(np.array(range(0,prod.size)),np.array(indices)),0)    
            np.put(incorrect,indices,1)
            weightedError = sum(Dt*incorrect)    
            #print("WeightedError: "+str(weightedError))

            alpha = 0.5 * math.log(((1-weightedError)/(weightedError)))
            z = 2*math.pow((weightedError*(1-weightedError)),0.5)    
            if(z==0):
                z=1
            #print("Alpha: "+str(alpha))
            #print("Z: "+str(z))
            TrainYconverted = np.array([-1 if x=='republican' else 1 for x in TrainY])    
            Dt = (Dt*np.exp(-alpha*TrainYconverted*pred1))/z
            dist = np.sum(Dt)
            #print("Distribution: "+str(dist))        
            #print("Distribution: "+str(Dt))
        
            #print("Validation Error: "+str(validationError))
            #print(pocketWeights)
            model.insert(t,pocketWeights.tolist())
            Alpha.insert(t,alpha)
            #print("T: "+str(t))   
            # Validation error
            currentModel = np.transpose(model)
            prod = np.dot(ValidationX,currentModel)
            prod = np.sum(prod,axis=1)
            pred = np.array(['republican' if x<0 else 'democrat' for x in prod])
            incorrect = ((pred==ValidationY).astype(int))
            indices = [i for i, e in enumerate(incorrect.tolist()) if e == 0]
            validationError = float(len(indices))/pred.size
            #print("Validation Error: "+str(validationError))
            #print("prev Validation Error: "+str(prevValidationError))
        
        return (version,model,Alpha)
        
        
        
    if(version=='stump'):
        # Validation data creation
        splitData = dataTrainTest(XTrain,YTrain,operation='v') 
        TrainX = splitData[0]
        TrainY = splitData[1]
        ValidationX = splitData[2]
        ValidationY = splitData[3]

        # Intial Distribution D    
        D = np.repeat(float(1)/TrainY.size,TrainY.size)  
        # First weak learner
        splitIndex = giniBestSplitAttribute(TrainX,TrainY)   
        predictedOne = decisionStump(TrainX, TrainY, splitIndex)    
        model = []
        splitIndices = []
        t = 0
        if(predictedOne<0):
            model.insert(t,list([1,-1,-1,1]))
        else:
            model.insert(t,list([1,1,-1,-1]))

        splitIndices.insert(t,splitIndex)        
        pred = np.array([model[t][1] if x==model[t][0] else model[t][3] for x in TrainX[:,splitIndex].astype(int)])
        pred1 = np.array(['republican' if x ==-1 else 'democrat' for x in pred])
        
        incorrect = ((pred1==TrainY).astype(int))
        indices = [i for i, e in enumerate(incorrect.tolist()) if e == 0]       
       
        np.put(incorrect,indices,5)
        np.put(incorrect,np.setdiff1d(np.array(range(0,pred1.size)),np.array(indices)),0)    
        np.put(incorrect,indices,1)
        len(incorrect)        
        
        weightedError = sum(D*incorrect)    
        # Calculating alpha and z
        alpha = 0.5 * math.log(((1-weightedError)/(weightedError)))
        z = 2*math.pow((weightedError*(1-weightedError)),0.5)        
        if(z==0):
            z=1
        # Calculating new Distribution
        TrainYconverted = np.array([-1 if x=='republican' else 1 for x in TrainY]) 
        pred2 = np.array([-1 if x=='republican' else 1 for x in pred1])
        Dt = (D*np.exp(-alpha*TrainYconverted*pred2))/z    
        sum(Dt)
       # Validation error
        pred = np.array([model[t][1] if x==model[t][0] else model[t][3] for x in ValidationX[:,splitIndex].astype(int)])
        pred1 = np.array(['republican' if x ==-1 else 'democrat' for x in pred])        
        incorrect = ((pred1==ValidationY).astype(int))
        indices = [i for i, e in enumerate(incorrect.tolist()) if e == 0]        
        
        validationError = float(len(indices))/pred.size
        #print("Validation Error: "+str(validationError))       
        t=0
        prevValidationError = validationError
    
        Alpha = []
        Alpha.insert(t,alpha)
        #print("\n starting for loop")
        while(prevValidationError >= validationError and t <= 10):
            #print(" ")
            prevValidationError = validationError
            t=t+1
            indicesAll = np.array(range(0,Dt.size))
            indicesTrain = np.random.choice(TrainY.size, TrainY.size, p=Dt)
            
            TrainXT = np.array(TrainX[indicesTrain])
            TrainYT = np.array(TrainY[indicesTrain])
            # Predicting on data
            splitIndex = giniBestSplitAttribute(TrainXT,TrainYT)   
            predictedOne = decisionStump(TrainXT, TrainYT, splitIndex) 
            if(predictedOne<0):
                model.insert(t,list([1,-1,-1,1]))
            else:
                model.insert(t,list([1,1,-1,-1]))
           
            splitIndices.insert(t,splitIndex)     
            pred = np.array([model[t][1] if x==model[t][0] else model[t][3] for x in TrainX[:,splitIndex].astype(int)])
            pred1 = np.array(['republican' if x ==-1 else 'democrat' for x in pred])
        
            pred1.shape
            TrainY.shape
            incorrect.shape
            incorrect = ((pred1==TrainY).astype(int))
            indices = [i for i, e in enumerate(incorrect.tolist()) if e == 0]       
       
            np.put(incorrect,indices,5)
            np.put(incorrect,np.setdiff1d(np.array(range(0,pred1.size)),np.array(indices)),0)    
            np.put(incorrect,indices,1)
        
            weightedError = sum(Dt*incorrect)    
            # Calculating alpha and z
            alpha = 0.5 * math.log(((1-weightedError)/(weightedError)))
            z = 2*math.pow((weightedError*(1-weightedError)),0.5)     
            if(z==0):
                z=1
            Alpha.insert(t,alpha)
            
            # Calculating new Distribution
            TrainYconverted = np.array([-1 if x=='republican' else 1 for x in TrainY]) 
            pred2 = np.array([-1 if x=='republican' else 1 for x in pred1])
            Dt = (Dt*np.exp(-alpha*TrainYconverted*pred))/z
            #print(sum(Dt))
           # Validation error
            predList = []
            for j in range(0,len(model)):
                pred = np.array([model[j][1] if x==model[j][0] else model[j][3] for x in ValidationX[:,splitIndices[j]].astype(int)])
                predList.insert(j,pred.tolist())
            arr = np.array(predList)
            arr1 = arr*np.array(Alpha)[:,None]
            prod = np.sum(arr1, axis = 0)
            pred1 = np.array(['republican' if x <0 else 'democrat' for x in prod])            
            incorrect = ((pred1==ValidationY).astype(int))
            indices = [i for i, e in enumerate(incorrect.tolist()) if e == 0]        
            validationError = float(len(indices))/pred.size
            #print("Validation Error: "+str(validationError))
            #print("prev Validation Error: "+str(prevValidationError))
            #print("T: "+str(t))           
        
        return(version, model, Alpha, splitIndices)
        
        
    if(version=='both'):
        # Validation data creation
        splitData = dataTrainTest(XTrain,YTrain,operation='v') 
        TrainX = splitData[0]
        TrainY = splitData[1]
        ValidationX = splitData[2]
        ValidationY = splitData[3]
        # Adding intercepts to Validation
        X = np.ones((ValidationY.size,16+1))
        X.shape    
        X[:,1:] = ValidationX
        ValidationX1 = X
        # Adding intercepts to Train
        X = np.ones((TrainY.size,16+1))
        X.shape    
        X[:,1:] = TrainX
        TrainX1 = X    
        # Intial Distribution D    
        Dt = np.repeat(float(1)/TrainY.size,TrainY.size)        
        sT, pT, t = 0, 0, 0
        modelS = []
        modelP = []
        alphaS = []
        alphaP = []
        splitIndices = []
        
        for t in range(0,10):
            indicesAll = np.array(range(0,Dt.size))
            indicesTrain = np.random.choice(TrainY.size, TrainY.size, p=Dt)
            TrainXT = np.array(TrainX[indicesTrain])
            TrainYT = np.array(TrainY[indicesTrain])
            
            if(random.randint(0,1)==0):
                # First weak learner
                splitIndex = giniBestSplitAttribute(TrainXT,TrainYT)   
                predictedOne = decisionStump(TrainXT, TrainYT, splitIndex)    
                if(predictedOne<0):
                    modelS.insert(sT,list([1,-1,-1,1]))
                else:
                    modelS.insert(sT,list([1,1,-1,-1]))

                splitIndices.insert(sT,splitIndex)        
                pred = np.array([modelS[sT][1] if x==modelS[sT][0] else modelS[sT][3] for x in TrainX[:,splitIndex].astype(int)])
                pred1 = np.array(['republican' if x ==-1 else 'democrat' for x in pred])
                incorrect = ((pred1==TrainY).astype(int))
                indices = [i for i, e in enumerate(incorrect.tolist()) if e == 0]
                np.put(incorrect,indices,5)
                np.put(incorrect,np.setdiff1d(np.array(range(0,pred1.size)),np.array(indices)),0)    
                np.put(incorrect,indices,1)
                weightedError = sum(Dt*incorrect)    
                # Calculating alpha and z
                alpha = 0.5 * math.log(((1-weightedError)/(weightedError)))
                z = 2*math.pow((weightedError*(1-weightedError)),0.5) 
                if(z==0):
                    z=1
                alphaS.insert(sT,alpha)
                # Calculating new Distribution
                TrainYconverted = np.array([-1 if x=='republican' else 1 for x in TrainY]) 
                pred2 = np.array([-1 if x=='republican' else 1 for x in pred1])
                Dt = (Dt*np.exp(-alpha*TrainYconverted*pred2))/z    
                #print(Dt)
                # Validation error
                pred = np.array([modelS[sT][1] if x==modelS[sT][0] else modelS[sT][3] for x in ValidationX[:,splitIndex].astype(int)])
                pred1 = np.array(['republican' if x ==-1 else 'democrat' for x in pred])        
                incorrect = ((pred1==ValidationY).astype(int))
                indices = [i for i, e in enumerate(incorrect.tolist()) if e == 0]        
                validationError = float(len(indices))/pred.size 
                #print("stump")
                #print("Validation Error: "+str(validationError))                  
                sT=sT+1
                
                
            else:          
                # 
                w0 = pseudoinverse(TrainXT,TrainYT)
                ans = pla(TrainXT, TrainYT,w0)
                pocketWeights = ans[0]

                prod = np.dot(TrainX1,pocketWeights)
                pred1 = np.array([-1 if x<0 else 1 for x in prod])
                pred = np.array(['republican' if x<0 else 'democrat' for x in prod])
                incorrect = ((pred==TrainY).astype(int))
                indices = [i for i, e in enumerate(incorrect.tolist()) if e == 0]       
                np.put(prod,np.setdiff1d(np.array(range(0,prod.size)),np.array(indices)),0)     
                np.put(incorrect,indices,5)
                np.put(incorrect,np.setdiff1d(np.array(range(0,prod.size)),np.array(indices)),0)    
                np.put(incorrect,indices,1)
                weightedError = sum(Dt*incorrect)    
                # Calculating alpha and z
                alpha = 0.5 * math.log(((1-weightedError)/(weightedError)))
                z = 2*math.pow((weightedError*(1-weightedError)),0.5)   
                if(z==0):
                    z=1
                #print("Alpha: "+str(alpha))
                #print("Z: "+str(z))    
                # Calculating new Distribution
                TrainYconverted = np.array([-1 if x=='republican' else 1 for x in TrainY])    
                Dt = (Dt*np.exp(-alpha*TrainYconverted*pred1))/z    
                #print(Dt)
                # Validation error
                prod = np.dot(ValidationX1,pocketWeights)
                pred = np.array(['republican' if x<0 else 'democrat' for x in prod])
                incorrect = ((pred==ValidationY).astype(int))
                indices = [i for i, e in enumerate(incorrect.tolist()) if e == 0]
                validationError = float(len(indices))/pred.size
                #print("perceptron")
                #print("Validation Error: "+str(validationError))                  
                modelP.insert(pT,pocketWeights.tolist())
                alphaP.insert(pT,alpha)
                pT=pT+1
            
        return(version,modelS,alphaS,modelP,alphaP,splitIndices)


 
'''
Calculate gini index and the attribute for best split for training data and its labels
'''

def giniBestSplitAttribute(XTrain, YTrain):
        # Calculate gini and best split
        pred = np.array([0 if x=='republican' else 1 for x in YTrain])
        GiniParent = np.prod((np.bincount(pred))/float(pred.size))        
        
        classLabel = np.array([0 if x=='republican' else 1 for x in YTrain])
        i=0
        GiniChild = []
        for i in range(0,len(XTrain[0])):
            pred = np.array([0 if x=='-1' else 1 for x in XTrain[:,i]])
            counts = np.bincount(pred)            
            minusOne = counts[0] 
            One = counts[1]
            minusOneR = counts[0]/float(len(pred)) 
            OneR = counts[1]/float(len(pred))
            minusOnerep = float(((pred==0) & (classLabel==0)).sum())
            minusOnedem = float(((pred==0) & (classLabel==1)).sum())
            Onerep = float(((pred==1) & (classLabel==0)).sum())
            Onedem = float(((pred==1) & (classLabel==1)).sum())
            GiniChild.insert(i, (minusOneR*((minusOnerep*minusOnedem)/(math.pow(minusOne,2)))) + (OneR*((Onerep*Onedem)/(math.pow(One,2)))) )
            
        return(GiniChild.index(min(GiniChild)))
      
      
'''
Build a decision Stump using the Training data, their labels and the split index for the data
'''

def decisionStump(XTrain, YTrain, splitIndex):
    pred =np.copy(XTrain[:,splitIndex])
    pred[pred == '-1'] = 0
    pred[pred == '1'] = 1     
    a = pred.astype(int)
    one = np.nonzero(a)
    minusOne = np.setdiff1d(np.array(range(0,pred.size)),np.array(one))
    Y = np.array([-1 if x=='republican' else 1 for x in YTrain])

    return(sum(Y[one]))

        
'''
YTest = adaPredict(model, XTest) which takes as input:
the model object returned by adaTrain
XTest, the testing examples, a (N, D) numpy ndarray where N is the number of testing examples and D is the dimensionality
This method returns a 1-D array of predicted labels corresponding to the provided test examples
'''     

def adaPredict(model, XTest):

    
    if(model[0]=='perceptron'):    
        # Adding intercepts to XTest
        X = np.ones((XTest.size/16,16+1))
        X[:,1:] = XTest
        XTest = X
        #Predicting on test data
        weights = np.array(model[1])
        alpha = np.array(model[2])
        wT = np.transpose(weights)
        prod = np.dot(np.dot(XTest,wT),alpha)
        pred = np.array(['republican' if x<0 else 'democrat' for x in prod])
        return pred
        
    if(model[0]=='stump'):
        models = np.array(model[1])
        alpha = np.array(model[2])
        splitIndex = np.array(model[3])
        # Predicting on test data
        t=0
        predList = []
        for t in range(0,len(splitIndex)):
            pred = np.array([models[t][1] if x==models[t][0] else models[t][3] for x in XTest[:,splitIndex[t]].astype(int)])
            predList.insert(t,pred.tolist())
        arr = np.array(predList)
        arr1 = arr*alpha[:,None]
        prod = np.sum(arr1, axis = 0)
        pred1 = np.array(['republican' if x <0 else 'democrat' for x in prod])
        
        return pred1


    if(model[0]=='both'):
        modelS = np.array(model[1])
        alphaS = np.array(model[2])
        modelP = np.array(model[3])
        alphaP = np.array(model[4])
        splitIndex = np.array(model[5])
        
        # Predicting on test data
        t=0
        predList = []
        for t in range(0,len(splitIndex)):
            pred = np.array([modelS[t][1] if x==modelS[t][0] else modelS[t][3] for x in XTest[:,splitIndex[t]].astype(int)])
            predList.insert(t,pred.tolist())
        arr = np.array(predList)
        arr1 = arr*alphaS[:,None]
        prod = np.sum(arr1, axis = 0)
        # Adding intercepts to XTest
        X = np.ones((XTest.size/16,16+1))
        X[:,1:] = XTest
        XTest1 = X
        XTest1.shape
        #Predicting on test data
        wT = np.transpose(modelP)
        prod1 = np.dot(np.dot(XTest1,wT),alphaP)        
        combined = np.add(prod,prod1)
        prediction = np.array(['republican' if x <0 else 'democrat' for x in combined])        
        
        return prediction        
    


'''
Main Function: This makes a call to all our algorithms.

'''

def main():
    dataFile = "C:\Users\pdlalingkar\Documents\Masters in CS\Spring 2016\Machine Learning\Assignments\\adaboost\data\data2oneminus1.csv";
    data = np.loadtxt(dataFile, delimiter=',', usecols=range(1,17),dtype=str)
    labels = np.loadtxt(dataFile, delimiter=',', usecols=range(0,1), dtype=str)
    splitData = dataTrainTest(data,labels) 
    XTrain = splitData[0]
    YTrain = splitData[1]
    XTest = splitData[2]
    YTest = splitData[3]
 
   # Running perceptron
    print("Perceptron")
    splitData = dataTrainTest(data,labels) 
    XTrain = splitData[0]
    YTrain = splitData[1]
    XTest = splitData[2]
    YTest = splitData[3]
    version = 'perceptron'
    val = adaTrain(XTrain, YTrain, version)
    #print(len(val[2]))
    pred = adaPredict(val, XTest)
    incorrect = ((pred==YTest).astype(int))
    indices = [i for i, e in enumerate(incorrect.tolist()) if e == 0]
    error = float(len(indices))/float(YTest.size)
    Accuracy = (1-error)*100
    print("Execution Accuracy: "+str(Accuracy)+" %")

    print("Stump")
    splitData = dataTrainTest(data,labels) 
    XTrain = splitData[0]
    YTrain = splitData[1]
    XTest = splitData[2]
    YTest = splitData[3]
    version = 'stump'
    val = adaTrain(XTrain, YTrain, version)
    #print(len(val[2]))
    pred = adaPredict(val, XTest)
    incorrect = ((pred==YTest).astype(int))
    indices = [i for i, e in enumerate(incorrect.tolist()) if e == 0]
    error = float(len(indices))/float(YTest.size)
    Accuracy = (1-error)*100
    print("Execution Accuracy: "+str(Accuracy)+" %")

    print("Both")
    splitData = dataTrainTest(data,labels) 
    XTrain = splitData[0]
    YTrain = splitData[1]
    XTest = splitData[2]
    YTest = splitData[3]
    version = 'both'
    val = adaTrain(XTrain, YTrain, version)
    pred = adaPredict(val, XTest)
    incorrect = ((pred==YTest).astype(int))
    indices = [i for i, e in enumerate(incorrect.tolist()) if e == 0]
    error = float(len(indices))/float(YTest.size)
    Accuracy = (1-error)*100
    print("Execution Accuracy: "+str(Accuracy)+" %")
        


  

if __name__ == "__main__":
    main()


