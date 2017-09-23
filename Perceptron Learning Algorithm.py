# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 16:12:20 2016
hw2 : Implementation of the linear classification (PLA) and regression (pseudoinverse) 
algorithms.
@author: pdlalingkar
"""
from __future__ import print_function
import numpy as np
from random import random
import matplotlib.pyplot as plot
import random
import matplotlib


'''
[X, Y] = generateData(N)
This function creates the input and output data arrays. Also the data can be plotted to visualize.
N - number of data points to be created.
*Uncomment the two lines of code to plot the data*
'''
def generateData(N):
    data = np.array(np.random.uniform(-1,1,N*2)).reshape(N,2)
    X = np.ones((N,2+1))
    X[:,1:] = data  
    line = np.array(np.random.uniform(-1,1,4)).reshape(2,2)
    Y = np.sign((line[1,0] - line[0,0]) * (data[:,1] - line[0,1]) - (line[1,1] - line[0,1]) * (data[:,0] - line[0,0]))
#    col = ['red' if x == -1 else 'green' for x in Y]
#    plotOriginal(data,Y,col,line = line)
    retval=(X,Y)        
    return retval

    
'''
plotOriginal(data,Y,col,line = None,w = None)
data - The independent varialbes.
Y - Class label, col - color of the Label Y.
line - If present, randomly choose and plot a line which we use to assign class labels.
w - If present, use the w vector to plot the decision boundary.
 
Function to create a graphical plot of the data. Performs operations to calculate the
equation of the line and plot it.
'''
def plotOriginal(data,Y,col,line = None,w = None): 
    matplotlib.use('Agg')
    if w is None:
        x1, y1, x2, y2 = line[0,0], line[0,1], line[1,0], line[1,1]
        m = (y2-y1)/(x2-x1)
        c = y1 - (m*x1)
        x3, x4 = 1, -1
        y3 = m*x3 + c
        y4 = m*x4 + c
        x = [x1,x2,x3,x4]
        y = [y1,y2,y3,y4]
        plot.title("Generated Input Data") 
        plot.plot(x,y,lw=2)     
   
    if line is None:
        wx1, wy2, wx3, wx4, wy5, wy6 = 0, 0, -1, 1, -1, 1
        wy1 = -w[0]/w[2]
        wx2 = -w[0]/w[1]
        wy3 = (-w[0]-(w[1]*wx3))/w[2]
        wy4 = (-w[0]-(w[1]*wx4))/w[2]
        wx5 = (-w[0]-(w[2]*wy5))/w[1]
        wx6 = (-w[0]-(w[2]*wy6))/w[1]
        wx = [wx1,wx2,wx3,wx4,wx5,wx6]
        wy = [wy1,wy2,wy3,wy4,wy5,wy6] 
        plot.title("W vector")
        plot.plot(wx,wy,'r',lw=2)
    
    plot.scatter(data[:,0],data[:,1],c=col)
    plot.axis([-1.2,1.2,-1.2,1.2])
    plot.grid()
    plot.show()



'''
[w, iters] = pla(X, Y, w0)
X - Input data, independent variables.
Y - Output vector, class labels.
w0 - If present, gives initial set of weights to be used, else 0.
Returns a learned weight vector and number of iterations for the 
perceptron learning algorithm. w0 is the (optional) initial set of weights.
*Uncomment the lines of code to plot the data and view values*
'''
def pla(X, Y, w0=np.zeros(3)):
    indices=[0]
    iters = 0
    w = w0

    while(not(not(indices))):
        indices = []   
        iters=iters+1
        Xt = np.transpose(X)
        prod = np.dot(w,Xt)
        pred = np.array([-1 if x<0 else 1 for x in prod])
#        col = ['red' if x == -1 else 'green' for x in pred]
        mis = ((pred==Y).astype(int))
        indices = [i for i, e in enumerate(mis.tolist()) if e == 0]
        if len(indices)>0:    
            pick = random.choice(indices)   
        else:
#            print("No incorrectly classified samples")
#            print("Final weights: "+str(w[:]))        
#            plotOriginal(X[:,1:3], Y, col, w=w)
            retVal = [w,iters]
            return retVal
            
        X[pick]
        Y[pick]    
        w=w+np.dot(Y[pick],X[pick])  
#        print("Iteration:" + str(iters))
#        print("Incorrect Samples: "+str(len(indices[:])))
#        plotOriginal(X[:,1:3], Y, col, w=w)
        
        

'''
[w] = pseudoinverse(X, Y) returns the learned weight vector for the pseudoinverse algorithm for 
linear regression.
'''
def pseudoinverse(X, Y):
    Xt = np.transpose(X)
    Xdag = np.dot(np.linalg.inv(np.dot(Xt,X)),Xt)
    w = np.dot(Xdag,Y)
    return w


'''
Main Function: This makes a call to all our algorithms and contains experiments
Code for generating log files 
* Change filepath to create output*
'''
def main():
    nVals = [10, 50, 100, 200, 500, 1000]
    
    # Type1: Using initial weights from the pseudo inverse algorithm    
    log = open("C:\\Users\\pdlalingkar\\Documents\\Masters in CS\\Spring 2016\\Machine Learning\\Assignments\\PLAReg\\Logs6\\withPI.txt", "w")
    print("Log File ", file = log)
    print("With Weights ",file = log)
    for N in nVals:
        total = 0
        print("\n  N: "+str(N)+" ", file = log)
        for i in range(0,100):        
            print("    Run: "+str(i+1)+" ", file = log)
            ret = generateData(50)
            X = ret[0]
            Y = ret[1]
            w0 = pseudoinverse(X,Y)
            print("    Weights returned by pseudo-inverse: " + str(w0) + " ", file = log)
            plaWithW0 = pla(X,Y,w0)
            print("    Final Weights: " + str(plaWithW0[0]), file = log)
            print("    Iterations to converge: " + str(plaWithW0[1]), file = log)
            print(" ", file = log)
            total= total + plaWithW0[1]
        print("\n  Average Iterations to converge: " + str(total/100)+" for N: " + str(N), file = log)
    
    log.close()

    # Type2: Initial weights as zeros 
    log = open("C:\\Users\\pdlalingkar\\Documents\\Masters in CS\\Spring 2016\\Machine Learning\\Assignments\\PLAReg\\Logs6\\withoutPI.txt", "w")
    print("Log File ", file = log)
    print("With weights 0 initially",file = log)
    for N in nVals:
        total = 0
        print("\n  N: "+str(N)+" ", file = log)
        for i in range(0,100):        
            print("    Run: "+str(i+1)+" ", file = log)
            ret = generateData(N)
            X = ret[0]
            Y = ret[1]
            plaWithW0 = pla(X,Y)
            print("    Final Weights: " + str(plaWithW0[0]), file = log)
            print("    Iterations to converge: " + str(plaWithW0[1]), file = log)
            print(" ", file = log)
            total= total + plaWithW0[1]
        print("\n  Average Iterations to converge: " + str(total/100)+" for N: " + str(N), file = log)
    
    log.close()

    # Type3: Random weight values between -1 to 1
    log = open("C:\\Users\\pdlalingkar\\Documents\\Masters in CS\\Spring 2016\\Machine Learning\\Assignments\\PLAReg\\Logs6\\withRandom.txt", "w")
    print("Log File ", file = log)
    print("With random weights initially",file = log)
    for N in nVals:
        total = 0
        print("\n  N: "+str(N)+" ", file = log)
        for i in range(0,100):   
            print("    Run: "+str(i+1)+" ", file = log)
            ret = generateData(N)
            X = ret[0]
            Y = ret[1]
            w0 = np.array(np.random.uniform(-1,1,3)).reshape(3,)
            print("    Initial Random Weights: " + str(w0) + " ", file = log)
            plaWithW0 = pla(X,Y,w0)
            print("    Final Weights: " + str(plaWithW0[0]), file = log)
            print("    Iterations to converge: " + str(plaWithW0[1]), file = log)
            print(" ", file = log)
            total= total + plaWithW0[1]
        print("\n  Average Iterations to converge: " + str(total/100)+" for N: " + str(N), file = log)
        
    log.close()


if __name__ == "__main__":
    main()



