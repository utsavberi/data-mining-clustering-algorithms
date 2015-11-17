from __future__ import division
import numpy as np

__author__ = 'utsav'

def createIncidenceMatrix(label):
    n = len(label)
    im = np.zeros(n*n).reshape(n,n)
    for i in range (0,n):
        for j in range (0,n):
            if label[i] == label[j]:
                im[i][j] = 1
    return im

def calculateJaccardCoeff(trueLabel,predLabel):
    n = len(trueLabel)
    trueLabelIncidenceMatrix = createIncidenceMatrix(trueLabel)
    predLabelIncidenceMatrix = createIncidenceMatrix(predLabel)
    cmp = [True if xv==1 and yv==1  else False for (xv,yv) in zip(trueLabelIncidenceMatrix.ravel(),predLabelIncidenceMatrix.ravel())]
    cmp = np.reshape(cmp,(n,n))
    ss = np.sum(cmp)
    sdds = np.sum(trueLabelIncidenceMatrix!=predLabelIncidenceMatrix)
    return ss/(ss+sdds)

def computeEuclideanDistance(X):
    XX = np.sum(X * X, axis=1)[:, np.newaxis]
    distances =np.dot(X, X.T)
    distances *= -2
    distances += XX
    distances += XX.T
    distances.flat[::distances.shape[0] + 1] = 0.0
    return np.nan_to_num(np.sqrt(distances))

import math
def computeCorrelation(X,labels):
    D = computeEuclideanDistance(X)
    meanD = np.mean(D)
    # print "meanD", meanD
    C = createIncidenceMatrix(labels)
    # print C
    meanC = np.mean(C)
    # print "meanC", meanC
    num = (D - meanD)*(C - meanC)

    num = np.sum(num)
    den1 = (D - meanD)*(D - meanD)
    den1 = np.sum(den1)
    den1 = math.sqrt(den1)
    den2 = (C - meanC)*(C - meanC)
    den2 = np.sum(den2)
    den2 = math.sqrt(den2)
    # print "numeratr ", num
    return num/(den1*den2)

# X = np.array([[0,2,3,4,1],[5,0,6,7,8],[1,2,0,3,4],[1,2,3,0,4],[1,2,5,6,0]])
# I = np.array([[0,1,0,0,0],[1,0,0,0,0],[0,0,0,1,0],[0,0,1,0,0],[0,0,0,0,0]])
# print computeCorrelation(X,I)