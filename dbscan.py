from __future__ import division
__author__ = 'utsav'
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.mlab import PCA as mlabPCA

from metrics import computeEuclideanDistance
__author__ = 'utsav'
class DBScan:

    #    return all points within P's eps-neighborhood (including P)
    def regionQuery(self,P,eps,neighborhoods):
        return neighborhoods[P]

    def plotKnn(self,X,k):
        D = computeEuclideanDistance(X);
        sorted = np.sort(D)
        s = sorted[:,1:k]
        kDist =  np.sort(np.mean(s[:,1:4],axis = 1))
        objectId = np.arange(0,len(kDist))
        plt.plot(objectId,kDist)
        plt.show()




    def fit(self,data,eps,MinPts):
        X = data
        n = X.shape[0]
        dist = computeEuclideanDistance(X)#distancematrix
        neighborhoods = [np.where(x<=eps)[0] for x in dist]#find all neighbourhoods within eps distance

        C = 0
        D = data
        n = D.shape[0]
        visited = [False]* n
        # labels = [None]*n
        labels = -np.ones(n)#initialise labels to all -1

        for index in range(0,n):
            if visited[index] == True:
                continue
            visited[index] = True
            NeighbourPts = self.regionQuery(index,eps,neighborhoods)

            if len(NeighbourPts) < MinPts:
                labels[index] = -1
            else:
                C += 1
                self.expandCluster(index,NeighbourPts,C,eps,MinPts,visited,labels,neighborhoods)
        return labels

    def expandCluster(self,ptIndex,neighbourPts,C,eps,MinPts,visited,labels,neighborhoods):
        labels[ptIndex] = C
        i = 0
        while i < len(neighbourPts):
            pp = neighbourPts[i]
            if visited[pp]==False:
                visited[pp]=True
                neighbourPts2 = self.regionQuery(pp,eps,neighborhoods)
                if len(neighbourPts2)>=MinPts:
                    neighbourPts = np.append(neighbourPts,neighbourPts2)
            if labels[pp] == None or labels[pp]==-1:
                labels[pp] = C
            i +=1


def plotPCA(data,title,showNow,labels):
        fig = plt.figure(title)
        mlab_pca = mlabPCA(data)
        plt.scatter(mlab_pca.Y[:,0],mlab_pca.Y[:,1],c=labels.astype(np.float), alpha=1)
        if(showNow):plt.show()


import sys

def main():
    ep = .8
    min = 3
    path = "data/dataset1.txt"

    print str(len(sys.argv))
    if len(sys.argv)<4:
        print "Usage : python dbscan.py <eps> <minPts> <input file path>"
        print "running for default values"
    if len(sys.argv)>1 and sys.argv[1]:
        ep = float(sys.argv[1]);

    if len(sys.argv)>2 and sys.argv[2]:
        min = int(sys.argv[2]);

    if len(sys.argv)>3 and sys.argv[3]:
        path = str(sys.argv[3]);

    si = 2
    if "iyer" in path: si = 3
    X = np.loadtxt(path)[:,si:]
    trueLabels = np.loadtxt(path)[:,1]

    dbscan = DBScan();
    labels = dbscan.fit(X,ep,min)
    # np.savetxt("dbscanLabels.txt",labels.astype(int),fmt='%d')
    # np.savetxt("dbscanout.txt",labels)

    plotPCA(X,path.split("/")[1].split(".")[0]+"_predicted_clusters_min:"+str(min)+"_eps:"+str(ep),True,labels)

    import metrics #code in metrics.py
    print "jaccard coeff"
    jac_metric = metrics.calculateJaccardCoeff(trueLabels,labels)
    print jac_metric

    print "correlation"
    cor =  metrics.computeCorrelation(X,labels)
    print cor

    from sklearn.metrics import adjusted_rand_score as rand_score #library function
    print "adjusted rand score"
    rand_met = rand_score(trueLabels.T,labels.T)
    print rand_met

    # results.append([jac_metric,rand_met,cor,ep,min])
    # print np.corrcoef(X,labels)

    dbscan.plotKnn(X,min)

    # print results
    # plotPCA(X,"Ground Truth",True,trueLabels)

if __name__ == "__main__": main()