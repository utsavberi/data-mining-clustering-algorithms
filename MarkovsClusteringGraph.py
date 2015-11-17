from __future__ import division
import numpy as np
__author__ = 'utsav'


class MarcovsClustering:
    def fit(self,AdjMatrix,e = 2,r = 2):
        if e <= 1:
            raise ValueError("e should be bigger than 1")
        if r <= 1:
            raise ValueError("r should be bigger than 1")

        dim = AdjMatrix.shape[0]

        #Add self loop
        AdjMatrix = AdjMatrix+np.eye(dim,dim)

        #normalize
        colSum = np.sum(AdjMatrix,axis = 0)
        AdjMatrix = AdjMatrix/colSum
        for o in range(0,1000):
            #prev to check for convergence
            prev = AdjMatrix.copy();

            #expand
            for i in range (e-1):
                AdjMatrix = np.dot(AdjMatrix,AdjMatrix)

            #inflate and normalize
            AdjMatrix = np.power(AdjMatrix,r)
            colSum = np.sum(AdjMatrix,axis = 0)
            AdjMatrix = AdjMatrix/colSum

            #check for convergence
            if np.array_equal(prev,AdjMatrix):
                print("Converged at iteration %d" % o)
                break;

        return AdjMatrix


def convertStrVerticesToNumeric(A):
    uniq = np.unique(A.ravel())
    counter = 0
    dict = {}
    for i in uniq:
        dict[i] = counter
        counter+=1

    B = np.empty((0,2), int)
    for row in A:
        x = np.array([[dict[row[0]],dict[row[1]]]])
        B = np.vstack((B,x))

    return B


def edgesToAdjacencyMatrix(E):
    # find the dimension of the square matrix
    dim = np.amax(E)
    #int(max(max(set(i)), max(set(j))))
    dim += 1
    i, j = E[:,0], E[:,1]
    print dim

    #generate adj matrix from edge data
    B = np.zeros((dim,dim))
    for ii,jj in zip(i,j):
        B[ii,jj] = 1
        B[jj,ii] = 1
    B=np.delete(B, 0, 0)
    B=np.delete(B, 0, 1)
    return B

def markovsToClusters(M):
    return [np.where(x>0)[0] for x in M]


def main():
    # inputFiles = ["data/attweb_net.txt","data/yeast_undirected_metabolic.txt","data/physics_collaboration_net.txt"]
    inputFiles = ["data/new_att.txt","data/new_collaboration.txt","data/new_yeast.txt"]
    for file in inputFiles:
        # Edges = convertStrVerticesToNumeric(np.loadtxt(file, str))
        # AdjMat = edgesToAdjacencyMatrix(Edges)
        AdjMat = edgesToAdjacencyMatrix(np.loadtxt(file))

        for r in np.arange(1.1,2,.2):
            C = MarcovsClustering().fit(AdjMat,r=r)
            numClusters=0
            for x in markovsToClusters(C):
                if len(x)>0:
                    numClusters+=1
            print ("num of clusters in ",file,r,numClusters)
            np.savetxt("out/" + file.split("/")[1] + str(r), C.astype(int), fmt='%.4f')

        r = 2
        C = MarcovsClustering().fit(AdjMat,r=r)
        numClusters=0
        for x in markovsToClusters(C):
            if len(x)>0:
                numClusters+=1
        print ("num of clusters in ",file,r,numClusters)
        np.savetxt("out/" + file.split("/")[1] + str(r), C,'%.4f')

    print "done"


if __name__ == "__main__": main()
