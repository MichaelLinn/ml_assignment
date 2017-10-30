# -*- coding: utf-8 -*-
# @Time    : 16/10/2017 8:33 PM
# @Author  : Jason Lin
# @File    : kmeans.py
# @Software: PyCharm Community Edition

import numpy as np
import os
import matplotlib.pylab as plt


class kmeans:


    def __init__(self, k):
        self.dataset = self.loadData()
        self.k = k

    def countDist(self, vecA, vecB):
        return np.sum(np.sum(np.power((vecA-vecB), 2)))


    def initCenteroids(self):
        n = np.shape(self.dataset)[1]
        print n
        centroids = np.mat(np.zeros((self.k,n)))
        for j in range(n):          # n is the dimension of the input
            minJ = min(self.dataset[:,j])
            rangeJ = float(max(self.dataset[:, j]) - minJ)
            centroids[:, j] = minJ + rangeJ * np.random.rand(self.k, 1)
        # print centroids
        return centroids


    def kmeansClustering(self):
        m = np.shape(self.dataset)[0]  # the number of input
        clusterAssment = np.mat(np.zeros((m,2)))
        centroids = self.initCenteroids()
        clusterChanged = True
        while clusterChanged:
            clusterChanged = False
            for i in range(m):
                minDist = np.inf
                minIdx = -1
                for j in range(self.k):
                    distJI = self.countDist(centroids[j,:], self.dataset[i,:])
                    if distJI < minDist:
                        minDist = distJI
                        minIdx = j
                if clusterAssment[i, 0] != minIdx:
                    clusterChanged = True
                clusterAssment[i, :] = minIdx, minDist
            for cent in range(self.k):
                ptsInClust = self.dataset[np.nonzero(clusterAssment[:, 0] == cent)[0]]
                centroids[cent, :] = np.mean(ptsInClust, axis=0)

        # print centroids, clusterAssment
        return centroids, clusterAssment


    def loadData(self):
        os.getcwd()
        dataA_file = "../data/PA2-cluster-data/cluster_data_text/cluster_data_dataA_X.txt"
        matDataA = np.mat(np.loadtxt(dataA_file)).T
        return matDataA

    def plotRes(self, centroids, clusterAssment):

        dataMap = {}
        for i in range(self.k):
            dataMap[i] = []
        clusterAssment = np.array(clusterAssment)
        for idx in range(len(self.dataset)):
            c = int(clusterAssment[idx][0])
            dataMap[c].append(idx)

        fig = plt.figure()

        i = 0
        for c, data in dataMap.items():
            d = np.array(self.dataset[data])
            x = np.array(d[:,0])
            y = np.array(d[:,1])
            plt.scatter(x,y)
            i = i + 1

        fig.savefig("pic")



def main():

    km = kmeans(4)
    cent, res = km.kmeansClustering()
    km.plotRes(centroids=cent, clusterAssment=res)

if __name__ == '__main__':
    main()





























