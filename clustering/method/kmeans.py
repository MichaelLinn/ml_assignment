# -*- coding: utf-8 -*-
# @Time    : 16/10/2017 8:33 PM
# @Author  : Jason Lin
# @File    : kmeans.py
# @Software: PyCharm Community Edition

import numpy as np
import os
import matplotlib.pylab as plt
import picProcess as pa2
from PIL import Image
import pylab as pl
import scipy.cluster.vq as vq

class kmeans:


    def __init__(self, k=2, data=None):
        if data is None:
            self.dataset = self.loadData()
        else:
            self.dataset = data
        self.k = k

    def loadData(self):
        os.getcwd()
        dataA_file = "../data/PA2-cluster-data/cluster_data_text/cluster_data_dataC_X.txt"
        matDataA = np.mat(np.loadtxt(dataA_file)).T
        return matDataA

    def countDist(self, vecA, vecB):
        return np.sum(np.power((vecA-vecB), 2))


    def initCenteroids(self):
        n = np.shape(self.dataset)[1]
        # print n
        centroids = np.mat(np.zeros((self.k, n)))
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
        print centroids
        clusterChanged = True
        t = 0
        while clusterChanged:
            t += 1
            print "----------",t,"----------"

            clusterChanged = False
            for i in range(m):
                minDist = np.inf
                minIdx = -1
                for j in range(self.k):
                    c = centroids[j,:]
                    p = self.dataset[i,:]
                    distJI = self.countDist(c, p)
                    if distJI < minDist:
                        minDist = distJI
                        minIdx = j
                # print minIdx
                if clusterAssment[i, 0] != minIdx:
                    clusterChanged = True
                clusterAssment[i, :] = minIdx, minDist
                # print minIdx,":",minDist
            for cent in range(self.k):
                ptsInClust = self.dataset[np.nonzero(clusterAssment[:, 0] == cent)[0]]
                centroids[cent, :] = np.mean(ptsInClust, axis=0)
                # print centroids

            if t > 1000:
                clusterChanged = False

        # print centroids, clusterAssment
        return centroids, clusterAssment


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


def main(dataset=None):

    img = Image.open('../data/PA2-cluster-images/images/62096.jpg')

    pl.subplot(1, 3, 1)
    pl.imshow(img)
    X, L = pa2.getfeatures(img, 7)

    # print X.T
    X = vq.whiten(X.T)
    # X = X.T
    # lambda_ = 0.1
    # X[:, 2:4] = X[:, 2:4] * lambda_

    km = kmeans(2, data=X)
    cent, res = km.kmeansClustering()
    # km.plotRes(centroids=cent, clusterAssment=res)
    label = (np.array(res[:, 0])+1).flatten()

    # make segmentation image from labels
    segm = pa2.labels2seg(label, L)
    pl.subplot(1, 3, 2)
    pl.imshow(segm)

    # color the segmentation image
    csegm = pa2.colorsegms(segm, img)
    pl.subplot(1, 3, 3)
    pl.imshow(csegm)
    # pl.show()
    pl.savefig("kmean_62096.eps", format='eps', dpi=1000)

if __name__ == '__main__':
    main()




















