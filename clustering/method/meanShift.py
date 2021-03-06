# -*- coding: utf-8 -*-
# @Time    : 29/10/2017 5:45 PM
# @Author  : Jason Lin
# @File    : meanShift.py
# @Software: PyCharm Community Edition

import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import pylab as pl
from PIL import Image
import picProcess as pa2

import scipy.cluster.vq as vq
from sklearn.cluster import MeanShift

class meanShift:

    h = 1.65

    def __init__(self,data=None, h_c=8.0, h_p=15.0):
        self.loadData(data)
        self.n_ = len(self.inputX)
        self.dim = len(self.inputX[0])
        self.h_c = h_c
        self.h_p = h_p

    def loadData(self, data):
        if data is None:
            os.getcwd()
            dataA_file = "../data/PA2-cluster-data/cluster_data_text/cluster_data_dataA_X.txt"
            data = np.loadtxt(dataA_file)
            self.inputX = data.T
            self.inputX_mu = data.T
        else:
            self.inputX = data
            self.inputX_mu = data


    def gaussian(self, inputX, mu_):
        dimX = self.dim
        cov_ = self.h**2 * np.eye(dimX)
        inputX = np.mat(inputX).T
        mu_ = np.mat(mu_).T

        res = 1.0/(np.power((2 * np.pi), 0.5 * dimX) * np.power(np.linalg.det(cov_), 0.5)) * \
              np.exp(-0.5 * (inputX - mu_).T * np.linalg.inv(cov_) * (inputX - mu_))
        # print res
        return float(res)

    def kernel_pic(self, inputX, mu_):

        h_c = self.h_c
        h_p = self.h_p

        res = 1.0 / (np.power(2 * np.pi * h_c * h_p, 2))
        expon = -1.0 / (2*np.power(h_c, 2)) * np.sum(np.power((inputX[:2] - mu_[:2]), 2)) \
                -1.0 / (2*np.power(h_p, 2)) * np.sum(np.power((inputX[2:] - mu_[2:]), 2))

        res = res * np.exp(expon)
        return float(res)


    def updateX_i(self, i, mu_):

        n = self.n_
        numertor = 0
        denomonator = 0
        for j in range(n):
            # ga = self.gaussian(self.inputX[j], mu_)
            ga = self.kernel_pic(self.inputX[j], mu_)
            numertor += self.inputX[j] * ga
            denomonator += ga

        res = np.array(numertor*1.0/denomonator)

        return res

    def run(self):
        mu_ = []
        for i in range(self.n_):
            print "--------",i,"--------"
            mu_t = self.inputX_mu[i]
            j = 0
            while(True):
                j += 1
                if j > 200:
                    break
                tem = self.updateX_i(i, mu_t)
                if np.sum(np.abs(mu_t - tem)) < 0.1:
                    break
                mu_t = tem

            mu_.append(mu_t)
        return np.array(mu_)

def solveProblem1():
    ms = meanShift()
    label = ms.run()

    label = np.around(label, decimals=2)

    x_value = np.unique(label[:, 0])
    y_value = np.unique(label[:, 1])

    print len(x_value)

    fig = plt.figure()
    plt.scatter(label[:, 0], label[:, 1], s=2, c="black")
    plt.scatter(ms.inputX[:, 0], ms.inputX[:, 1], s=1)

    for item in x_value:
        index = label[:, 0] == item
        plt.scatter(ms.inputX[index, 0], ms.inputX[index, 1], s=2)
    fig.savefig("meashift_1A.eps", format='eps', dpi=1000)

def imageSegament(image_name, h_c=15., h_p=20.):

    imageFile = '../data/PA2-cluster-images/images/' + str(image_name)

    name = "../result/meanshift/" + str(image_name).split(".")[0] + "_" + str(int(h_c)) + "_" + str(int(h_p)) + ".eps"

    print imageFile
    print name

    img = Image.open(imageFile)
    fig = pl.figure()
    pl.subplot(1, 3, 1)
    pl.imshow(img)
    X, L = pa2.getfeatures(img, 7)
    # print X.T
    # X = vq.whiten(X.T)
    X = X.T
    # cent, res = km.kmeansClustering()
    # km.plotRes(centroids=cent, clusterAssment=res)
    # X = vq.whiten(X)

    ms = meanShift(data=X, h_c=h_c, h_p=h_p)
    mu_all = ms.run()
    mu_all = np.around(mu_all, decimals=0)
    x_mu = np.unique(mu_all[:, 0])
    # y_mu = np.unique(mu_all[:, 1])
    label = np.zeros(len(ms.inputX))
    cls = 0
    for item in x_mu:
        idx = mu_all[:, 0] == item
        label[idx] = cls
        cls += 1

    # ms = MeanShift(bandwidth=1.2, bin_seeding=True).fit(X)

    # Get clustering result
    # Y = np.array(ms.labels_).flatten()
    Y = label
    # make segmentation image from labels
    Y = np.array(Y)
    Y = Y + 1
    segm = pa2.labels2seg(np.array(Y), L)
    s = "h_c=" + str(ms.h_c) + ", " + "h_p=" + str(ms.h_p)
    pl.title(s)
    pl.subplot(1, 3, 2)
    pl.imshow(segm)

    # color the segmentation image
    csegm = pa2.colorsegms(segm, img)
    pl.subplot(1, 3, 3)
    pl.imshow(csegm)

    fig.savefig(name, format='eps', dpi=1000)

def main():

    os.getcwd()
    imageFilePath = "../data/PA2-cluster-images/images"
    image_list = []
    for file in os.listdir(imageFilePath):
        if file.endswith(".jpg"):
            image_list.append(file)
    print image_list
    inver = image_list[::-1]
    for img in image_list:
        for h_c in range(10, 21, 5):
            for h_p in range(10, 21, 5):
                print h_c
                print h_p
                imageSegament(img, h_c, h_p)


if __name__ == "__main__":
    main()
