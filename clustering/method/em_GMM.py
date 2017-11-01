# -*- coding: utf-8 -*-
# @Time    : 20/10/2017 11:57 PM
# @Author  : Jason Lin
# @File    : em_GMM.py
# @Software: PyCharm Community Edition

import numpy as np
import matplotlib.pylab as plt
import os
from numpy.linalg import cholesky
import pylab as pl
from PIL import Image
import picProcess as pa2
import time
import scipy.cluster.vq as vq


class em_GMM:

    # inputX = np.matrix([[1, 6, 3], [4, 5, 6], [3, 2, 1], [1, 3, 7]]).T
    mu_ = []
    sigma_ = []

    def __init__(self, class_k = 4, data=None):

        self.class_k = class_k
        self.loadData(data)

        self.dimX = len(self.inputX[0])
        t_mu = []

        for i in range(len(self.inputX.T)):
            t_max = max(self.inputX.T[i])
            t_min = min(self.inputX.T[i])
            tt_mu = []
            for k in range(self.class_k):
                tt_mu.append(np.random.uniform(t_min, t_max))
            t_mu.append(np.array(tt_mu))
        t_mu = np.array(t_mu)

        for i in range(self.class_k):
            self.mu_.append(t_mu[:, i])

        for i in range(self.class_k):
            t_sigma = np.eye(self.dimX)
            self.sigma_.append(t_sigma)

        self.pi_ = np.ones(self.class_k) * 1.0/self.class_k
        print self.mu_


    def loadData(self, data=None):

        if data is None:
            os.getcwd()
            dataA_file = "../data/PA2-cluster-data/cluster_data_text/cluster_data_dataC_X.txt"
            d = np.loadtxt(dataA_file)
            self.inputX = d.T

        self.inputX = data


    def gen_Z(self, i, j):

        denominator = 0
        numerator = 0
        for k in range(self.class_k):
            tem = self.pi_[k] * self.gaussianDistribution(i, k)
            if k == j:
                numerator = tem
            denominator = denominator + tem
        z_ij = numerator / denominator
        # print "z:", z_ij
        return float(z_ij)


    def gen_PreciseZ(self, i, j):
        max_l = -np.inf
        log_pxij = 0
        log_ga = []
        for k in range(self.class_k):
            tem = self.logGaussian(i, k)
            log_ga.append(tem)
            if tem > max_l:
                max_l = tem
            if k == j:
                log_pxij = tem + np.log(self.pi_[j])

        log_pxi = max_l + np.log(np.sum(np.exp(np.array(log_ga) - max_l)))
        z_ij = np.exp(log_pxij - log_pxi)
        return float(z_ij)

    def logGaussian(self, i, j):

        mu_ = np.mat(self.mu_[j]).T
        cov_ = np.mat(self.sigma_[j])
        inputX = np.mat(self.inputX[i]).T

        # print cov_
        L_mat = cholesky(cov_)

        logDetCov = 2.0 * np.sum(np.log(np.diag(L_mat)))

        res = - 0.5 * self.dimX * np.log(2 * np.pi) - 0.5 * logDetCov \
              - 0.5 * (inputX - mu_).T * np.linalg.inv(cov_) * (inputX - mu_)

        return float(res)


    def gaussianDistribution(self, i, j):

        mu_ = np.mat(self.mu_[j]).T
        cov_ = np.mat(self.sigma_[j])
        inputX = np.mat(self.inputX[i]).T

        res = 1.0/(np.power((2 * np.pi), 0.5 * self.dimX) * np.power(np.linalg.det(cov_), 0.5)) * \
              np.exp(-0.5 * (inputX - mu_).T * np.linalg.inv(cov_) * (inputX - mu_))

        return float(res)

    def updateParams(self, j):

        n = len(self.inputX)
        z_ = []
        for i in range(n):
            z_.append(self.gen_PreciseZ(i, j))

        # print z_
        n_j = sum(z_)

        mu_j = np.zeros(self.inputX[0].T.shape)
        sigma_j = np.zeros((self.dimX, self.dimX))

        for i in range(n):
            inputX = self.inputX[i].T
            mu_j = mu_j + z_[i] * inputX
        mu_j = mu_j / n_j

        for i in range(n):
            inputX = np.mat(self.inputX[i]).T
            mu_t = np.mat(mu_j).T
            tem = z_[i] * np.mat(inputX - mu_t) * np.mat(inputX - mu_t).T
            sigma_j = sigma_j + tem

        sigma_j = sigma_j / n_j
        pi_j = n_j / n

        self.mu_[j] = mu_j
        self.sigma_[j] = sigma_j
        self.pi_[j] = pi_j

    def plotRes(self):

        dataMap = {}
        for i in range(self.class_k):
            dataMap[i] = []

        for i in range(len(self.inputX)):
            z_i = - np.inf
            label = -1
            for k in range(self.class_k):
                z_ik = self.gen_PreciseZ(i, k)
                if z_ik > z_i:
                    z_i = z_ik
                    label = k
            dataMap[label].append(list(self.inputX[i]))

        fig = plt.figure()

        print dataMap

        for c, data in dataMap.items():
            data = np.array(data)
            if len(data) <= 0:
                continue
            x = np.array(data[:, 0])
            y = np.array(data[:, 1])
            plt.scatter(x, y, s=5)

        fig.savefig("em_GMM_1C.eps",format='eps', dpi=1000)

    def solveProblem1(self):
        for i in range(30):
            print "--------", i, "--------"
            for j in range(self.class_k):
                self.updateParams(j)
        self.plotRes()

    def runEMGMM(self, iteration=30):
        for i in range(iteration):
            print "--------", i, "--------"
            for j in range(self.class_k):
                self.updateParams(j)


def imageSegment():
    img = Image.open('../data/PA2-cluster-images/images/62096.jpg')
    fig = pl.figure()
    pl.subplot(1, 3, 1)
    pl.imshow(img)
    X, L = pa2.getfeatures(img, 7)
    # print X.T
    # X = vq.whiten(X.T)
    X = X.T
    # cent, res = km.kmeansClustering()
    # km.plotRes(centroids=cent, clusterAssment=res)
    X = vq.whiten(X)
    em = em_GMM(class_k=2, data=X)
    em.runEMGMM(iteration=100)

    # Get clustering result
    Y = []
    for i in range(len(em.inputX)):
        z_i = - np.inf
        label = -1
        for k in range(em.class_k):
            z_ik = em.gen_PreciseZ(i, k)
            if z_ik > z_i:
                z_i = z_ik
                label = k
        Y.append(label)
    # make segmentation image from labels
    Y = np.array(Y)
    Y = Y + 1
    segm = pa2.labels2seg(np.array(Y), L)
    pl.subplot(1, 3, 2)
    pl.imshow(segm)

    # color the segmentation image
    csegm = pa2.colorsegms(segm, img)
    pl.subplot(1, 3, 3)
    pl.imshow(csegm)
    # pl.show()
    fig.savefig("emGMM_62096.eps", format='eps', dpi=1000)

if __name__ == "__main__":
    try:
        imageSegment()
    except Exception:
        pass






