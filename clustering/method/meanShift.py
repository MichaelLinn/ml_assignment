# -*- coding: utf-8 -*-
# @Time    : 29/10/2017 5:45 PM
# @Author  : Jason Lin
# @File    : meanShift.py
# @Software: PyCharm Community Edition

import numpy as np
import os
import matplotlib.pylab as plt


class meanShift:

    h = 1.65
    def __init__(self):
        self.loadData()
        self.n_ = len(self.inputX)
        self.dim = len(self.inputX[0])

    def loadData(self):
        os.getcwd()
        dataA_file = "../data/PA2-cluster-data/cluster_data_text/cluster_data_dataA_X.txt"
        data = np.loadtxt(dataA_file)
        self.inputX = data.T
        self.inputX_mu = data.T


    def gaussian(self, inputX, mu_):
        dimX = self.dim
        cov_ = self.h**2 * np.eye(dimX)
        inputX = np.mat(inputX).T
        mu_ = np.mat(mu_).T

        res = 1.0/(np.power((2 * np.pi), 0.5 * dimX) * np.power(np.linalg.det(cov_), 0.5)) * \
              np.exp(-0.5 * (inputX - mu_).T * np.linalg.inv(cov_) * (inputX - mu_))
        # print res
        return float(res)

    def updateX_i(self, i, mu_):

        n = self.n_
        numertor = 0
        denomonator = 0
        for j in range(n):
            ga = self.gaussian(self.inputX[j], mu_)

            numertor += self.inputX[j] * ga
            denomonator += ga

        res = np.array(numertor*1.0/denomonator)

        return res

    def run(self):
        label = []
        for i in range(self.n_):
            print "--------",i,"--------"
            mu_t = self.inputX_mu[i]
            while(True):
                # print mu_t
                tem = self.updateX_i(i, mu_t)
                # print tem
                # print mu_t
                # print "***************"
                if np.sum(np.abs(mu_t - tem)) < 0.0001:
                    break
                mu_t = tem
            label.append(mu_t)
        return np.array(label)



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





