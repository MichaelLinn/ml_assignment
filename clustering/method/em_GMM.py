# -*- coding: utf-8 -*-
# @Time    : 20/10/2017 11:57 PM
# @Author  : Jason Lin
# @File    : em_GMM.py
# @Software: PyCharm Community Edition

import numpy as np
import matplotlib.pylab as plt
import os

class em_GMM:

    # inputX = np.matrix([[1, 6, 3], [4, 5, 6], [3, 2, 1], [1, 3, 7]]).T
    mu_ = []
    sigma_ = []
    class_k = 4

    def __init__(self):

        self.loadData()

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




    def loadData(self):
        os.getcwd()
        dataA_file = "../data/PA2-cluster-data/cluster_data_text/cluster_data_dataA_X.txt"
        data = np.loadtxt(dataA_file)
        self.inputX = data.T

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

    def gaussianDistribution(self, i, j):

        mu_ = np.mat(self.mu_[j]).T
        cov_ = np.mat(self.sigma_[j])
        inputX = np.mat(self.inputX[i]).T

        # print inputX
        # print i,":",j, "----\n", cov_
        res = 1.0/(np.power((2 * np.pi), 0.5 * self.dimX) * np.power(np.linalg.det(cov_), 0.5)) * np.exp(-0.5 * (inputX - mu_).T * np.linalg.inv(cov_) * (inputX - mu_))

        return res

    def updateParams(self, j):

        n = len(self.inputX)
        z_ = []
        for i in range(n):
            z_.append(self.gen_Z(i, j))

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
                z_ik = self.gen_Z(i, k)
                if z_ik > z_i:
                    z_i = z_ik
                    label = k
            dataMap[label].append(list(self.inputX[i]))

        fig = plt.figure()


        for c, data in dataMap.items():
            data = np.array(data)
            x = np.array(data[:, 0])
            y = np.array(data[:, 1])
            plt.scatter(x, y, s=5)

        fig.savefig("pic.eps",format='eps', dpi=1000)


def main():
    test = em_GMM()
    print test.dimX
    for i in range(70):
        print "--------", i, "--------"
        for j in range(test.class_k):
            test.updateParams(j)
    test.plotRes()

if __name__ == "__main__":
    main()


