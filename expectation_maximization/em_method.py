# -*- coding: utf-8 -*-
# @Time    : 11/10/2017 12:56 PM
# @Author  : Jason Lin
# @File    : em_method.py
# @Software: PyCharm Community Edition


import numpy as np
import math

class em_mixPossion:


    x_London = [229, 211, 93, 35, 7, 1]
    x_Antwerp = [325, 115, 67, 30, 18, 21]


    def __init__(self, K):

        self.x_selected = self.x_Antwerp
        self.K = K
        d = 2.0/(K*(K+1))
        self.X = np.array([])
        self.pi_ = np.zeros(self.K) + 1.0/self.K
        # self.pi_ = np.array([i*d for i in range(1, self.K+1)])
        # self.pi_ = self.pi_[::-1]
        self.lambda_ = np.ones(K)

        for i in range(len(self.x_selected)):
            t = np.ones(self.x_selected[i]) * i
            self.X = np.concatenate((self.X, t))


    def genZ_(self, x, j):

        denominator = 0
        numerator = 0
        for k in range(self.K):
            tem = self.pi_[k] * self.poisson(x, k)
            if k == j:
                numerator = tem
            denominator = denominator + tem
        return numerator / denominator

    def poisson(self, x, j):

        res = (1.0 / math.factorial(x)) * np.exp(-self.lambda_[j]) * (self.lambda_[j] ** x)
        return res

    def updataParam(self, j):

        n_j = np.sum(np.array([self.genZ_(self.X[i], j) for i in range(len(self.X))]))

        pi_j = n_j / len(self.X)

        lambda_j = (1.0 / n_j) * np.sum(np.array([self.genZ_(self.X[i], j) * self.X[i] for i in range(len(self.X))]))

        self.pi_[j] = pi_j
        self.lambda_[j] = lambda_j

    def countLogLikelihood(self):
        sum = 0
        for i in range(len(self.X)):
            sum = sum + np.log(np.sum(np.array([self.pi_[j]*self.poisson(self.X[i], j) for j in range(self.K)])))

        print sum
        return sum

    def main(self):

        for j in range(self.K):
            for i in range(1000):
                print "-----------", i, "-----------"
                self.updataParam(j)

        print self.pi_
        print self.lambda_


sol = em_mixPossion(K=5)
sol.main()
sol.countLogLikelihood()

