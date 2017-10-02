# -*- coding: utf-8 -*-
# @Time    : 10/1/17 5:22 PM
# @Author  : Jason Lin
# @File    : test.py
# @Software: PyCharm Community Edition

import pandas as pd
import numpy as np
import os
from pylab import *
import matplotlib.pylab as plt
from scipy.optimize import linprog
from scipy import stats


class Regression:

    polyOrder = 5


    def __init__(self):
        plt.figure(1)
        self.load_dataMat()

    def load_dataMat(self):
        os.getcwd()
        os.chdir(os.pardir)
        os.chdir("data")

        self.vecX = np.loadtxt("polydata_data_sampx.txt")
        self.vecY = np.loadtxt("polydata_data_sampy.txt")

        # scatter(self.vecX, self.vecY, marker="o", s=9, alpha=0.5, c="black")
        # show()

    # least squares regression (LS)
    def leastSquaresReg(self):
        matPhi = self.genPhi()
        theta_LS = (matPhi * matPhi.T).I * matPhi * np.mat(self.vecY).T
        # print theta_LS
        plt.subplot(221)
        self.plotReg(theta_LS)

    # regularized LS (RLS)
    def regularizedLSReg(self):
        lambda_ = 0.1
        matPhi = self.genPhi()
        matUnit = np.identity((matPhi * matPhi.T).shape[0])
        theta_RLS = (matPhi * matPhi.T + lambda_ * matUnit).I * matPhi * np.mat(self.vecY).T
        plt.subplot(222)
        self.plotReg(theta_RLS)


    def plotReg(self, theta_):

        # print theta_
        # print "-------------------"

        testX = np.linspace(-2.0, 2.0, 200, endpoint=True)

        theta_ = np.array(theta_[::-1])  # reverse
        theta_ = theta_.flatten()     # change ND array into 1D array
        p = np.poly1d(theta_)
        resY = p(testX)

        plt.plot(testX, resY)
        plt.scatter(self.vecX, self.vecY, marker="o", s=9, alpha=0.5, c="black")

    def genPhi(self):

        f = lambda x, n: x ** n
        matPhi = []
        for x in self.vecX:
            col = []
            for i in range(self.polyOrder + 1):
                col.append(f(x, i))
            matPhi.append(col)
        matPhi = (np.asmatrix(matPhi)).T
        # print matPhi
        return matPhi

    def robustReg(self):

        vecF = []
        for i in range(self.polyOrder + 1):
            vecF.append(0)
        for i in range(len(self.vecX)):
            vecF.append(1)
        vecF = np.array(vecF)

        matPhi = self.genPhi()
        matIdentity = np.identity(len(self.vecX))
        matA_1 = np.column_stack((-matPhi.T, -matIdentity))
        matA_2 = np.column_stack((matPhi.T, -matIdentity))
        matA = np.row_stack((matA_1, matA_2))
        # print matA.shape
        vec_b = np.concatenate((-self.vecY, self.vecY))
        # objectiveFunc = lambda X: vecF.T * X
        # consFunc = lambda X: -(matA * X - vec_b)
        res = linprog(vecF, A_ub = matA, b_ub = vec_b)
        # print res
        theta_ = res.get("x")[:6]
        # print theta_
        plt.subplot(223)
        self.plotReg(theta_)

    def baysesianReg(self):
        alpha_ = 1
        noise = 5
        matPhi = self.genPhi()
        sigma_theta = (1.0/alpha_*np.identity(self.polyOrder+1) + 1.0/ 5 * matPhi*matPhi.T).I
        mu_theta = 1.0/(noise) * sigma_theta * matPhi * np.mat(self.vecY).T

        # plot regression
        testX = np.linspace(-2.0, 2.0, 200, endpoint=True)

        f = lambda x, n: x ** n
        resY = []
        for x in testX:
            phi_s = []
            for i in range(self.polyOrder + 1):
                phi_s.append(f(x,i))
            phi_s = np.mat(phi_s)
            mu_s = phi_s * mu_theta
            sigma_s = phi_s * sigma_theta * phi_s.T
            resY.append(stats.norm.pdf(x, loc=mu_s, scale=sqrt(sigma_s))[0])
        # print testX
        # print resY

        plt.subplot(224)
        plt.plot(testX, resY)
        plt.scatter(self.vecX, self.vecY, marker="o", s=9, alpha=0.5, c="black")


reg = Regression()

reg.leastSquaresReg()  # least_squares(LS)
reg.regularizedLSReg()   # regularized LS (RLS)
reg.robustReg()   # Robust Regression (RR)

reg.baysesianReg()

plt.show()

