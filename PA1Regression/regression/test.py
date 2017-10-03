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
from cvxopt import matrix, solvers


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
        plt.subplot(321, xlabel="Least Squares")
        self.plotReg(theta_LS)

    # regularized LS (RLS)
    def regularizedLSReg(self):
        lambda_ = 0.1
        matPhi = self.genPhi()
        matUnit = np.identity((matPhi * matPhi.T).shape[0])
        theta_RLS = (matPhi * matPhi.T + lambda_ * matUnit).I * matPhi * np.mat(self.vecY).T
        plt.subplot(322, xlabel="Regularized")
        self.plotReg(theta_RLS)


    def plotReg(self, theta_):

        # print theta_
        # print "-------------------"

        testX = np.linspace(-2.0, 2.0, 200, endpoint=True)

        theta_ = np.array(theta_[::-1])  # reverse
        theta_ = theta_.flatten()     # change ND array into 1D array
        p = np.poly1d(theta_)
        resY = p(testX)

        predictY = p(self.vecX)

        error = self.countMeanSqureErr(predictY, self.vecY)
        text = "Error=" + str(error)
        plt.text(-0.5,25,text)

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
            vecF.append(0.0)
        for i in range(len(self.vecX)):
            vecF.append(1.0)
        vecF = np.array(vecF)

        matPhi = self.genPhi()
        matIdentity = np.identity(len(self.vecX))
        matA_1 = np.column_stack((-matPhi.T, -matIdentity))
        matA_2 = np.column_stack((matPhi.T, -matIdentity))
        matA = np.row_stack((matA_1, matA_2))
        vec_b = np.concatenate((-self.vecY, self.vecY))
        # objectiveFunc = lambda X: vecF.T * X
        # consFunc = lambda X: -(matA * X - vec_b)
        # res = linprog(vecF, A_ub=matA, b_ub=vec_b)
        # theta_ = res.x[:6]
        sol = solvers.lp(matrix(vecF), matrix(matA), matrix(vec_b))
        theta_ =  list(sol['x'][:6])
        print theta_


        plt.subplot(323, xlabel="Robust")
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
        varErr = []
        for x in testX:
            phi_s = []
            for i in range(self.polyOrder + 1):
                phi_s.append(f(x,i))
            phi_s = np.mat(phi_s)
            mu_s = phi_s * mu_theta
            sigma_s = phi_s * sigma_theta * phi_s.T

            varErr.append(sqrt(sigma_s.tolist()[0][0]))
            resY.append(mu_s.tolist()[0][0])

        predictY = []
        for x in self.vecX:
            phi_s = []
            for i in range(self.polyOrder + 1):
                phi_s.append(f(x, i))
            phi_s = np.mat(phi_s)
            mu_s = phi_s * mu_theta
            # sigma_s = phi_s * sigma_theta * phi_s.T
            predictY.append(mu_s.tolist()[0][0])


        err = self.countMeanSqureErr(predictY, self.vecY)
        text = "Error=" + str(err)

        plt.subplot(324, xlabel="Baysesian")
        # plt.plot(testX, resY)
        plt.errorbar(testX, resY, yerr=varErr, ecolor="red")
        plt.text(-0.5, 25, text)

        plt.scatter(self.vecX, self.vecY, marker="o", s=9, alpha=0.5, c="black")

    def lassoReg(self):
        lambda_ = 1.0
        x_dim = 2 * (self.polyOrder + 1)
        matPhi = self.genPhi()
        matH_1 = np.column_stack((matPhi*matPhi.T, -matPhi*matPhi.T))
        matH_2 = np.column_stack((-matPhi*matPhi.T, matPhi*matPhi.T))
        matH = np.row_stack((matH_1, matH_2))
        print matH
        mat_func = lambda_ * np.mat(np.ones(x_dim)).T - np.row_stack((matPhi* (np.mat(self.vecY)).T, -matPhi* (np.mat(self.vecY).T)))
        print mat_func
        matG = np.identity(x_dim)
        vec_h = np.zeros(x_dim)

        sol = solvers.qp(matrix(matH), matrix(mat_func), matrix(matG), matrix(vec_h))
        # print sol['x']

        theta_ = list(sol['x'][:6] - sol['x'][6:])
        # print theta_
        plt.subplot(325, xlabel="Lasso")
        self.plotReg(theta_)

    def countMeanSqureErr(self, resY, trueY):
        p_y = np.array(resY)
        t_y = np.array(trueY)
        err = sum((p_y - t_y) ** 2)
        return err



reg = Regression()


reg.leastSquaresReg()  # least_squares(LS)
reg.regularizedLSReg()   # regularized LS (RLS)
reg.robustReg()   # Robust Regression (RR)
reg.baysesianReg() # Baysesian Regression (BR)
reg.lassoReg()   # Lasso Regression (Lasso)

plt.show()

