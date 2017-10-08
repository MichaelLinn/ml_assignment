# -*- coding: utf-8 -*-
# @Time    : 10/1/17 5:22 PM
# @Author  : Jason Lin
# @File    : polynomial_function.py
# @Software: PyCharm Community Edition

import numpy as np
import os
import matplotlib.pylab as plt
from cvxopt import matrix, solvers
import random


class Regression:

    polyOrder = 5
    rr_error = []
    ls_error =[]

    def __init__(self):
        plt.figure(1)
        self.load_dataMat()

    def load_dataMat(self):
        os.getcwd()
        os.chdir(os.pardir)
        os.chdir("data")

        self.vecX = np.loadtxt("polydata_data_sampx.txt")
        self.vecY = np.loadtxt("polydata_data_sampy.txt")

        self.polyx = np.loadtxt("polydata_data_polyx.txt")
        self.polyy = np.loadtxt("polydata_data_polyy.txt")


    # least squares regression (LS)
    def leastSquaresReg(self, matPhi = None):
        if matPhi is None:
            matPhi = self.genPhi()
        theta_LS = (matPhi * matPhi.T).I * matPhi * np.mat(self.vecY).T
        # print theta_LS
        # plt.subplot(321, xlabel="Least Squares")
        # self.plotReg(theta_LS)
        return theta_LS

    # regularized LS (RLS)
    def regularizedLSReg(self, lambda_=0.1, matPhi=None):
        # lambda_ = 0.1
        if matPhi is None:
            matPhi = self.genPhi()
        matUnit = np.identity((matPhi * matPhi.T).shape[0])
        theta_RLS = (matPhi * matPhi.T + lambda_ * matUnit).I * matPhi * np.mat(self.vecY).T
        # plt.subplot(322, xlabel="Regularized")
        # self.plotReg(theta_RLS)

        return theta_RLS


    def plotReg(self, theta_):

        testX = np.linspace(-2.0, 2.0, 200, endpoint=True)

        theta_ = np.array(theta_[::-1])  # reverse
        theta_ = theta_.flatten()     # change ND array into 1D array
        p = np.poly1d(theta_)
        resY = p(testX)

        predictY = p(self.polyx)

        error = self.countErr(predictY, self.polyy)
        text = "Error=" + str(error)
        plt.text(-0.5, 25, text)

        plt.plot(testX, resY, label="Regression")
        plt.scatter(self.vecX, self.vecY, marker="o", s=9, alpha=0.5, c="black", label="Sample point")
        plt.scatter(self.polyx, self.polyy, marker="o", c="red",s=7, label="True Point")
        plt.legend(loc="best")

    def predict(self, theta_, inputX):

        theta_ = np.array(theta_[::-1])  # reverse
        theta_ = theta_.flatten()  # change ND array into 1D array
        p = np.poly1d(theta_)
        predictY = p(inputX)
        return predictY


    def genPhi(self, inputX=None):

        if inputX is None:
            inputX = self.vecX
        theta_len = self.polyOrder + 1
        f = lambda x, n: x ** n
        matPhi = []
        for x in inputX:
            col = []
            for i in range(theta_len):
                col.append(f(x, i))
            matPhi.append(col)
        matPhi = (np.asmatrix(matPhi)).T
        # print matPhi
        self.matPhi = matPhi
        return matPhi

    def robustReg(self, matPhi=None):


        if matPhi is None:
            matPhi = self.genPhi()
            theta_len = self.polyOrder + 1
        else:
            theta_len = len(matPhi)

        vecF = []
        for i in range(theta_len):
            vecF.append(0.0)
        for i in range(len(self.vecX)):
            vecF.append(1.0)
        vecF = np.array(vecF)


        matIdentity = np.identity(len(self.vecX))

        matA_1 = np.column_stack((-matPhi.T, -matIdentity))
        matA_2 = np.column_stack((matPhi.T, -matIdentity))
        matA = np.row_stack((matA_1, matA_2))
        vec_b = np.concatenate((-self.vecY, self.vecY))

        # objectiveFunc = lambda X: vecF.T * X
        # consFunc = lambda X: -(matA * X - vec_b)
        # res = linprog(vecF, A_ub=matA, b_ub=vec_b)
        # theta_ = res.x[:6]
        sol = solvers.lp(matrix(vecF), matrix(matA), matrix(vec_b), solver=solvers)
        theta_ =  list(sol['x'][:theta_len])
        print theta_

        # plt.subplot(323, xlabel="Robust")
        # self.plotReg(theta_)

        return theta_


    def baysesianReg(self, alpha_=1.0, matPhi=None):

        if matPhi is None :
            matPhi = self.genPhi()
            theta_len = self.polyOrder + 1
        else:
            theta_len = len(matPhi)

        noise = 5

        sigma_theta = (1.0/alpha_*np.identity(theta_len) + 1.0/ 5 * matPhi * matPhi.T).I
        mu_theta = 1.0/noise * sigma_theta * matPhi * np.mat(self.vecY).T

        # plot regression
        testX = np.linspace(-2.0, 2.0, 200, endpoint=True)

        f = lambda x, n: x ** n
        resY = []
        varErr = []
        for x in testX:
            phi_s = []
            for i in range(theta_len):
                phi_s.append(f(x,i))
            phi_s = np.mat(phi_s)
            mu_s = phi_s * mu_theta
            sigma_s = phi_s * sigma_theta * phi_s.T
            varErr.append(np.sqrt(sigma_s.tolist()[0][0]))
            resY.append(mu_s.tolist()[0][0])


        # predict
        predictY = []
        for x in self.polyx:
            phi_s = []
            for i in range(theta_len):
                phi_s.append(f(x, i))
            phi_s = np.mat(phi_s)
            mu_s = phi_s * mu_theta
            # sigma_s = phi_s * sigma_theta * phi_s.T
            predictY.append(mu_s.tolist()[0][0])

        err, _ = self.countErr(predictY, self.polyy)

        return mu_theta, sigma_theta

        # text = "Error=" + str(err)

        # plt.subplot(324, xlabel="Baysesian")
        # plt.plot(testX, resY)
        # plt.errorbar(testX, resY, yerr=varErr, ecolor="red")
        # plt.text(-0.5, 25, text)

        # plt.scatter(self.vecX, self.vecY, marker="o", s=9, alpha=0.5, c="black")


    def lassoReg(self, matPhi=None):
        lambda_ = 1.0
        if matPhi is None:
            matPhi = self.matPhi
            theta_len = self.polyOrder + 1
        else:
            theta_len = len(matPhi)

        x_dim = 2 * theta_len

        matH_1 = np.column_stack((matPhi*matPhi.T, -matPhi*matPhi.T))
        matH_2 = np.column_stack((-matPhi*matPhi.T, matPhi*matPhi.T))
        matH = np.row_stack((matH_1, matH_2))

        mat_func = lambda_ * np.mat(np.ones(x_dim)).T - np.row_stack((matPhi* (np.mat(self.vecY)).T, -matPhi* (np.mat(self.vecY).T)))

        matG = np.identity(x_dim)
        vec_h = np.zeros(x_dim)

        sol = solvers.qp(matrix(matH), matrix(mat_func), matrix(matG), matrix(vec_h))
        # print sol['x']

        theta_ = list(sol['x'][:theta_len] - sol['x'][theta_len:])
        # print theta_
        # plt.subplot(325, xlabel="Lasso")
        # self.plotReg(theta_)
        return theta_

    def countErr(self, preY, trueY):

        p_y = np.array(preY).flatten()
        t_y = np.array(trueY).flatten()
        MSE = sum((p_y - t_y) ** 2)/len(preY)
        MAE = sum(abs(p_y - t_y))/len(preY)

        print MSE
        print "-----------------"
        print MAE
        return MSE, MAE



    def genSubSetError(self):

        set_idx = np.arange(len(self.vecX))

        error = {}
        error['LS'] = []
        error['RR'] = []
        error['Lasso'] = []
        error['RLS'] = []
        errorX = []
        for size in range(14, 100, 2):
            subset_size = size * 0.01 * len(self.vecX)
            errorX.append(size / 100.0)
            subset_idx = random.sample(set_idx, int(subset_size))
            subsetX = []
            subsetY = []
            temErr = {}
            temErr['LS'] = []
            temErr['RR'] = []
            temErr['Lasso'] = []
            temErr['RLS'] = []
            temError = []
            for i in range(300):
                print "---------------" + str(i) + "---------------"
                for idx in subset_idx:
                    subsetX.append(self.vecX[idx])
                    subsetY.append(self.vecY[idx])
                reg_model = Regression()
                reg_model.vecX = np.array(subsetX)
                reg_model.vecY = np.array(subsetY)
                # Least Square Regression
                theta_ = reg_model.leastSquaresReg()
                predicY = reg_model.predict(theta_, self.polyx)
                err, _ = reg_model.countErr(predicY, self.polyy)
                temErr['LS'].append(err)
                # Robust Regression
                theta_ = reg_model.robustReg()
                predicY = reg_model.predict(theta_, self.polyx)
                err, _ = reg_model.countErr(predicY, self.polyy)
                temErr['RR'].append(err)
                # Regularized Least Square Regression
                theta_ = reg_model.regularizedLSReg()
                predicY = reg_model.predict(theta_, self.polyx)
                err, _ = reg_model.countErr(predicY, self.polyy)
                temErr['RLS'].append(err)
                # Lasso
                theta_ = reg_model.lassoReg()
                predicY = reg_model.predict(theta_, self.polyx)
                err, _ = reg_model.countErr(predicY, self.polyy)
                temErr['Lasso'].append(err)

            error['LS'].append(np.mean(np.array(temErr['LS'])))
            error['RR'].append(np.mean(np.array(temErr['RR'])))
            error['RLS'].append(np.mean(np.array(temErr['RLS'])))
            error['Lasso'].append(np.mean(np.array(temErr['Lasso'])))

        self.plotError(errorX, error['LS'], "LS")
        self.plotError(errorX, error['RR'], "RR")
        self.plotError(errorX, error['RLS'], "RLS")
        self.plotError(errorX, error['RLS'], "RLS")


    def plotError(self, inputX, inputY, name):
        print inputX
        print inputY
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(inputX, inputY, s=8, c="red")
        ax.ylabel("Mean Square Error")
        ax.xlabel("Size of Subset")
        ax.title("Least Square Regression")
        ax.plot(inputX, inputY)
        ax.legend()
        fig.savefig()
        # ax.show()

        return 0

    def run(self):
        reg = Regression()
        reg.genSubSetError()


"""
reg = Regression()
reg.leastSquaresReg()  # least_squares(LS)f
reg.regularizedLSReg()  # regularized LS (RLS)
reg.robustReg()   # Robust Regression (RR)
reg.baysesianReg() # Baysesian Regression (BR)
reg.lassoReg()   # Lasso Regression (Lasso)
plt.show()
"""
# reg = Regression()
# reg.baysesianReg() # Baysesian Regression (BR)