# -*- coding: utf-8 -*-
# @Time    : 10/4/17 10:43 AM
# @Author  : Jason Lin
# @File    : counting_people.py
# @Software: PyCharm Community Edition

import os
import numpy as np
import pandas as pd
from polynomial_function import Regression
import matplotlib.pylab as plt

class counting_people:

    def __init__(self):
        self.load_data()


    def load_data(self):
        os.getcwd()
        os.chdir(os.pardir)
        os.chdir("data")
        self.testX = np.loadtxt("count_data_testx.txt")
        self.testY = np.loadtxt("count_data_testy.txt")
        # print self.testX.shape
        self.trainX = np.loadtxt("count_data_trainx.txt")
        self.trainY = np.loadtxt("count_data_trainy.txt")
        # print self.trainX.shape


    def genOriginalPhi(self, inputX=None):

        # inputX is transposed
        if inputX is None:
            inputX = self.trainX

        matPhi = []
        for x in inputX:
            col = []
            for i in x:
                col.append(i)
            matPhi.append(col)
        matPhi = (np.asmatrix(matPhi))
        # print matPhi
        # print self.trainX.shape
        # print "Phi:" + str(matPhi.shape)
        return matPhi

    def gen2ndPolyPhi(self, inputX=None):

        # inputX is transposed
        if inputX is None:
            inputX = self.trainX

        matPhi_1 = self.genOriginalPhi(inputX)
        matPhi_2 = []
        for x in inputX:
            col = []
            for i in x:
                col.append(i**3)
            matPhi_2.append(col)
        matPhi_2 = (np.asmatrix(matPhi_2))
        matPhi = np.row_stack((matPhi_1, matPhi_2))
        # print matPhi
        # print self.trainX.shape
        # print "Phi:" + str(matPhi.shape)
        return matPhi

    def gen3rdPolyPhi(self, inputX=None):

        if inputX is None:
            inputX = self.trainX

        matPhi_1 = self.gen2ndPolyPhi(inputX)
        matPhi_2 = []
        for x in inputX:
            col = []
            for i in x:
                col.append(i**3)
            matPhi_2.append(col)
        matPhi_2 = (np.asmatrix(matPhi_2))
        matPhi = np.row_stack((matPhi_1, matPhi_2))

        return matPhi


    def predict(self, theta_, inputX, phi_type=1):

        if phi_type == 1:
            theta_ = np.array(theta_)
            theta__ = np.mat(theta_.flatten()).T  # change ND array into 1D array
            phi_ = self.genOriginalPhi(inputX=inputX.T)
            predictY = phi_.T * theta__
            return predictY

        if phi_type == 2:
            theta_ = np.array(theta_)
            theta_ = np.mat(theta_.flatten()).T  # change ND array into 1D array
            phi_ = self.gen2ndPolyPhi(inputX=inputX.T)
            predictY = phi_.T * theta_
            return predictY

        if phi_type == 3:
            theta_ = np.array(theta_)
            theta_ = np.mat(theta_.flatten()).T  # change ND array into 1D array
            phi_ = self.gen3rdPolyPhi(inputX=inputX.T)
            predictY = phi_.T * theta_
            return predictY


    def lsRegression(self):
        reg = Regression()
        reg.vecX = self.trainX
        reg.vecY = self.trainY.T
        theta_ = reg.leastSquaresReg(matPhi=self.gen2ndPolyPhi())
        # print theta_.shape
        predictY = self.predict(theta_, self.testX.T, phi_type=2)
        # print self.testX.shape

        xAxis = np.arange(len(self.testX.T))
        # print xAxis.shape
        fig = plt.figure()
        plt.scatter(xAxis, predictY, c="red", s=4, marker=".", label="Predicted Value")
        plt.scatter(xAxis, self.testY, s=4, marker="^", label="True Value", facecolors="none", c="blue")
        mse, mae = reg.countErr(predictY, self.testY)
        err_str = "MSE=" + str(mse) + "\nMAE=" + str(mae)
        plt.text(200, -12, err_str)
        plt.title("Least Square Regression ($\phi(x)=[x]$)")
        plt.legend(loc="best")
        fig.savefig("/Users/jieconlin3/Desktop/ls_cp_1phi.eps", format='eps', dpi=1000)
        # plt.show()

    def rlsRegression(self):
        reg = Regression()
        reg.vecX = self.trainX.T
        reg.vecY = self.trainY.T
        theta_ = reg.regularizedLSReg(matPhi=self.gen2ndPolyPhi())
        # print theta_.shape
        predictY = self.predict(theta_, self.testX.T, phi_type=2)
        # print self.testX.shape

        xAxis = np.arange(len(self.testX.T))
        # print xAxis.shape
        fig = plt.figure()
        plt.scatter(xAxis, predictY, c="red", s=4, marker=".", label="Predicted Value")
        plt.scatter(xAxis, self.testY, c="blue", s=4, marker="^", label="True Value")
        mse, mae = reg.countErr(predictY, self.testY)
        err_str = "MSE=" + str(mse) + "\nMAE=" + str(mae)
        plt.text(200, -12, err_str)
        plt.title("Regularized Least Square Regression ($\phi(x)=[x]$)")
        plt.legend(loc="best")
        fig.savefig("/Users/jieconlin3/Desktop/rls_cp_1phi.eps", format='eps', dpi=1000)
        # plt.show()


    def robustRegression(self):
        reg = Regression()
        reg.vecX = self.trainX.T
        reg.vecY = self.trainY.T
        theta_ = reg.robustReg(matPhi=self.gen2ndPolyPhi())
        # print theta_.shape
        predictY = self.predict(theta_, self.testX.T, phi_type=2)
        # print self.testX.shape

        xAxis = np.arange(len(self.testX.T))
        # print xAxis.shape
        fig = plt.figure()
        plt.scatter(xAxis, predictY, c="red", s=4, marker=".", label="Predicted Value")
        plt.scatter(xAxis, self.testY, c="blue", s=4, marker="^", label="True Value")
        mse, mae = reg.countErr(predictY, self.testY)
        err_str = "MSE=" + str(mse) + "\nMAE=" + str(mae)
        plt.text(200, -12, err_str)
        plt.title("Robust Regression ($\phi(x)=[x]$)")
        plt.legend(loc="best")
        fig.savefig("/Users/jieconlin3/Desktop/rr_cp_1phi.eps", format='eps', dpi=1000)
        # plt.show()

    def lassoRegression(self):

        reg = Regression()
        reg.vecX = self.trainX.T
        reg.vecY = self.trainY.T
        theta_ = reg.lassoReg(matPhi=self.gen2ndPolyPhi())
        # print theta_.shape
        predictY = self.predict(theta_, self.testX.T, phi_type=2)
        # print self.testX.shape

        xAxis = np.arange(len(self.testX.T))
        # print xAxis.shape
        fig = plt.figure()
        plt.scatter(xAxis, predictY, c="red", s=4, marker=".", label="Predicted Value")
        plt.scatter(xAxis, self.testY, c="blue", s=4, marker="^", label="True Value")

        mse, mae = reg.countErr(predictY, self.testY)
        err_str = "MSE=" + str(mse) + "\nMAE=" + str(mae)
        plt.text(200, -12, err_str)
        plt.title("Lasso Regression ($\phi(x)=[x]$)")
        plt.legend(loc="best")
        fig.savefig("/Users/jieconlin3/Desktop/lasso_cp_1phi.eps", format='eps', dpi=1000)
        # plt.show()

    def baysesianRegression(self):

        reg = Regression()
        reg.vecX = self.trainX
        reg.vecY = self.trainY.T

        mu_theta, sigma_theta = reg.baysesianReg(matPhi=self.gen2ndPolyPhi())
        phi_s = self.gen2ndPolyPhi(inputX=self.testX)

        predictY = phi_s.T * mu_theta
        xAxis = np.arange(len(self.testX.T))

        fig = plt.figure()
        plt.scatter(xAxis, predictY, c="red", s=4, marker=".", label="Predicted Value")
        plt.scatter(xAxis, self.testY, c="blue", s=4, marker="^", label="True Value")

        mse, mae = reg.countErr(predictY, self.testY)
        err_str = "MSE=" + str(mse) + "\nMAE=" + str(mae)
        plt.text(200, -12, err_str)
        plt.title("Bayessian Regression ($\phi(x)=[x]$)")
        plt.legend(loc="best")
        fig.savefig("/Users/jieconlin3/Desktop/bayessian_cp_1phi.eps", format='eps', dpi=1000)



def main():

    t = counting_people()
    t.lsRegression()
    t.rlsRegression()
    t.robustRegression()
    t.lassoRegression()
    t.baysesianRegression()

if __name__ == "__main__":
    main()
