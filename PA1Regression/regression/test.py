# -*- coding: utf-8 -*-
# @Time    : 10/1/17 5:22 PM
# @Author  : Jason Lin
# @File    : test.py
# @Software: PyCharm Community Edition

import pandas as pd
import numpy as np
import os
from pylab import *


class Regression:

    polyOrder = 5

    def __init__(self):
        self.load_dataMat()

    def load_dataMat(self):
        os.getcwd()
        os.chdir(os.pardir)
        os.chdir("data")

        self.vecX = np.loadtxt("polydata_data_sampx.txt")
        self.vecY = np.loadtxt("polydata_data_sampy.txt")

        # scatter(self.vecX, self.vecY, marker="o", s=9, alpha=0.5, c="black")
        # show()

    def leastSquaresReg(self):
        matPhi = self.genPhi()
        theta_LS = (matPhi * matPhi.T).I * matPhi * np.mat(self.vecY).T
        # print theta_LS
        self.plotLSReg(theta_LS)


    def plotLSReg(self, theta_LS):

        testX = np.linspace(-2.0, 2.0, 200, endpoint=True)

        # print theta_LS
        theta_LS = np.array(theta_LS[::-1])  # reverse
        theta_LS = theta_LS.flatten()     # change ND array into 1D array
        p = np.poly1d(theta_LS)
        resY = p(testX)

        plot(testX, resY)
        scatter(self.vecX, self.vecY, marker="o", s=9, alpha=0.5, c="black")
        show()


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



reg = Regression()
# least_squares(LS)
reg.leastSquaresReg()


