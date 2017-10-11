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
    def regularizedLSReg(self, lambda_=0.5, matPhi=None):
        # lambda_ = 0.1
        if matPhi is None:
            matPhi = self.genPhi()
        matUnit = np.identity((matPhi * matPhi.T).shape[0])
        theta_RLS = (matPhi * matPhi.T + lambda_ * matUnit).I * matPhi * np.mat(self.vecY).T
        # plt.subplot(322, xlabel="Regularized")
        # self.plotReg(theta_RLS)

        return theta_RLS


    def plotReg(self, theta_, name, mu_theta=None, sigma_theta=None, idx=""):

        testX = np.linspace(-2.0, 2.0, 200, endpoint=True)
        fig = plt.figure()

        if name == "Bayesian Regression":
            predictY, deviation = self.predict_bayes(mu_theta, sigma_theta, self.polyx)
            resY, _ = self.predict_bayes(mu_theta, sigma_theta, testX)
            predictY = np.array(predictY).flatten()
            plt.errorbar(self.polyx, predictY, yerr=deviation, ecolor="red", label="Predicted Point", fmt="-o", ms=2, c="black", mfc="green")
            l = "lower left"

        else:
            theta_ = np.array(theta_[::-1])  # reverse
            theta_ = theta_.flatten()     # change ND array into 1D array
            p = np.poly1d(theta_)
            resY = p(testX)
            predictY = p(self.polyx)
            plt.scatter(self.polyx, predictY, marker="s", c="g", s=3, label="Predicted Point")
            l = "upper right"

        error, _ = self.countErr(predictY, self.polyy)
        text = "MeanSquareError=" + str(error)


        plt.text(-0.5, -20, text)
        name0 = name + " (" + str(idx) + "% of samples)"
        plt.title(name0)
        plt.plot(testX, resY, label="Regression Line", linewidth=1, c="black")
        plt.scatter(self.vecX, self.vecY, marker="o", s=3, alpha=0.5, c="blue", label="Sample point")
        plt.scatter(self.polyx, self.polyy, marker="^", c="r",s=3, label="True Point")


        plt.legend(loc=l)
        filename = "/Users/jieconlin3/Desktop/" + name.replace(" ", "") + "_outliers" + str(idx) + ".eps"
        fig.savefig(filename, format='eps', dpi=1000)


    def predict_bayes(self, mu_theta, sigma_theta, inputX):

        phi_s = self.genPhi(inputX=inputX)
        predictY = phi_s.T * mu_theta
        deviation = phi_s.T * sigma_theta * phi_s
        return predictY, deviation


    def predict(self, theta_, inputX=None):
        if inputX is None:
            inputX = self.polyx
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
        # self.matPhi = matPhi
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
        # print theta_

        # plt.subplot(323, xlabel="Robust")
        # self.plotReg(theta_)

        return theta_


    def baysesianReg(self, alpha_=10.4, matPhi=None):

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


    def lassoReg(self, matPhi=None, lambda_=2.1):

        if matPhi is None:
            matPhi = self.genPhi()
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

        # print MSE
        # print "-----------------"
        # print MAE
        return MSE, MAE



    def genSubSetError(self):

        set_idx = np.arange(len(self.vecX))

        error = {}
        error['LS'] = []
        error['RR'] = []
        error['Lasso'] = []
        error['RLS'] = []
        error['Bayes'] = []
        errorX = []
        for size in range(12, 100, 10):
            subset_size = size * 0.01 * len(self.vecX)
            errorX.append(size / 100.0 * len(self.vecX))


            temErr = {}
            temErr['LS'] = []
            temErr['RR'] = []
            temErr['Lasso'] = []
            temErr['RLS'] = []
            temErr['Bayes'] = []


            for i in range(1):
                print size, "---------------" + str(i) + "---------------"
                subsetX = []
                subsetY = []
                subset_idx = random.sample(set_idx, int(subset_size))
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


                # Bayessian
                mu_theta, sigma = reg_model.baysesianReg()
                phi_s = self.genPhi(inputX=self.polyx)
                predictY = phi_s.T * mu_theta
                err, _ = reg_model.countErr(predictY, self.polyy)
                temErr['Bayes'].append(err)


            error['LS'].append(np.mean(np.array(temErr['LS'])))
            error['RR'].append(np.mean(np.array(temErr['RR'])))
            error['RLS'].append(np.mean(np.array(temErr['RLS'])))
            error['Lasso'].append(np.mean(np.array(temErr['Lasso'])))
            error['Bayes'].append(np.mean(np.array(temErr['Bayes'])))

        self.plotError(errorX, error['LS'], "Least Squares Regression")
        self.plotError(errorX, error['RR'], "Robust Regression")
        self.plotError(errorX, error['RLS'], "Regularized LS Regression")
        self.plotError(errorX, error['RLS'], "Lasso Regression")
        self.plotError(errorX, error['Bayes'], "Bayessian Regression")


    def plotError(self, inputX, inputY, name):

        fig = plt.figure()
        # ax = fig.add_subplot(111)
        plt.scatter(inputX, inputY, s=8, c="red")
        plt.ylabel("Mean Square Error")
        plt.xlabel("Size of Subset")
        plt.title(name)
        plt.plot(inputX, inputY)
        plt.legend()
        # plt.show()
        filename = "/Users/jieconlin3/Desktop/" + name.replace(" ", "") + "subseterror.eps"
        fig.savefig(filename, format='eps', dpi=1000)
        return 0

    def run(self):
        reg = Regression()
        reg.genSubSetError()

    def plotRrgression(self, name=""):

        theta_ = self.leastSquaresReg()
        self.plotReg(theta_, "Least Square Regression", idx=name)
        theta_ = self.regularizedLSReg()
        self.plotReg(theta_, "Regularized LS Regression", idx=name)
        theta_ = self.robustReg()
        self.plotReg(theta_, "Robust Regression", idx=name)
        theta_ = self.lassoReg()
        self.plotReg(theta_, "Lasso Regression", idx=name)

        mu_theta, sigma_theta = self.baysesianReg()
        self.plotReg(theta_=None, mu_theta=mu_theta, sigma_theta=sigma_theta, name="Bayesian Regression", idx=name)


    def addOutliers(self):

        y_idx = np.arange(len(self.vecY))
        num_outlier = int(0.15 * len(self.vecY))
        outlier_idx = random.sample(y_idx, num_outlier)
        for idx in outlier_idx:
            self.vecY[idx] = self.vecY[idx] + (-1) ** int(random.randint(1, 10)) * random.randrange(20, 30)

    def genSubSet(self, size):

        x_idx = np.arange(len(self.vecX))
        sub_idx = random.sample(x_idx, int(size))
        new_vecX = []
        new_vecY = []
        for idx in sub_idx:
            new_vecX.append(self.vecX[idx])
            new_vecY.append(self.vecY[idx])
        self.vecX = np.array(new_vecX)
        self.vecY = np.array(new_vecY)

    def getBestLambda(self):

        reg = Regression()
        # choose hyperparameters
        l = np.arange(-1.0, 3.0, 0.1)
        MSE_ls = []
        MAE_ls = []
        MSE_lasso = []
        lsE_best = np.inf
        lassoE_best = np.inf
        ls_lambda = 0
        lasso_lambda = 0
        for lambda_ in l:
            if lambda_ == 0:
                continue
            theta_ = reg.regularizedLSReg(lambda_=lambda_)
            predictY = reg.predict(theta_)
            mse, mae = reg.countErr(predictY, reg.polyy)
            MSE_ls.append(mse)
            MAE_ls.append(mae)

            if mse < lsE_best:
                lsE_best = mse
                ls_lambda = lambda_

            theta_ = reg.lassoReg(lambda_=lambda_)
            predictY = reg.predict(theta_)
            mse, mae = reg.countErr(predictY, reg.polyy)
            MSE_lasso.append(mse)
            if mse < lassoE_best:
                lassoE_best = mse
                lasso_lambda = lambda_

        fig = plt.figure()
        plt.title("Regularized LS Regression")
        plt.xlabel("$\lambda$")
        plt.ylabel("Mean Square Error")
        plt.text(0.5, 0.44, ("Best $\lambda$ is " + str(ls_lambda)))
        plt.scatter(l, MSE_ls, label="MSE", s=5, c="r")
        # plt.scatter(l, MAE, label="MAE")
        plt.plot(l, MSE_ls)
        # plt.plot(l, MAE)
        plt.legend()
        fig.savefig("/Users/jieconlin3/Desktop/ls_param.eps", format='eps', dpi=1000)

        fig = plt.figure()
        plt.title("Lasso Regression")
        plt.xlabel("$\lambda$")
        plt.ylabel("Mean Square Error")
        plt.scatter(l, MSE_lasso, label="MSE", s=5, c="r")
        # plt.scatter(l, MAE, label="MAE")
        plt.plot(l, MSE_lasso)
        plt.text(0.5, 0.44, ("Best $\lambda$ is " + str(lasso_lambda)))
        # plt.plot(l, MAE)
        plt.legend()
        fig.savefig("/Users/jieconlin3/Desktop/lasso_param.eps", format='eps', dpi=1000)

    def getBsetParamBay(self):
        reg = Regression()

        # get param for bayessian
        MSE_b = []
        bamse_best = np.inf
        bay_alpha = 0

        alpha_ = np.arange(0.1, 20.0, 0.1)
        for a in alpha_:
            if a == 0:
                continue

            mu_theta, _ = reg.baysesianReg(alpha_=a)
            predictY = reg.genPhi(reg.polyx).T * mu_theta
            mse, mae = reg.countErr(predictY, reg.polyy)
            MSE_b.append(mse)

            if mse < bamse_best:
                bamse_best = mse
                bay_alpha = a

        fig = plt.figure()
        plt.title("Bayesian Regression")
        plt.xlabel("$\lambda$")
        plt.ylabel("Mean Square Error")
        plt.plot(alpha_, MSE_b)
        plt.scatter(alpha_, MSE_b, label="MSE", s=3, c="r")

        plt.text(10, 0.6, ("Best $\\alpha$ is " + str(bay_alpha)))
        plt.legend()
        fig.savefig("/Users/jieconlin3/Desktop/bayesian_param.eps", format='eps', dpi=1000)


def main():

    r = Regression()
    l = len(r.vecX)
    for i in range(15, 100 ,10):
        size = int(i * 0.01 * l)
        reg = Regression()
        reg.genSubSet(size)
        reg.plotRrgression(name=str(i))



if __name__ == "__main__":
    main()




"""
reg = Regression()
reg.leastSquaresReg()  # least_squares(LS)f
reg.regularizedLSReg()  # regularized LS (RLS)
reg.robustReg()   # Robust Regression (RR)
reg.baysesianReg() # Baysesian Regression (BR)
reg.lassoReg()   # Lasso Regression (Lasso)
plt.show()
reg = Regression()
reg.genSubSetError()
"""
