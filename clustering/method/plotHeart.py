# -*- coding: utf-8 -*-
# @Time    : 29/10/2017 10:23 PM
# @Author  : Jason Lin
# @File    : plotHeart.py
# @Software: PyCharm Community Edition

"""
import numpy as np
import matplotlib.pyplot as plt
T = np.linspace(0 , 2 * np.pi, 1024)
fig = plt.figure()
plt.axes(polar = True)
plt.plot(T, 1. - np.sin(T),color="r")
plt.show()
"""
import matplotlib.pyplot as plt
import numpy as np
t = np.arange(0,2*np.pi, 0.1)
x = 16*np.sin(t)**3
y = 13*np.cos(t)-5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t)
plt.plot(x,y,color = 'red')
plt.title("To My Darling!")
plt.show()

