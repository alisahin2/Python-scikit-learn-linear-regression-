# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:46:59 2020

@author: Ali
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("lineer_regression_dataset.csv")

plt.scatter(df.deneyim, df.maas )
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()

#%%linear regression

#sklearn library
from sklearn.linear_model import LinearRegression

#linear regression model
linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1) #type(x) = pandas.core.series.Series(we are converted the numpy array)
y = df.maas.values.reshape(-1,1) 

#now we are fitting
linear_reg.fit(x,y)

#prediction
b0 = linear_reg.predict(np.array([0,0]).reshape(-1,1))
print("b0: ", b0)

b0_ = linear_reg.intercept_
print("b0_: ", b0_) # y-eks intercept point

b1 = linear_reg.coef_
print("b1: ", b1) # slope



tenYearsEmployeeSalary = 1663 + 1138*10 # wrong way
print("salary: " , tenYearsEmployeeSalary)

#true way
print(linear_reg.predict(np.array([10,0]).reshape(-1,1)))# The salary of the employee with 10 years of experience

#visualize line

array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)

plt.scatter(x,y)

y_head = linear_reg.predict(array)

plt.plot(array, y_head, color = "red")
plt.show()














































