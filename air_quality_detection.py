import pandas as pd
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
'exec(%matplotlib inline)'
import os



print(os.listdir(r"E:\\ML\\Simple Linear Regression\\Predicting air quality"))


PATH_CITY_DAY = "E:\\ML\\Simple Linear Regression\\Predicting air quality\\dataset\\city_day.csv" #variable storing the path of the dataset

df = pd.read_csv(PATH_CITY_DAY) # reading the dataset
print(df)
df =df[~df.isin([np.nan, np.inf, -np.inf]).any(1)] #helps to avoid the Value error caused by NaN values
x=df[["PM2.5", "PM10", "NOx", "NH3", "CO", "SO2", "O3"]] #independent values used for prediction
y=df[["AQI"]] #target variable

#filling all the NaN values as 0
x["PM2.5"] = x["PM2.5"].fillna(0)
x['PM10'] = x["PM10"].fillna(0)
x['NOx'] = x["NOx"].fillna(0)
x['NH3'] = x["NH3"].fillna(0)
x['CO'] = x["CO"].fillna(0)
x['SO2'] = x["SO2"].fillna(0)
x['O3'] = x["O3"].fillna(0)
x = x.values.astype(np.float64)
y = y.values.astype(np.float64)

#applying Multiple Regression

x = sm.add_constant(x) # adding a constant
 
model = sm.OLS(y, x).fit()
predictions = model.predict(x)  #prediction
 
print_model = model.summary() #stores the summary of the model
print(print_model) #prints the summary


