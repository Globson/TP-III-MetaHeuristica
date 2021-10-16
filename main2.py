from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from numpy.random import seed, rand, randn
from numpy import dtype, sin, sqrt, asarray, mean, std
from Source.code import *
from sklearn import preprocessing
import numpy as np

#print(le_arq("./Data/telecom_users.csv"))
df = le_arq("./Data/telecom_users.csv")
le = preprocessing.LabelEncoder()
#print(df.head(10))

for i in df:
    if(dtype(df[i])=='object'):
        df[i] = le.fit_transform(df[i])
#print(df.head(10))

X_train, X_test, Y_train, Y_test = train_test_split(df,df['Churn'], random_state=42, test_size=0.3)


