import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Dataset = pd.read_csv("abalone_weka_dataset.csv")
X = Dataset.iloc[:,:-1].values
y = Dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
ct = ColumnTransformer([("Geography", OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)

X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X, y , test_size=0.4, random_state = 0, shuffle = True )

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_features='sqrt')
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

