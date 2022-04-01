#importing packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

#code
df = pd.read_csv("fuel_data.csv")
X = df.filter(['drivenKM'])
y = df.filter(['fuelAmount'])
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train,y_train)

#dumping into pickle file
pickle.dump(model,open('model.pkl','wb'))
model= pickle.load(open('model.pkl','rb'))