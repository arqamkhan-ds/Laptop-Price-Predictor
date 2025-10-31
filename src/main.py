import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv"
df = pd.read_csv(path)

print(df.head())  # See first few rows

# ✅ Create Linear Regression model
lm = LinearRegression()
print("LinearRegression model initialized successfully.")

# Create encoder object
le = LabelEncoder()

# Convert text labels to numbers
df['Manufacturer'] = le.fit_transform(df['Manufacturer']) 
df['Price-binned'] = le.fit_transform(df['Price-binned']) 

#print("The first 5 rows of the dataframe") 
#print(df.head(5))


X = df[['Manufacturer','Unnamed: 0','Price-binned','CPU_frequency','Category','GPU','OS','CPU_core','Weight_pounds','Screen_Size_inch','RAM_GB','Storage_GB_SSD','Screen-Full_HD','Screen-IPS_panel']]
Y = df['Price']


lm.fit(X, Y)
Yhat = lm.predict(X)

# ✅ Print output explicitly
print("Predicted prices (first 5):", Yhat[0:5])
print("Intercept (b0):", lm.intercept_)
print("Coefficient (b1):", lm.coef_)

r2 = lm.score(X, Y)
print("R² score:", r2)
print("Approx Accuracy:", round(r2 * 100, 2), "%")