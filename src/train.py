import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler



path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv"
df = pd.read_csv(path)

le = LabelEncoder()
df['Manufacturer'] = le.fit_transform(df['Manufacturer'])
df['Price-binned'] = le.fit_transform(df['Price-binned']) 

X = df[['Manufacturer','Unnamed: 0','Price-binned','CPU_frequency','Category','GPU','OS','CPU_core','Weight_pounds','Screen_Size_inch','RAM_GB','Storage_GB_SSD','Screen-Full_HD','Screen-IPS_panel']]
Y = df['Price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

lm = LinearRegression()
lm.fit(X_train, y_train)

# Evaluate model
r2 = lm.score(X_test, y_test)
print("✅ Model trained successfully!")
print("R² Score on Test Data:", r2)
print("Approx Accuracy:", round(r2 * 100, 2), "%")

