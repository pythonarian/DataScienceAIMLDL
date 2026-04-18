import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load data
from sklearn.datasets import load_boston
data = load_boston()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['MEDV'] = data.target

# Log transform
df['MEDV_trans'] = np.log(df['MEDV'])

# Features & target
X = df.drop(columns=['MEDV', 'MEDV_trans'])
y = df['MEDV_trans']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model saved!")
