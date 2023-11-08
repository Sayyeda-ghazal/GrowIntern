import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


data = pd.read_csv('/content/drive/MyDrive/ghazal/programming/Artificial Intelligence/machine learning/GrowIntern/Stock Price of Top 10 SmartPhone Company (2016-2021)/Alcatel Lucent/Alcatel Lucent.csv')

features = data.drop(['Date', 'Adj Close'], axis=1)
target = data['Adj Close']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

plt.figure(figsize=(12, 6))
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs. Predicted Stock Prices')
plt.show()
