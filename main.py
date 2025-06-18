
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

ticker_id = "AAPL" # APPLE stock tickets

df = yf.download(ticker_id, "2020-01-01")

# having the closing price of the stock
closing_price = df["Close"]
closing_price.plot(figsize=(12,8)) # plotting


# Preprocessing

## scaling the values
scaler = StandardScaler()
df['Close'] = scaler.fit_transform(df['Close'])

# Data split the dataset by different seq
seq_len = 30
data_list = []

for i in range(len(df)-seq_len):
    data_list.append(df.loc[i:i+seq_len, "Close"])

data = np.array(data_list)

# Get the train and test, optionally we can using the sklearn train test split by not shuffering it
train_size = int(0.8 * len(data)) # get the train of the dataset (by 80% the timepoint as the dateaset)

X_train = torch.from_numpy(data[:train_size, :-1, :]).type(torch.Tensor).to(DEVICE)
y_train = torch.from_numpy(data[:train_size, -1, :]).type(torch.Tensor).to(DEVICE)
X_test = torch.from_numpy(data[train_size:, :-1, :]).type(torch.Tensor).to(DEVICE)
y_test = torch.from_numpy(data[train_size:, -1, :]).type(torch.Tensor).to(DEVICE)

# Prediction for train data which is seen data
from predictors import PredictionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # To checking the machine are obtain GPU or CPU
model = PredictionModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 200

for i in range(num_epochs):
    y_train_pred = model(X_train)
    
    loss = criterion(y_train_pred, y_train) # metrics of the accurancy model

    if i % 25 ==0:
        print(i, loss.items())


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Prediction for the unseen data
model.eval()

y_test_pred = model(X_test)

# inverse the scalar for the original base > price base
y_train_pred = scaler.inverse_transform(y_train_pred.detach().cpu().numpy())
y_train = scaler.inverse_transform(y_train.detach().cpu().numpy())

y_test_pred = scaler.inverse_transform(y_test_pred.detach().cpu().numpy())
y_test = scaler.inverse_transform(y_test.detach().cpu().numpy())

# have the metrics error
train_rmse = root_mean_squared_error(y_train[:, 0], y_train_pred[:, 0]) # need to be the numpy array
test_rmse = root_mean_squared_error(y_test[:, 0], y_test_pred[:, 0]) # need to be the numpy array

print(f"RMSE_train: {train_rmse}")
print(f"RMSE_test: {test_rmse}")


# ploting on the unseen dateset part
## ploting on the Actual vs Prediction
fig = plt.figure(figsize=(12,10))
gs = fig.add_gridspec(4,1)

ax1 = fig.add_subplot(gs[:3, 9])
ax1.plot(df.iloc[-len(y_test):].index, y_test, color="blue", label = "Actual Price")
ax1.plot(df.iloc[-len(y_test):].index, y_test_pred, color="green", label = "Prediction")
ax1.legend()
plt.title(f"Stock price prediction - stock:{ticker_id}")
plt.xlabel("Date")
plt.ylabel("Value")

## Plotting on the error metrics
ax2 = fig.add_subplot(gs[3,0])
ax2.axline(test_rmse, color = "blue", linestyle="--", labeL = "RMSE")
ax2.plot(df[-len(y_test):].index, abs(y_test-y_test_pred), "r", label="Prediction error")
ax2.legend()
plt.title(f"Prediction Error")
plt.xlabel("Date")
plt.ylabel("Value")

plt.tight_layout()