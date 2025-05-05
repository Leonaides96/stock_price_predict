import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # To checking the machine are obtain GPU or CPU


ticker_id = "AAPL" # APPLE stock tickets

df = yf.download(ticker_id, "2020-01-01")

# having the closing price of the stock
closing_price = df["Close"]
closing_price.plot(figsize=(12,8)) # plotting


# scaling the values
scaler = StandardScaler()
df['Close'] = scaler.fit_transform(df['Close'])

# 
seq_len = 30
data_list = []

for i in range(len(df)-seq_len):
    data_list.append(df.loc[i:i+seq_len, "Close"])

data = np.array(data_list)

train_size = int(0.8 * len(data))

X_train = torch.from_numpy(data[:train_size, :-1, : ])