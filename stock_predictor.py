import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle  # For saving the model

# Fetch stock data
ticker = "AAPL"  # Change to any stock symbol
stock_data = yf.download(ticker, start="2019-01-01", end="2024-01-01")
stock_data = stock_data[['Close']]

# Shift data to predict the next day's price
stock_data['Future Price'] = stock_data['Close'].shift(-1)
stock_data.dropna(inplace=True)

# Define features and labels
X = stock_data[['Close']]
y = stock_data['Future Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model
with open("stock_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as stock_model.pkl")
