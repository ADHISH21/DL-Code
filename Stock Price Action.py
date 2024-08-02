#importing needed libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.model_selection import 
train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error # Import mean_squared_error
# Load the dataset
df=pd.read_csv('/content/sorted_merged_stocks_cap_2.csv') # Replace 'your_data.csv' with the path to your CSV file
print(df)

# Select data for the 'INF' stock
inf_data = df[df['symbol'] == "INF"]

# Extract the 'close' prices as the target variable
prices = inf_data['close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# Split the data into training and testing sets
train_size = int(len(prices_scaled) * 0.8)
test_size = len(prices_scaled) - train_size
train_data, test_data=prices_scaled[0:train_size], 
prices_scaled[train_size:len(prices_scaled)]

# Function to create dataset with look back
def create_dataset(dataset, look_back=1):
  X, Y = [], []
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i+look_back), 0]
    X.append(a)
    Y.append(dataset[i + look_back, 0])
return np.array(X), np.array(Y)

# Create the dataset with look back
look_back = 100 # Adjust this parameter as needed
train_X, train_Y = create_dataset(train_data, look_back)
test_X, test_Y = create_dataset(test_data, look_back) 

# Reshape input to be [samples, time steps, features]
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, 
input_shape=(1, look_back)))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
print(model)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
model.fit(train_X, train_Y, epochs=100, batch_size=64, verbose=1)
# Make predictions
train_predictions = model.predict(train_X)
test_predictions = model.predict(test_X)
# Invert predictions
train_predictions = scaler.inverse_transform(train_predictions)
train_Y = scaler.inverse_transform([train_Y])
test_predictions = scaler.inverse_transform(test_predictions)
test_Y = scaler.inverse_transform([test_Y])

• # Calculate root mean squared error
train_score = np.sqrt(mean_squared_error(train_Y[0], train_predictions[:,0]))
print('Train RMSE:', train_score)
test_score = np.sqrt(mean_squared_error(test_Y[0], test_predictions[:,0]))
print('Test RMSE:', test_score)

# Plot the actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.plot(train_predictions, label='Train Predictions')
plt.plot(test_predictions, label='Test Predictions')
plt.plot(prices, label='Actual Prices')
plt.title('INF Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()



