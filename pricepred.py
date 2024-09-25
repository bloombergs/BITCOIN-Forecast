import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

df = pd.read_csv("BTC_All_graph_coinmarketcap.csv", sep=";")
df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601')
df = df.sort_values("timestamp").set_index("timestamp")

data = df[["close"]].values.astype(float)

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

normalized_data = normalize(data)

def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

seq_length = 30
X, y = create_sequences(normalized_data, seq_length)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.LSTM(50, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model.fit(X_train, y_train, epochs=50, batch_size=32)

predictions = model.predict(X_test)

def inverse_normalize(scaled_data, original_data):
    return scaled_data * (np.max(original_data) - np.min(original_data)) + np.min(original_data)

predictions_inverse = inverse_normalize(predictions, data)
y_test_inverse = inverse_normalize(y_test, data)

test_dates = df.index[-len(y_test):]  

plt.figure(figsize=(7, 4))
plt.plot(test_dates, y_test_inverse, label='Actual Prices', color='blue')
plt.plot(test_dates, predictions_inverse, label='Predicted Prices', color='red')
plt.title('BTC Price Prediction with LSTM')
plt.xlabel('Date')
plt.ylabel('BTC Price')
plt.legend()
plt.show()
