import os
import sys
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set PYTHONIOENCODING environment variable
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Redirect stdout and stderr to a file with utf-8 encoding
sys.stdout = open('C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python\\output.log', 'w', encoding='utf-8')
sys.stderr = sys.stdout

# Load the processed data
X_train = np.load('C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python\\X_train.npy', allow_pickle=True)
X_test = np.load('C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python\\X_test.npy', allow_pickle=True)
y_train = np.load('C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python\\y_train.npy', allow_pickle=True)
y_test = np.load('C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python\\y_test.npy', allow_pickle=True)

# Load the scaler used for training and fit it to the training data
scaler = MinMaxScaler()
scaler.fit(y_train.reshape(-1, 1))

try:
    # Check the value of PYTHONIOENCODING
    pythonioencoding = os.getenv('PYTHONIOENCODING')
    print(f"PYTHONIOENCODING: {pythonioencoding}")

    # Building the LSTM Model
    regressor = Sequential()
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    print("Starting model training...")
    history = regressor.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, shuffle=False)
    print("Model training completed.")

    # Save the model
    regressor.save('C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python\\lstm_model.h5')
    print("Model saved successfully.")

    # Make predictions
    print("Starting predictions...")
    y_pred = regressor.predict(X_test)
    
    # Inverse transform the predictions to original scale
    y_pred_original = scaler.inverse_transform(y_pred)
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

    print("Predictions completed.")

    # Ensure the directory exists before saving files
    os.makedirs('C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python', exist_ok=True)

    # Save the predictions and true values to files
    np.save('C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python\\y_pred.npy', y_pred_original)
    np.save('C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python\\y_test.npy', y_test_original)

    # Verify file creation
    print("y_pred.npy file created successfully.")

    # Generate dates for the forecast
    start_date = sys.argv[1]  # Example: pass '2024-01-01' as the start date
    forecast_days = 72  # Number of days to forecast
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    dates = [start_date + timedelta(days=i) for i in range(len(y_pred) + forecast_days)]
    dates = [date.strftime('%Y-%m-%d') for date in dates]

    # Save dates to a text file
    with open('C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python\\dates.txt', 'w') as f:
        for date in dates:
            f.write(f"{date}\n")

    # Prepare input data for future predictions
    last_known_data = X_test[-1]  # Use the last known data point as the starting point
    future_predictions = []

    for i in range(forecast_days):
        # Predict the next step
        next_pred = regressor.predict(last_known_data.reshape(1, X_test.shape[1], X_test.shape[2]))

        # Scale the predicted value back to original scale
        next_pred_original = scaler.inverse_transform(next_pred)

        # Append the prediction in original scale
        future_predictions.append(next_pred_original[0][0])

        # Update last_known_data by removing the oldest time step and appending the new prediction
        last_known_data = np.concatenate((last_known_data[1:], next_pred.reshape(1, -1)), axis=0)

    # Convert future_predictions to a 2D array
    future_predictions = np.array(future_predictions).reshape(-1, 1)

    # Save the combined predictions to a file
    combined_predictions = np.concatenate((y_pred_original, future_predictions))
    np.save('C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python\\combined_predictions.npy', combined_predictions)
    print("combined_predictions.npy file created successfully.")

except UnicodeEncodeError as e:
    print(f"UnicodeEncodeError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    sys.stdout.close()
