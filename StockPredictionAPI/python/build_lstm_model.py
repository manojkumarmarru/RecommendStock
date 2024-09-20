import os
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from datetime import datetime, timedelta

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

try:
    # Check the value of PYTHONIOENCODING
    pythonioencoding = os.getenv('PYTHONIOENCODING')
    print(f"PYTHONIOENCODING: {pythonioencoding}")

    # Print the shape of the arrays
    print(f"Original X_train shape: {X_train.shape}")
    print(f"Original X_test shape: {X_test.shape}")

    # Building the LSTM Model
    lstm = Sequential()
    lstm.add(Input(shape=(1, X_train.shape[2])))
    lstm.add(LSTM(32, activation='relu', return_sequences=False))
    lstm.add(Dense(1))
    lstm.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    print("Starting model training...")
    history = lstm.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False)
    print("Model training completed.")

    # Save the model
    lstm.save('C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python\\lstm_model.h5')
    print("Model saved successfully.")

    # Make predictions
    print("Starting predictions...")
    y_pred = lstm.predict(X_test)
    print("Predictions completed.")

    # Print predictions for verification
    print(f"Predictions: {y_pred}")

    # Ensure the directory exists before saving files
    os.makedirs('C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python', exist_ok=True)

    # Save the predictions and true values to files
    np.save('C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python\\y_pred.npy', y_pred)
    np.save('C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python\\y_test.npy', y_test)

    # Verify file creation
    print("y_pred.npy file created successfully.")

    # Generate dates for the forecast
    start_date = sys.argv[1]
    forecast_days = int(sys.argv[2])  # Number of days to forecast
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    dates = [start_date + timedelta(days=i) for i in range(forecast_days)]
    dates = [date.strftime('%Y-%m-%d') for date in dates]

    # Save the dates to a file
    np.save('C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python\\dates.npy', np.array(dates))
    print("dates.npy file created successfully.")
except UnicodeEncodeError as e:
    print(f"UnicodeEncodeError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    sys.stdout.close()