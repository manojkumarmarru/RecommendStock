import os
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

# Normalize the data
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_train = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

# Debugging: Print shapes and first few values of data
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Print the first 5 rows of X_train in a readable format
print("First 5 rows of X_train:")
for i in range(5):
    print([f"{x:.2f}" for x in X_train[i][0]])

try:
    # Check the value of PYTHONIOENCODING
    pythonioencoding = os.getenv('PYTHONIOENCODING')
    print(f"PYTHONIOENCODING: {pythonioencoding}")

    # Building the LSTM Model with increased complexity
    regressor = Sequential()
    regressor.add(LSTM(units=256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    regressor.add(Dropout(0.3))
    regressor.add(LSTM(units=256, return_sequences=True))
    regressor.add(Dropout(0.3))
    regressor.add(LSTM(units=256, return_sequences=True))
    regressor.add(Dropout(0.3))
    regressor.add(LSTM(units=256))
    regressor.add(Dropout(0.3))
    regressor.add(Dense(units=1))
    optimizer = Adam(learning_rate=0.001)
    regressor.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model
    print("Starting model training...")
    history = regressor.fit(X_train, y_train, epochs=500, batch_size=32, verbose=1, shuffle=False)
    print("Model training completed.")

    # Save the model in native Keras format
    regressor.save('C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python\\lstm_model.keras')
    print("Model saved successfully.")

    # Make predictions
    print("Starting predictions...")
    y_pred = regressor.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
    y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    
    # Debugging: Print some predictions
    print(f"First 10 predictions: {y_pred[:10]}")
    print(f"First 10 true values: {y_test[:10]}")
    print("Predictions completed.")

    # Ensure the directory exists before saving files
    os.makedirs('C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python', exist_ok=True)

    # Save the predictions and true values to files
    np.save('C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python\\y_pred.npy', y_pred)
    np.save('C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python\\y_test.npy', y_test)

    # Verify file creation
    print("y_pred.npy file created successfully.")

    # Generate dates for the forecast
    start_date = sys.argv[1]  # Example: pass '2024-01-01' as the start date
    forecast_days = 14  # Number of days to forecast
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    dates = [start_date + timedelta(days=i) for i in range(len(y_pred) + forecast_days)]
    dates = [date.strftime('%Y-%m-%d') for date in dates]

    # Save dates to a text file
    with open('C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python\\dates.txt', 'w') as f:
        for date in dates:
            f.write(f"{date}\n")

    # Prepare for future predictions
    last_known_data = X_test[-1]  # The last known data point in test set
    future_predictions = []

    for i in range(forecast_days):  # Number of future days
        # Predict the next step
        next_pred = regressor.predict(last_known_data.reshape(1, X_test.shape[1], X_test.shape[2]))

        # Append the prediction for saving and analysis
        future_predictions.append(next_pred[0][0])

        # Update last_known_data: Use the next_pred for the next step
        next_pred_reshaped = np.repeat(next_pred, X_test.shape[2], axis=1)
        last_known_data = np.concatenate((last_known_data[1:], next_pred_reshaped), axis=0)

    # Convert future_predictions to a numpy array
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    # Unscale the future predictions
    future_predictions = scaler_y.inverse_transform(future_predictions)
    print("future_predictions: ", future_predictions)
    print("future_predictions shape:", future_predictions.shape)
    print("Last known data shape:", last_known_data.shape)
    print("Next prediction shape:", next_pred.shape)

    # Ensure y_pred is 2-dimensional
    y_pred = y_pred.reshape(-1, 1)

    # Save the combined predictions to a file
    combined_predictions = np.concatenate((y_pred, future_predictions), axis=0)
    np.save('C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python\\combined_predictions.npy', combined_predictions)
    print("combined_predictions.npy file created successfully.")

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Plot the results
    plt.figure(figsize=(14, 5))
    plt.plot(y_test, color='blue', label='Actual Stock Price')
    plt.plot(y_pred, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig('C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python\\stock_price_prediction.png')
    plt.show()

except UnicodeEncodeError as e:
    print(f"UnicodeEncodeError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    sys.stdout.close()