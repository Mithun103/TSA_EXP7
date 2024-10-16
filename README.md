# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'ECOMM DATA.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Convert 'Order Date' column to datetime format and set it as index
data['Order Date'] = pd.to_datetime(data['Order Date'], infer_datetime_format=True)
data.set_index('Order Date', inplace=True)

# Assume 'Profit' column represents the data we want to analyze
profit_data = data['Profit']  # Adjust the column name if it's different

# Resample to weekly data by taking the mean
weekly_profit_data = profit_data.resample('W').mean()

# Perform Augmented Dickey-Fuller test for stationarity
result = adfuller(weekly_profit_data.dropna())
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
if result[1] < 0.05:
    print("The data is stationary.")
else:
    print("The data is non-stationary.")

# Split data into training and test sets (80% training, 20% testing)
train_size = int(len(weekly_profit_data) * 0.8)
train, test = weekly_profit_data[:train_size], weekly_profit_data[train_size:]

# Plot ACF and PACF for training data
fig, ax = plt.subplots(2, figsize=(8, 6))
plot_acf(train.dropna(), ax=ax[0], title='Autocorrelation Function (ACF)')
plot_pacf(train.dropna(), ax=ax[1], title='Partial Autocorrelation Function (PACF)')
plt.show()

# Fit AutoRegressive (AR) model with 13 lags
ar_model = AutoReg(train.dropna(), lags=13).fit()

# Make predictions on the test set
ar_pred = ar_model.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Plot the predictions against the actual test data
plt.figure(figsize=(10, 4))
plt.plot(test, label='Test Data')
plt.plot(ar_pred, label='AR Model Prediction', color='red')
plt.title('AR Model Prediction vs Test Data')
plt.xlabel('Time')
plt.ylabel('Profit')
plt.legend()
plt.show()

# Calculate Mean Squared Error
mse = mean_squared_error(test, ar_pred)
print(f'Mean Squared Error (MSE): {mse}')

# Plot the full data: Train, Test, and Predictions
plt.figure(figsize=(10, 4))
plt.plot(train, label='Train Data')
plt.plot(test, label='Test Data')
plt.plot(ar_pred, label='AR Model Prediction', color='red')
plt.title('Train, Test, and AR Model Prediction')
plt.xlabel('Time')
plt.ylabel('Profit')
plt.legend()
plt.show()
```
### OUTPUT:

GIVEN DATA
![image](https://github.com/user-attachments/assets/4cb247f4-d6c6-4011-a9a9-2361b9fa2839)

PACF - ACF
![image](https://github.com/user-attachments/assets/38e048cd-c13d-4205-a302-bcbfc3ea9a0d)


PREDICTION
![image](https://github.com/user-attachments/assets/b09c2dd0-1afa-4fc6-bab4-c58e173db600)

FINIAL PREDICTION
![image](https://github.com/user-attachments/assets/86aa0b5f-2b00-474d-bf91-ca3bb513d0e6)

### RESULT:
Thus we have successfully implemented the auto regression function using python.
