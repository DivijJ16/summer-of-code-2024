import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from sklearn.ensemble import VotingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split

# Load the datasets
data = pd.read_csv('train.csv')
stores_data = pd.read_csv('stores.csv')
oil_data = pd.read_csv('oil.csv')

# Preprocessing 'stores.csv' and 'oil.csv'
# 1) Encode categorical columns

# data_encoded = pd.get_dummies(data, columns=['family'], drop_first=True)  -> This created a lot of additional columns
# I tried one-hot encoding but it created a lot of additional boolean columns which made the dataset very large. 
# Therefore I used label encoding instead.

encoder = LabelEncoder()

# Apply Label Encoding to the categorical columns of stores_data, data and oil_data
data['family'] = encoder.fit_transform(data['family'])
stores_data['city'] = encoder.fit_transform(stores_data['city'])
stores_data['state'] = encoder.fit_transform(stores_data['state'])
stores_data['type'] = encoder.fit_transform(stores_data['type'])


# 2) Preprocessing 'oil.csv'
# Convert 'date' column to datetime as this is a time series dataset
oil_data['date'] = pd.to_datetime(oil_data['date'])
data['date'] = pd.to_datetime(data['date'])

# Handling missing values in 'dcoilwtico' using backward fill. 
# I used backward fill because the dataset is a time series dataset.
oil_data['dcoilwtico'] = oil_data['dcoilwtico'].bfill()

# Merge 'stores.csv' with 'train.csv' on 'store_nbr'
merged_data = data.merge(stores_data, on='store_nbr', how='left')

# Merge 'oil.csv' with the merged dataset on 'date'
merged_data = merged_data.merge(oil_data, on='date', how='left')


# Feature Engineering

# 1) Convert 'date' column to datetime
merged_data['date'] = pd.to_datetime(merged_data['date'])
# 2)Extract the day of the month from the 'date' column as it is instrumental in predicting sales
merged_data['day'] = merged_data['date'].dt.day
# 3) Encode the 'day' column
merged_data['day'] = encoder.fit_transform(merged_data['day'])
# 4) Extract the month from the 'date' column
merged_data['month'] = merged_data['date'].dt.month
# 5) Encode the 'month' column
merged_data['month'] = encoder.fit_transform(merged_data['month'])
# 6) Extract the year from the 'date' column
merged_data['year'] = merged_data['date'].dt.month
# 7) Encode the 'year' column
merged_data['year'] = encoder.fit_transform(merged_data['year'])
# 8) Extract the day of the week from the 'date' column
merged_data['day_of_week'] = merged_data['date'].dt.dayofweek
# 9) Encode the 'day_of_week' column
merged_data['day_of_week'] = encoder.fit_transform(merged_data['day_of_week'])
# 10) Drop the 'id' column as it is no longer needed for analysis.
merged_data.drop('id', axis=1, inplace=True)


# Normalizing numerical features
num_features = merged_data.select_dtypes(include=['float64', 'int64']).columns
scaler = MinMaxScaler()
merged_data[num_features] = scaler.fit_transform(merged_data[num_features])

#CODE TO PLOT DATA  - 2D SCATTER PLOTS AND CORRELATION MATRIX

x_features = ['store_nbr', 'family','onpromotion','city','state','type','cluster','dcoilwtico','day_of_week','day','month','year']
#  these are the features we want to analyze
for i in range(len(x_features)):
    x_feature = x_features[i]  
    y_feature = 'sales'  
    # 2D Scatter Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=merged_data, x=x_feature, y=y_feature, alpha=0.7)
    plt.title(f"2D Scatter Plot: {x_feature} vs {y_feature}")
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.grid(True)
    plt.show()

# Selecting the columns I want to analyze
columns_to_analyze = ['store_nbr', 'family','onpromotion','city','state','type','cluster','dcoilwtico','day_of_week','day','month','year','sales']
# Correlation matrix
correlation_matrix = merged_data[columns_to_analyze].corr()

# Heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# Some Conclusions : 
# 1) max sales on sundays as well as mondays.
# 2) Also, the sales spiked during the months december and april.
#3) Similarly sales were skewed for some particular stores and families.

# Train-test split (85% train, 15% test)
training_data_size = int(len(merged_data) * 0.85)
training_data, test_data = merged_data.iloc[:training_data_size], merged_data.iloc[training_data_size:]

# Augmented Dickey-Fuller test to check for stationarity
def adf_test(series):
    result = adfuller(series)
    print("ADF Statistic: ", result[0])
    print("p-value: ", result[1])

# Applying ADF test to check stationarity
adf_test(merged_data['sales'])
# Applying first differencing
merged_data_diff = merged_data['sales'].diff().dropna()
# Check stationarity after differencing
adf_test(merged_data_diff)
# we estimated d=1  as we needed to differentiate the data once to make it stationary



# Plotting ACF and PACF for differenced data. This will help us determine the order of the ARIMA model by predicting the values of p and q
plt.figure(figsize=(10, 6))
plt.subplot(211)
plot_acf(merged_data_diff, lags=40, ax=plt.gca()) # to determine q
plt.subplot(212)
plot_pacf(merged_data_diff, lags=40, ax=plt.gca()) # to determine p
plt.show()
# from the given plots, I estimated p=1 and q=1 as the plot died out suddenly after lag 1 for ACF and PACF

# AutoARIMA to find the best parameters
auto_arima_model = pm.auto_arima(merged_data['sales'], seasonal=False, stepwise=True, trace=True,suppress_warnings=True,error_action="ignore")
# Extractig the parameters
p = auto_arima_model.order[0]
d = auto_arima_model.order[1]
q = auto_arima_model.order[2]
# Print the best parameters found by AutoARIMA
print(f"Best ARIMA parameters: {auto_arima_model.order}")
# I found that the estimated values match the predicted ones, hence I will use these values for the ARIMA model

# Fitting the ARIMA model
arima_model = ARIMA(training_data['sales'], order=(p, d, q))
arima_fitted = arima_model.fit()
# Forecasting the ARIMA model
arima_forecast = arima_fitted.forecast(steps=len(test_data))
# Plotting ARIMA forecast vs actual
plt.figure(figsize=(10, 6))
plt.plot(training_data['sales'], label='Training Data')
plt.plot(test_data['sales'], label='Actual Sales', color='green')
plt.plot(test_data.index, arima_forecast, label='ARIMA Forecast', color='red')
plt.legend(loc='upper left')
plt.title('ARIMA Model Forecast')
plt.show()

# Evaluating ARIMA performance
arima_mse = mean_squared_error(test_data['sales'], arima_forecast)
arima_mae = mean_absolute_error(test_data['sales'], arima_forecast)
arima_mape = np.mean(np.abs((test_data['sales'] - arima_forecast) / test_data['sales'])) * 100
print(f"ARIMA Evaluation:\nMAE: {arima_mae}, MSE: {arima_mse}, MAPE: {arima_mape}%")


# AutoSARIMA to find the best seasonal parameters
auto_sarima_model = pm.auto_arima(merged_data['sales'], seasonal=True, m=12, stepwise=True, trace=True,suppress_warnings=True,error_action="ignore")

# Print the best SARIMA parameters
print(f"Best SARIMA parameters: {auto_sarima_model.order}, Seasonal order: {auto_sarima_model.seasonal_order}")
# Extract parameters
P = auto_arima_model.seasonal_order[0]
D = auto_arima_model.seasonal_order[1]
Q = auto_arima_model.seasonal_order[2]
S = auto_arima_model.seasonal_order[3]  # Seasonal period (12 months)

# Fitting the SARIMA model
sarima_model = SARIMAX(training_data['sales'], order=(p, d, q), seasonal_order=(P, D, Q, S))
sarima_fitted = sarima_model.fit()
# Forecasting the SARIMA model
sarima_forecast = sarima_fitted.forecast(steps=len(test_data))

# Plotting SARIMA forecast vs actual
plt.figure(figsize=(10, 6))
plt.plot(training_data['sales'], label='Training Data')
plt.plot(test_data['sales'], label='Actual Sales', color='green')
plt.plot(test_data.index, sarima_forecast, label='SARIMA Forecast', color='orange')
plt.legend(loc='upper left')
plt.title('SARIMA Model Forecast')
plt.show()

# Evaluate SARIMA performance
sarima_mse = mean_squared_error(test_data['sales'], sarima_forecast)
sarima_mae = mean_absolute_error(test_data['sales'], sarima_forecast)
sarima_mape = np.mean(np.abs((test_data['sales'] - sarima_forecast) / test_data['sales'])) * 100

print(f"SARIMA Evaluation:\nMAE: {sarima_mae}, MSE: {sarima_mse}, MAPE: {sarima_mape}%")

# Compare ARIMA and SARIMA
# 1) based on MAE
print(f"ARIMA MAPE: {arima_mape}%")
print(f"SARIMA MAPE: {sarima_mape}%")
# 2) based on MSE
print(f"ARIMA MSE: {arima_mse}")
print(f"SARIMA MSE: {sarima_mse}")

# The obtained values of MSE and MAE for ARIMA and SARIMA (after normalizing the data) are :
# ARIMA: MAE: 0.00569, MSE: 0.000113
# SARIMA: MAE: 0.00374, MSE: 0.0000491

# Hence, the SARIMA model performed better than the ARIMA model in terms of MAE and MSE.
# This is probably because the SARIMA model takes into account the seasonality of the data, which the ARIMA model does not.
# Therefore, I would choose the SARIMA model for forecasting sales.


# Using Prophet Model for forecasting
from prophet import Prophet

#  Renaming columns to 'ds' and 'y' as required by Prophet
merged_data.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)
# Ensuring the 'ds' column is in datetime format
merged_data['ds'] = pd.to_datetime(merged_data['ds'])
# Initializing the model
model = Prophet()
# Fitting the model to the data
model.fit(merged_data)
# Created a dataframe for future dates
future = model.make_future_dataframe(periods=15)  # Predict for the next 15 days
# Predicting future values
forecast = model.predict(future)
# Plot the forecast
model.plot(forecast)
# Plotting forecast components (trend, seasonality, etc.)
model.plot_components(forecast)




#  Voting Regressor
merged_data['date'] = pd.to_datetime(merged_data['date'])  
merged_data.set_index('date', inplace=True)

# defining target and features
target = 'sales'  # Replace with the actual target column
features = merged_data.drop(columns=[target]).values
target_values = merged_data[target].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target_values, test_size=0.15, shuffle=True)

# Custom wrapper for ARIMA model
class ARIMARegressor(BaseEstimator, RegressorMixin):
    def __init__(self, order=(1, 1, 1)):
        self.order = order

    def fit(self, X, y):
        self.model_ = ARIMA(y, order=self.order).fit()
        return self

    def predict(self, X):
        return self.model_.forecast(steps=len(X))

# Custom wrapper for SARIMA model
class SARIMARegressor(BaseEstimator, RegressorMixin):
    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)):
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, X, y):
        self.model_ = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order).fit(disp=False)
        return self

    def predict(self, X):
        return self.model_.forecast(steps=len(X))

# Creating ARIMA and SARIMA models with the best parameters found
arima_model = ARIMARegressor(order=(p,q,d)) # Replace p, q, d with the values found using AutoARIMA
sarima_model = SARIMARegressor(order=(p,q,d), seasonal_order=(P,Q,S,D)) # Replace P, Q, D, S with the values found using AutoSARIMA

# Calling Voting Regressor
voting_regressor = VotingRegressor(estimators=[
    ('arima', arima_model),
    ('sarima', sarima_model)
])

# Fitting the voting regressor
voting_regressor.fit(X_train, y_train)

# Making predictions
y_pred = voting_regressor.predict(X_test)

# Evaluating it
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
#  I found that the Voting Regressor performed better than the individual ARIMA and SARIMA models in terms of MAE and MSE

