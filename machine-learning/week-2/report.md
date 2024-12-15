Time_Series_Forecasting_Using_Ecuador_Dataset

In this project, I used 3 datasets to predict the sales of particular items depending upon various parameters.
I have uplaoded two of them, and the third one(train.csv) was too large and couldn't be uplaoded.
I have also given the links of all three of them in line 21 of 'week2_submission.py' file...
 

APPROACH: 
1)Reading and pre-processing the data: Firstly, I encoded the categorical columns in the three datasets like family,city,state etc. with the help of Label Encoding so as to not unnecessarily create a large number of columns. Also, I converted the column of date to datetime format in order for the model to interpret it correctly. Lastly, I handled missing values in the column 'dcoilwtico' of 'oil.csv' using backward fill. 

2)Applied Feature Engineering and Merged the Datasets: I introduced new columns like day_of_week, month, year etc. because these parameters affect the sales in a strong way. Finally, I merged the three datasets into one based on some common column for applying the model on it.

3)Applied ARIMA/ SARIMA , plotted their forecast values and compared their performance by checking their MSE, MAE and MAPE. I also applied ADF DIfferencing test and plotted the ACF and PACF plots to estimate the values of p,d,q parameters.

4) I also applied the PROPHET Model to the merged data to forecast the sales for next 15 days. However, the model had a very large runtime and I couldn't get any results out of it..

5) Lastly, I made an ensemble model to combine ARIMA and SARIMA to improve the model's accuracy.

CHALLENGES:
1)I faced a lot of difficulties in analysing the ACF and PACF plots, as i was initially plotting them for the non- differenced data. 

2)The PROPHET model could not evaluate the data effectively and its runtime was too large. Therefore, I could not evaluate this particular model.

RESULTS:
The obtained values of MSE and MAE for ARIMA and SARIMA (after normalizing the data) are :
-> ARIMA: MAE: 0.00569, MSE: 0.000113
-> SARIMA: MAE: 0.00374, MSE: 0.0000491

Hence, the SARIMA model performed better than the ARIMA model in terms of MAE and MSE.
This is probably because the SARIMA model takes into account the seasonality of the data, which the ARIMA model does not.
