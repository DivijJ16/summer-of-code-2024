# Fraud_detection_dsoc
Starting off my 1st ML Project!

Approach: 
1)Reading and pre-processing the data: Since the data had some missing values, I randomly replaced these missing values with the mean of the known values in that column. I also used feature scaling to reduce the range of the parameters and to increase the models' efficiency.
I trained my model on 80% of the data and tested my model on the remaining 20% of the available data. This led to better predicition and higher AUC-ROC score.

2)Applied SMOTE to balance the training data to check outliers and inliers: I checked the AUC-ROC score of each model for various values of sampling_strategy(0.1,0.2,0.3,0.4,0.5) and concluded the best fit values for the models.

3) Applied various models like to my dataset including Logistic Regression, Decision Trees, LightGBM, and XGBoosting.

4) Evaluation: I evaluated the models using AUC-ROC score as well as Confusion Matrix and getting a Classification report of the model. I used scikit's tools to do this.

Challenges:
1)The dataset contained significantly more legitimate transactions than fraudulent ones, leading to potential bias in model training.

2)Certain models like Random Forests and SVM could not evaluate the data effectively and their runtime was too large.
Therefore, I could not evaluate this particular model.


Results:
1)The model successfully identified 90% of fraudulent transactions while maintaining a low false positive rate.

2)The final ROC-AUC scores of the models were:
a)Logistic regression: 0.9692
b)XGBoost: 0.9819
c)LightGBM: 0.9781
Therefore, I came to the conclusion that XGBoost was he most accurate model for this dataset.




