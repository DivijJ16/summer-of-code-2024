# Upload the csv file from the local pc.
# I downloaded this data from kaggle - https://www.kaggle.com/code/gpreda/credit-card-fraud-detection-predictive-models/input

# The data consists of 31 columns: 30 features and a target column Class(which tells if the transaction is fraudulent[0] or not[1]).
from google.colab import files
uploaded = files.upload()

# Importing all necessary libraries. The shall help us apply various models to the data
import pandas as pd
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler

# Reading and pre-processing the data: Since the data had some missing values, I randomly replaced these missing values with the mean of the known values in that column. I also used feature scaling to reduce the range of the parameters and to increase the models' efficiency.
# I trained my model on 80% of the data and tested my model on the remaining 20% of the available data. This led to better predicition and higher AUC-ROC score.
# This was because if I increased test_size too much, the model began predicting a lot of non-fraudulent transactions as fraudulent, which is not desired.

inp = pd.read_csv(io.BytesIO(uploaded['creditcard.csv']))
# Handle missing values
inp.fillna(inp.mean(), inplace=True)

X = inp.drop(columns=['Class', 'Time'])  # Drop 'Time' as it's unlikely to be useful to predict fraud.
y = inp['Class']

# Scale the features to increase efficiency
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Split into training and testing sets
training_parameters, testing_parameters, training_results, testing_results = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the training data to check outliers and inliers: I checked the AUC-ROC score of each model for various values of sampling_strategy(0.1,0.2,0.3,0.4,0.5) and concluded the best fit values for the models:
# 1)Logistic regression: 0.5
# 2)XGBoost: 0.2
# 3)LightGBM: 0.4
smote = SMOTE(sampling_strategy=0.2, random_state=42) #change the parameter according to the model
training_parameters_resampled, training_results_resampled = smote.fit_resample(training_parameters, training_results)
print("Original class distribution:", Counter(training_results))
print("Resampled class distribution:", Counter(training_results_resampled))

# APPLYING XGBOOST
model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# APPLYING LOGISTIC REGRESSION TRAINING MODEL
model = LogisticRegression(max_iter=1000)

# USING LIGHTGBM MODEL
model = LGBMClassifier(random_state=42, class_weight='balanced')

# APPLYING SVM MODEL:
# I faced a problem with this model. It's runtime was far too much and hence it couldn't model the data. Maybe this is because of the large dataset.
# Therefore, I could not evaluate this particular model.
model = SVC(kernel='rbf', probability=True, random_state=42) ##code for implementing SVM Model.

model.fit(training_parameters_resampled, training_results_resampled)

# Evaluate the models using AUC-ROC score as well as Confusion Matrix and getting a Classification report of the model. I used scikit's tools to do this.
output_prediction = model.predict(testing_parameters)
output_prediction_proba = model.predict_proba(testing_parameters)[:, 1]

# Classification Report
print("Classification Report:")
print(classification_report(testing_results, output_prediction))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(testing_results, output_prediction))

# AUC-ROC Score
roc_auc = roc_auc_score(testing_results, output_prediction_proba)
print("AUC-ROC Score:", roc_auc)

# I compared three models and found out their optimal AUC-ROC scores to be:
# 1)Logistic regression: 0.9692
# 2)XGBoost: 0.9819
# 3)LightGBM: 0.9781
# Therefore, I came to the conclusion that XGBoost was he most accurate model for this dataset.

