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
