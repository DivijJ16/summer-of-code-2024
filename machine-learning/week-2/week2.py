import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the datasets
data = pd.read_csv('train.csv')
stores_data = pd.read_csv('stores.csv')
oil_data = pd.read_csv('oil.csv')

# Preprocessing 'stores.csv'
# Encode categorical columns
stores_encoded = stores_data.copy()
categorical_columns = ['city', 'state', 'type']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    stores_encoded[col + '_encoded'] = le.fit_transform(stores_encoded[col])
    label_encoders[col] = le

# Preprocessing 'oil.csv'
# Convert 'date' column to datetime
oil_data['date'] = pd.to_datetime(oil_data['date'])

# Handle missing values in 'dcoilwtico' using forward fill
oil_data['dcoilwtico'] = oil_data['dcoilwtico'].fillna(method='ffill')

# Merge 'stores.csv' with 'test.csv' on 'store_nbr'
merged_data = data.merge(stores_encoded, on='store_nbr', how='left')

# Merge 'oil.csv' with the merged dataset on 'date'
merged_data['date'] = pd.to_datetime(merged_data['date'])
merged_data = merged_data.merge(oil_data, on='date', how='left')

merged_data = merged_data.drop(columns=['city', 'state', 'type']) # Drop the original categorical columns


# Checking for remaining missing values and handling them
merged_data = merged_data.fillna(merged_data.mean(), inplace=True)

# Normalizing numerical features
numerical_features = merged_data.select_dtypes(include=['float64', 'int64']).columns

scaler = MinMaxScaler()
merged_data[numerical_features] = scaler.fit_transform(merged_data[numerical_features])

# Save the merged data to a new file
merged_data.to_csv('merged_data.csv', index=False)
merged_data.head()


