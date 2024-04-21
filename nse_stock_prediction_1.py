# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the provided NSE data for ITC
data = {
    'Date': ['19-Apr-24', '18-Apr-24', '16-Apr-24', '15-Apr-24', '12-Apr-24', '10-Apr-24', '09-Apr-24', '08-Apr-24', '05-Apr-24', '04-Apr-24'],
    'OPEN': [418, 426, 423.25, 428, 435, 428.3, 430.45, 428.3, 422.5, 425.55],
    'HIGH': [426.25, 426.9, 427, 429.15, 435.75, 437.8, 431.5, 431.4, 431.7, 427.35],
    'LOW': [416, 417.65, 423.2, 422.9, 428.3, 425.75, 425.55, 427.75, 419.95, 419.9],
    'PREV. CLOSE': [418.85, 425.9, 425.9, 430.1, 436.95, 426.35, 429.1, 427.55, 422.75, 425.2]
}

df = pd.DataFrame(data)

# Feature selection and model training
X = df[['OPEN', 'HIGH', 'LOW', 'PREV. CLOSE']]
y = df['OPEN']  # Predicting OPEN prices

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predicting the next five days' stock prices
next_five_days_data = {
    'OPEN': [430, 432, 431, 433, 435],
    'HIGH': [435, 437, 436, 438, 440],
    'LOW': [428, 430, 429, 431, 433],
    'PREV. CLOSE': [429, 431, 430, 432, 434]
}

next_five_days_df = pd.DataFrame(next_five_days_data)
predictions = model.predict(next_five_days_df)

print(predictions)
