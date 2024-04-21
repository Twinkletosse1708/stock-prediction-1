import pandas as pd

# Read the data
data = pd.read_csv('Quote-Equity-ITC-EQ-21-03-2024-to-21-04-2024.csv')

# Data preprocessing steps (e.g., handling missing values, converting date format)
# Perform any necessary data cleaning and transformation

#----------------------------------------------------------
from statsmodels.tsa.arima_model import ARIMA

# Fit the ARIMA model
model = ARIMA(data['CLOSE'], order=(5,1,0))
model_fit = model.fit(disp=0)

# Make predictions for the next 5 days
forecast = model_fit.forecast(steps=5)
print(forecast)
#----------------------------------------------------------------------------
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Perform prediction using the trained ARIMA model
    # Return the predicted stock prices
    return jsonify({'predictions': forecast.tolist()})

if __name__ == '__main__':
    app.run()
#---------------------------------------------
import streamlit as st

# Create a Streamlit app for user interaction
st.title('Stock Price Prediction App')

# Add user input components (e.g., date selection)
# Display the predicted stock prices
