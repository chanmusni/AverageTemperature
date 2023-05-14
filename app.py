# -*- coding: utf-8 -*-
"""app

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DwEIxdxS4X1M0tiaXJHnZCMaOIvaaaEA
"""

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# load the saved LSTM model
model = load_model('lstm_model.h5')

# function to make predictions using the loaded model
def predict_temperature(data):
    # reshape the input data to match the expected input shape of the model
    data = data.reshape((1, data.shape[0], 1))
    # make the prediction
    prediction = model.predict(data)
    # return the predicted temperature value
    return prediction[0][0]

# set up the Streamlit app
st.title("Daily Temperature Predictor")
st.write("This app predicts the next day's average temperature in Szeged, Hungary based on historical data.")

# load the temperature data
data = pd.read_csv('GlobalTemperatures.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)

# display the current temperature data
st.subheader("Current Temperature Data")
st.write(data.tail())

# get the latest temperature value
latest_temp = data['Temperature'].iloc[-1]

# get the date of the latest temperature value
latest_date = data.index[-1]

# get the predicted temperature value for the next day
next_date = latest_date + pd.Timedelta(days=1)
predicted_temp = predict_temperature(np.array(data['Temperature']))

# display the predicted temperature value for the next day
st.subheader("Next Day's Predicted Temperature")
st.write("Date:", next_date)
st.write("Predicted Temperature:", predicted_temp)