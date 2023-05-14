import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import plotly.express as px

# load the saved LSTM model
model = load_model('lstm_model.h5')

# function to make predictions using the loaded model
def predict_temperature(data):
    # slice the data to only include the last 60 days
    data = data[-60:]
    # reshape the input data to match the expected input shape of the model
    data = data.reshape((1, data.shape[0], 1))
    # make the prediction
    prediction = model.predict(data)
    # return the predicted temperature value
    return prediction[0][0]

# set up the Streamlit app
st.set_page_config(page_title="Daily Temperature Predictor", page_icon="🌡️")
st.title("Daily Temperature Predictor")
st.write("This app predicts the next 30 days' average temperature in Szeged, Hungary based on historical data.")

# load the temperature data
data = pd.read_csv('GlobalTemperatures.csv', parse_dates=['dt'])
data.set_index('dt', inplace=True)

# display the current temperature data
st.subheader("Current Temperature Data")
st.write(data.tail())

# get the latest temperature value
latest_temp = data['LandAverageTemperature'].iloc[-1]

# get the date of the latest temperature value
latest_date = data.index[-1]

# make predictions for the next 30 days
predicted_temps = []
for i in range(30):
    next_date = latest_date + pd.Timedelta(days=1)
    predicted_temp = predict_temperature(np.array(data['LandAverageTemperature']))
    predicted_temps.append(predicted_temp)
    latest_date = next_date
    data.loc[next_date] = predicted_temp

# display the predicted temperature values for the next 30 days
st.subheader("Next 30 Days' Predicted Temperatures")
st.write(data.tail(30)['LandAverageTemperature'])

# plot the predicted temperatures using Plotly Express
fig = px.line(data.tail(30), y='LandAverageTemperature', title="Predicted Temperatures' Graph for the Next 30 Days")
st.plotly_chart(fig)

# get the date and temperature of the highest predicted temperature
highest_temp = max(predicted_temps)
highest_temp_index = predicted_temps.index(highest_temp)
highest_temp_date = data.index[-30:][highest_temp_index]

# display the highest predicted temperature and its date
st.subheader("Highest Predicted Temperature")
st.write(f"The highest predicted temperature is {highest_temp:.2f}°C and it will happen on {highest_temp_date.date()}.")
st.write("Be sure to stay cool and hydrated, and take necessary precautions to prevent heat-related illnesses.") 
