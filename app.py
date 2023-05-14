import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

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
st.set_page_config(page_title="Daily Temperature Predictor", page_icon=":sunny:")
st.title("Daily Temperature Predictor")
st.write("This app predicts the average temperature in Szeged, Hungary for the next 30 days based on historical data.")
st.write("")

# load the temperature data
data = pd.read_csv('GlobalTemperatures.csv', parse_dates=['dt'])
data.set_index('dt', inplace=True)

# display the current temperature data
st.header("Current Temperature Data")
st.dataframe(data.tail())
st.write("")

# get the latest temperature value
latest_temp = data['LandAverageTemperature'].iloc[-1]

# get the date of the latest temperature value
latest_date = data.index[-1]

# get the predicted temperature values for the next 30 days
next_dates = [latest_date + pd.Timedelta(days=i) for i in range(1,31)]
predicted_temps = [predict_temperature(np.array(data['LandAverageTemperature'][-60:])) for i in range(30)]

# display the predicted temperature values for the next 30 days
st.header("Predicted Temperature for the Next 30 Days")
for i in range(30):
    st.write(f"Date: {next_dates[i].strftime('%Y-%m-%d')}, Predicted Temperature: {predicted_temps[i]:.2f} °C")
    st.write("")

# add some CSS styling
st.markdown(
    """
    <style>
        .css-17eq0hr {
            background-color: #f5f5f5;
            padding: 10px 20px;
            border-radius: 10px;
        }
        .css-1aumxhk {
            font-size: 24px;
            font-weight: bold;
            color: #666666;
            margin-bottom: 20px;
        }
        .css-1aumxhk small {
            font-size: 16px;
            color: #999999;
            margin-left: 10px;
        }
        .css-1f3l2lq {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px 1px rgba(0,0,0,0.1);
        }
        .css-xq1lnh {
            font-size: 18px;
            font-weight: bold;
            color: #333333;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)
