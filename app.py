import streamlit as st
import tensorflow as tf
import yfinance as yf
from curl_cffi import requests
from pickle import load
import numpy as np


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Stock Price Prediction")
st.write("Enter a stock ticker to predict the next stock price.")


# --------------------------------------------------
# Load scaler
# --------------------------------------------------

@st.cache_resource
def load_scaler():
    with open("scale_min_max.sav", "rb") as f:
        return load(f)


# --------------------------------------------------
# Load LSTM model
# --------------------------------------------------

@st.cache_resource
def load_prediction_model():
    return tf.keras.models.load_model("lstm_model.keras")


scale = load_scaler()
prediction_model = load_prediction_model()


# --------------------------------------------------
# Yahoo Finance session
# --------------------------------------------------

@st.cache_resource
def create_yahoo_session():
    return requests.Session(impersonate="chrome")


session = create_yahoo_session()


# --------------------------------------------------
# User input
# --------------------------------------------------

ticker = st.text_input(
    "Enter stock ticker",
    value="BABA",
    max_chars=10
).strip().upper()


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Get Stock Data"):

    if not ticker:
        st.error("Please enter a stock ticker.")
        st.stop()

    with st.spinner(f"Downloading data for {ticker}..."):

        try:
            data = yf.download(
                ticker,
                period="1y",
                session=session,
                progress=False,
                auto_adjust=False
            )

        except Exception as e:
            st.error(f"Unable to download stock data: {e}")
            st.stop()

    # --------------------------------------------------
    # Check downloaded data
    # --------------------------------------------------

    if data.empty:
        st.error(
            f"No stock data was returned for '{ticker}'. "
            "Please check that the ticker symbol is correct."
        )
        st.stop()

    # --------------------------------------------------
    # Get Close price
    # --------------------------------------------------

    try:
        close_data = data[["Close"]]
    except KeyError:
        st.error("The downloaded data does not contain a Close price.")
        st.stop()

    if close_data.empty:
        st.error("No Close price data was returned.")
        st.stop()

    # --------------------------------------------------
    # Make sure we have at least 60 observations
    # --------------------------------------------------

    if len(close_data) < 60:
        st.error(
            f"Only {len(close_data)} days of data were returned. "
            "At least 60 observations are required."
        )
        st.stop()

    # --------------------------------------------------
    # Scale stock prices
    # --------------------------------------------------

    try:
        scaled_stock = scale.transform(close_data)
    except Exception as e:
        st.error(f"Error scaling the stock data: {e}")
        st.stop()

    # --------------------------------------------------
    # Prepare LSTM input
    # --------------------------------------------------

    x_input = scaled_stock[-60:, 0]

    x_input = np.array(x_input)

    # Shape:
    # (samples, timesteps, features)
    #
    # 1 sample
    # 60 days
    # 1 feature (Close price)

    x_input = x_input.reshape(1, 60, 1)

    # --------------------------------------------------
    # Make prediction
    # --------------------------------------------------

    with st.spinner("Generating prediction..."):

        try:
            pred = prediction_model.predict(
                x_input,
                verbose=0
            )
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            st.stop()

    # --------------------------------------------------
    # Convert prediction back to original price
    # --------------------------------------------------

    try:
        prediction = scale.inverse_transform(pred)
    except Exception as e:
        st.error(f"Unable to convert prediction back to price: {e}")
        st.stop()

    predicted_price = float(prediction[0][0])

    # --------------------------------------------------
    # Display result
    # --------------------------------------------------

    st.success(
        f"Predicted next price for {ticker}: "
        f"${predicted_price:,.2f}"
    )

    # --------------------------------------------------
    # Display recent prices
    # --------------------------------------------------

    st.subheader("Recent Stock Prices")

    st.dataframe(
        close_data.tail(10),
        use_container_width=True
    )