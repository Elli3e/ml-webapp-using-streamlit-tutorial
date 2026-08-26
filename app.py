import streamlit as st
import tensorflow as tf
import yfinance as yf
from curl_cffi import requests
from pickle import load
import numpy as np


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Stock Price Prediction")
st.write("Enter a stock ticker to predict the next stock price.")


# ============================================================
# Load MinMaxScaler
# ============================================================

@st.cache_resource
def load_scaler():
    with open("scale_min_max.sav", "rb") as f:
        return load(f)


# ============================================================
# Load trained LSTM model
# ============================================================

@st.cache_resource
def load_prediction_model():
    return tf.keras.models.load_model("lstm_model.keras")


scale = load_scaler()
prediction_model = load_prediction_model()


# ============================================================
# Create Yahoo Finance session
# ============================================================

@st.cache_resource
def create_yahoo_session():
    return requests.Session(impersonate="chrome")


session = create_yahoo_session()


# ============================================================
# User input
# ============================================================

ticker = st.text_input(
    "Enter stock ticker",
    value="BABA",
    max_chars=10
).strip().upper()


# ============================================================
# Get data and make prediction
# ============================================================

if st.button("Get Stock Data"):

    # --------------------------------------------------------
    # Validate ticker
    # --------------------------------------------------------

    if not ticker:
        st.error("Please enter a stock ticker.")
        st.stop()

    # --------------------------------------------------------
    # Download stock data
    # --------------------------------------------------------

    with st.spinner(f"Downloading data for {ticker}..."):

        try:
            stock = yf.Ticker(
                ticker,
                session=session
            )

            data = stock.history(
                period="1y",
                auto_adjust=False
            )

        except Exception as e:
            st.error(
                f"Unable to download data for {ticker}."
            )
            st.exception(e)
            st.stop()

    # --------------------------------------------------------
    # Check whether data was returned
    # --------------------------------------------------------

    if data.empty:
        st.error(
            f"No stock data was returned for '{ticker}'. "
            "Please check that the ticker symbol is correct."
        )
        st.stop()

    # --------------------------------------------------------
    # Get Close price
    # --------------------------------------------------------

    if "Close" not in data.columns:
        st.error(
            "The downloaded stock data does not contain "
            "a 'Close' price column."
        )
        st.stop()

    close_data = data[["Close"]].copy()

    # --------------------------------------------------------
    # Remove missing Close prices
    # --------------------------------------------------------

    close_data = close_data.dropna()

    # --------------------------------------------------------
    # Check number of observations
    # --------------------------------------------------------

    if len(close_data) < 60:
        st.error(
            f"Only {len(close_data)} valid closing prices "
            "were returned."
        )

        st.info(
            "The LSTM model requires at least "
            "60 days of historical data."
        )

        st.stop()

    # --------------------------------------------------------
    # Display downloaded data information
    # --------------------------------------------------------

    st.write(
        f"Downloaded {len(close_data)} days of data "
        f"for **{ticker}**."
    )

    # --------------------------------------------------------
    # Scale the Close prices
    # --------------------------------------------------------

    try:
        scaled_stock = scale.transform(close_data)

    except Exception as e:
        st.error("Error while scaling the stock data.")
        st.exception(e)
        st.stop()

    # --------------------------------------------------------
    # Select the most recent 60 days
    # --------------------------------------------------------

    x_input = scaled_stock[-60:, 0]

    # --------------------------------------------------------
    # Convert to NumPy array
    # --------------------------------------------------------

    x_input = np.array(x_input)

    # --------------------------------------------------------
    # Reshape for LSTM
    #
    # Expected shape:
    #
    # (samples, timesteps, features)
    #
    # 1 sample
    # 60 time steps
    # 1 feature
    # --------------------------------------------------------

    x_input = x_input.reshape(1, 60, 1)

    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    with st.spinner("Generating prediction..."):

        try:
            pred = prediction_model.predict(
                x_input,
                verbose=0
            )

        except Exception as e:
            st.error("The LSTM model failed to make a prediction.")
            st.exception(e)
            st.stop()

    # --------------------------------------------------------
    # Convert scaled prediction back to original price
    # --------------------------------------------------------

    try:
        prediction = scale.inverse_transform(pred)

    except Exception as e:
        st.error(
            "Could not convert the prediction "
            "back to the original price."
        )
        st.exception(e)
        st.stop()

    # --------------------------------------------------------
    # Get predicted price
    # --------------------------------------------------------

    predicted_price = float(prediction[0][0])

    # --------------------------------------------------------
    # Display prediction
    # --------------------------------------------------------

    st.success(
        f"Predicted next price for {ticker}: "
        f"${predicted_price:,.2f}"
    )

    # --------------------------------------------------------
    # Display recent prices
    # --------------------------------------------------------

    st.subheader("Recent Stock Prices")

    st.dataframe(
        close_data.tail(10),
        use_container_width=True
    )