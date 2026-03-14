import json
import streamlit as st
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from sklearn.linear_model import LinearRegression

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

# ----------------------------
# Load Prompt JSON
# ----------------------------
with open("prompt_15.json", "r") as f:
    data = json.load(f)

prompt = PromptTemplate(
    template=data["template"],
    input_variables=["ticker", "price_now", "p3", "p6", "p12"]
)

parser = StrOutputParser()

# ----------------------------
# LLM Model
# ----------------------------
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    max_new_tokens=400,
    temperature=0.7
)

chat = ChatHuggingFace(llm=llm)

chain = prompt | chat | parser

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI Stock Predictor", page_icon="📈")

st.title("📈 AI Stock Price Predictor")

st.write("Supports **Indian NSE stocks** and **US stocks**")

# ----------------------------
# Stock Input
# ----------------------------
ticker_input = st.text_input(
    "Enter Stock Symbol (Example: RELIANCE, TCS, INFY, AAPL)",
    "RELIANCE"
)

# Convert to uppercase
ticker_input = ticker_input.upper()

# Automatically add NSE suffix if user enters Indian stock
if "." not in ticker_input and ticker_input not in ["AAPL", "MSFT", "GOOGL", "AMZN"]:
    ticker = ticker_input + ".NS"
else:
    ticker = ticker_input

# ----------------------------
# Predict Button
# ----------------------------
if st.button("Predict Price"):

    st.write(f"Fetching data for **{ticker}** ...")

    # ----------------------------
    # Download stock data
    # ----------------------------
    df = yf.download(ticker, start="2020-01-01", progress=False)

    if df.empty:
        st.error("❌ No stock data found. Please check the symbol.")
        st.stop()

    df = df[['Close']]
    df = df.dropna()

    # ----------------------------
    # Create prediction column
    # ----------------------------
    df['Prediction'] = df['Close'].shift(-1)
    df = df.dropna()

    X = df[['Close']]
    y = df['Prediction']

    if len(X) < 10:
        st.error("❌ Not enough data to train the model.")
        st.stop()

    # ----------------------------
    # Train Linear Regression
    # ----------------------------
    model = LinearRegression()
    model.fit(X, y)

    last_price = df[['Close']].iloc[-1].values.reshape(1, -1)
    price_now = float(df['Close'].iloc[-1])

    base_prediction = model.predict(last_price)[0]

    # ----------------------------
    # Future Predictions
    # ----------------------------
    p3 = base_prediction * 1.05
    p6 = base_prediction * 1.10
    p12 = base_prediction * 1.20

    # ----------------------------
    # Show predictions
    # ----------------------------
    st.subheader("📊 Future Price Forecast")

    st.metric("Current Price", f"{price_now:.2f}")
    st.metric("3 Month Prediction", f"{p3:.2f}")
    st.metric("6 Month Prediction", f"{p6:.2f}")
    st.metric("12 Month Prediction", f"{p12:.2f}")

    # ----------------------------
    # AI Analysis
    # ----------------------------
    st.write("### 🤖 AI Investment Analysis")

    result = chain.invoke({
        "ticker": ticker,
        "price_now": price_now,
        "p3": p3,
        "p6": p6,
        "p12": p12
    })

    st.write(result)

    # ----------------------------
    # Stock Chart
    # ----------------------------
    st.write("### 📉 Stock Price History")

    st.line_chart(df['Close'])