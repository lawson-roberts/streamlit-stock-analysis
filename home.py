import streamlit as st
import base64
from PIL import Image

def app():

    
    st.markdown("<h1 style='text-align: center;'>Superior Returns Stock Analysis Home Page</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.beta_columns([1,2,1])
    #st.title('Stock Trends')
    with col2:
        bull = Image.open('bear_bull_new.jpg')
        st.image(bull, caption='Superior Returns', width=680)
    st.write("""## See Instructions Below:""")
    st.write("""### 1.) Stock Analysis Page: Here you can search for stocks you are interested in and see key details that are helpful for understanding the performance of a stock.""")
    st.write("""    a.) Technical Indictor Analysis including Candles, MACD (Moving Average Convergence Divergence) Indicators, and RSI (Relative Strength Indicators)""")
    st.write("""    b.) Analyst Recommendation Data""")
    st.write("""    c.) Sentiment Analysis from Stock specific news articles""")
    st.write("""### 2.) Industry Trends Analysis Page: Coming Soon!""")