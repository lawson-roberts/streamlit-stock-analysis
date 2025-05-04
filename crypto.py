import os
import pandas as pd
import numpy as np
from pandas import json_normalize
#from selenium.webdriver.support.expected_conditions import element_selection_state_to_be
#import matplotlib.pyplot as plt
#import seaborn as sns
import streamlit as st
import base64
import plotly.express as px
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#import matplotlib.pyplot as plt
import io
from math import floor
from datetime import date
import datetime
import time

##Crypto packages
import finnhub

def RSI(data, time_window):
    diff = data.diff(1).dropna()        # diff in one field(one day)

    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff
    
    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]
    
    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]
    
    # check pandas documentation for ewm
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi

def get_macd(price, slow, fast, smooth):
    exp1 = price.ewm(span = fast, adjust = False).mean()
    exp2 = price.ewm(span = slow, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns = {'close':'macd'})
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'macd':'signal'})
    hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns = {0:'hist'})
    frames =  [macd, signal, hist]
    df = pd.concat(frames, join = 'inner', axis = 1)
    return df

def app():


    ##Setting Streamlit Settings
    #st.set_page_config(layout="wide")

    ##importing files needed for web app

    today = date.today()
    year_ago = today - datetime.timedelta(days=365)
    unixtime_today = time.mktime(today.timetuple())
    unixtime_year = time.mktime(year_ago.timetuple())
    
    # Setup client
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise ValueError("FINNHUB_API_KEY environment variable is not set.")
    finnhub_client = finnhub.Client(api_key=api_key)
    symbol_df = pd.read_csv('data/crypto_symbol_df.csv')
    symbol_df['currency'] = symbol_df['description'].str[-3:]
    symbol_df_short = symbol_df[symbol_df['currency'] == 'USD']
    symbol_df_short = symbol_df_short.drop(columns = ['currency', 'Unnamed: 0'])

    filler_df = pd.DataFrame({"description": ['Please Search for a Crypto'],"displaySymbol": 'Please Search for a Crypto', "symbol": 'Please Search for a Crypto'})

    today = date.today()
    month_ago = today - datetime.timedelta(days=31)
    two_months_ago = today - datetime.timedelta(days=62)
    year_ago = today - datetime.timedelta(days=365)
    unixtime_today = time.mktime(today.timetuple())
    unixtime_year = time.mktime(year_ago.timetuple())

    symbol_selection = filler_df.append(symbol_df_short)

    col1, col2 = st.beta_columns(2)

    ## Streamlit
    with col1:
        st.write("""
             # Superior Returns Crypto Exploration Application
             """)
        st.write("""## Data Sources:""")
        st.write("""1.) finnhub python package""")
        #st.write("""2.) used for crawling avaialble Crypto tickers""")

    #with col2:
        #bull = Image.open('image.jpg')
        #st.image(bull, caption='Superior Returns', use_column_width=True) #use_column_width=True, width = 100)
        ## Need to make container and columns for Filters
    st.write("## Filters")
    filter_expander = st.beta_expander(" ", expanded=True)

    with filter_expander:
        col3, col4 = st.beta_columns(2)
    
        with col3:
            symbol_sel = st.selectbox("Select Crypto Symbol", symbol_selection["description"].unique())
            #pick_ticker_all = pick_ticker
            st.write("You have selected", symbol_sel)

        with col4:
            symbol_selection = symbol_selection.loc[symbol_selection["description"] == symbol_sel]
            symbol_desc = symbol_selection['symbol'].unique()
            symbol_desc = symbol_desc[0]
            st.write("Crypto Symbol:", symbol_desc, ".")

    if symbol_sel == "Please Search for a Crypto":
            pass
    else:
        candles = finnhub_client.crypto_candles(symbol_desc, 'D', int(unixtime_year), int(unixtime_today))
        candles_df = pd.DataFrame(candles)
        #candles_df = candles_df.reset_index()
        #candles_df = candles_df.drop(columns = ['index'])
        candles_df = candles_df.rename(columns = {'c':'close', 'h': 'high', 'l': 'low', 'o': 'open', 's': 'status', 't': 'timestamp','v': 'volumne'})
        candles_df['date'] = pd.to_datetime(candles_df['timestamp'], unit='s')

        st.write("## Crypto Performance")
        crypto_performance_expander = st.beta_expander(" ", expanded=True)

        ##making options to show different graphs
        period_list = {'Period':['1 Week', '1 Month', '3 Months', '6 Months', '1 Year'], 'Period_value':[5, 23, 69, 138, 250]}
        period_dict = pd.DataFrame(period_list)
        
        with crypto_performance_expander:

            col5, col6 = st.beta_columns((1,3))

            with col5:
                period_selection = st.selectbox("Select Time Period", period_dict['Period'].unique())
                period_row_selected = period_dict.loc[period_dict["Period"] == period_selection]
                period_desc = period_row_selected['Period_value'].unique()
                period_desc = period_desc[0]
            
                chart_selection = st.radio("Pick Which Crypto Price Analysis you would like to look at", ("Candles", "MACD (Moving Average Convergence Divergence)", "RSI (Relative Strength Indictor)", "All"))

            with col6:
                #st.write(candles_df.astype('object'))
                candles_df.index = candles_df['date']
                candles_df = candles_df.drop(columns=['date'])
                candles_df['RSI'] = RSI(candles_df['close'], 14)
                candles_df['30_ma'] = candles_df['close'].rolling(30).mean()
                candles_df['30_st_dev'] = candles_df['close'].rolling(30).std()
                candles_df['Upper Band'] = candles_df['30_ma'] + (candles_df['30_st_dev'] * 2)
                candles_df['Upper Band'] = candles_df['30_ma'] + (candles_df['30_st_dev'] * 2)
                slow = 26
                fast = 12
                smooth = 9
                exp1 = candles_df['close'].ewm(span = fast, adjust = False).mean()
                exp2 = candles_df['close'].ewm(span = slow, adjust = False).mean()
                candles_df['macd'] = exp1 - exp2
                candles_df['signal'] = candles_df['macd'].ewm(span = smooth, adjust = False).mean()
                candles_df['hist'] = candles_df['macd'] - candles_df['signal']
                candles_df['macd_buy'] = np.where(candles_df['macd'] > candles_df['signal'], 1, 0)
                candles_df['macd_sell'] = np.where(candles_df['macd'] < candles_df['signal'], 1, 0)
                candles_df = candles_df.tail(period_desc)

                if chart_selection == "Candles":
            
                    # Create Candlestick Chart
                    candles = go.Figure(data=[go.Candlestick(x=candles_df.index,
                        open=candles_df['open'],
                        high=candles_df['high'],
                        low=candles_df['low'],
                        close=candles_df['close'])])
                    candles.add_trace(go.Scatter(x=candles_df.index, y=candles_df['close'], name='Crypto Close Price', opacity=0.5))
                    candles.update_yaxes(title="Crypto Price")
                    candles.update_xaxes(title="Date")
                    candles.update_layout(title="Daily Crypto Pricing")
                    st.plotly_chart(candles, use_container_width = True)

                elif chart_selection == "MACD (Moving Average Convergence Divergence)":
                    # Create MACD Chart
                    macd = make_subplots(specs=[[{"secondary_y": True}]])

                    #macd = go.Figure(data=[go.Candlestick(x=candles_df.index, open=candles_df['open'], high=candles_df['high'], low=candles_df['low'], close=candles_df['close'])], secondary_y=False)
                    #macd.add_trace(go.Scatter(x=candles_df.index, y=candles_df['close'], name='Crypto Close Price', opacity=0.5), secondary_y=False)
                    macd.add_trace(go.Scatter(x=candles_df.index, y=candles_df['signal'], name='MACD Signal'), secondary_y=True)
                    macd.add_trace(go.Scatter(x=candles_df.index, y=candles_df['macd'], name='MACD Formula'), secondary_y=True)
                    # Set y-axes titles
                    macd.update_yaxes(title_text="<b>Candles</b> Crypto Price Data", secondary_y=False)
                    macd.update_yaxes(title_text="<b>MACD</b> Signals", secondary_y=True)
                    macd.update_xaxes(title="Date")
                    macd.update_layout(title="Crypto MACD Graph")
                    st.plotly_chart(macd, title="Crypto RSI Graph", use_container_width = True)
                    st.markdown('**Note: In general the guidance is when these two lines cross this should signal some action to be taken. When the MACD Signal > MACD Formula Line you should sell the Crypto based on this technical. And vice versa.**')
            
                elif chart_selection == "RSI (Relative Strength Indictor)":
                    # Create RSI Chart
                    #rsi = go.Figure()
                    rsi = make_subplots(specs=[[{"secondary_y": True}]])

                    rsi.add_trace(go.Scatter(x=candles_df.index, y=candles_df['RSI'], name='RSI Value'), secondary_y=False)
                    rsi.add_hline(y=30, line_dash="dot", annotation_text="Under Bought Signal", annotation_position="bottom right", line_color='green')
                    rsi.add_hline(y=70, line_dash="dot", annotation_text="Over Bought Signal", annotation_position="bottom right", line_color='red')
                    rsi.add_trace(go.Scatter(x=candles_df.index, y=candles_df['close'], name='Crypto Price Close', opacity=0.3), secondary_y=True)
                    rsi.update_yaxes(title_text="<b>RSI</b> Relative Strength Indictor", secondary_y=False)
                    rsi.update_yaxes(title_text="<b>Crypto Price</b> Close", secondary_y=True)
                    rsi.update_xaxes(title="Date")
                    rsi.update_layout(title="Crypto RSI Graph")
                    st.plotly_chart(rsi, title="Crypto RSI Graph", use_container_width = True)

                else:
                    # Create Candlestick Chart
                    candles = go.Figure(data=[go.Candlestick(x=candles_df.index,
                        open=candles_df['open'],
                        high=candles_df['high'],
                        low=candles_df['low'],
                        close=candles_df['close'])])
                    candles.add_trace(go.Scatter(x=candles_df.index, y=candles_df['close'], name='Crypto Close Price', opacity=0.5))
                    candles.update_yaxes(title="Crypto Price")
                    candles.update_xaxes(title="Date")
                    candles.update_layout(title="Daily Crypto Pricing")

                    # Create MACD Chart
                    macd = make_subplots(specs=[[{"secondary_y": True}]])

                    #macd = go.Figure(data=[go.Candlestick(x=candles_df.index, open=candles_df['open'], high=candles_df['high'], low=candles_df['low'], close=candles_df['close'])], secondary_y=False)
                    macd.add_trace(go.Candlestick(x=candles_df.index, open=candles_df['open'], high=candles_df['high'], low=candles_df['low'], close=candles_df['close']), secondary_y=False)
                    macd.add_trace(go.Scatter(x=candles_df.index, y=candles_df['signal'], name='MACD Signal'), secondary_y=True)
                    macd.add_trace(go.Scatter(x=candles_df.index, y=candles_df['macd'], name='MACD Formula'), secondary_y=True)
                    # Set y-axes titles
                    macd.update_yaxes(title_text="<b>Candles</b> Crypto Price Data", secondary_y=False)
                    macd.update_yaxes(title_text="<b>MACD</b> Signals", secondary_y=True)
                    macd.update_xaxes(title="Date")
                    macd.update_layout(title="Crypto MACD Graph")

                    # Create RSI Chart
                    #rsi = go.Figure()
                    rsi = make_subplots(specs=[[{"secondary_y": True}]])

                    rsi.add_trace(go.Scatter(x=candles_df.index, y=candles_df['RSI'], name='RSI Value'), secondary_y=False)
                    rsi.add_hline(y=30, line_dash="dot", annotation_text="Under Bought Signal", annotation_position="bottom right", line_color='green')
                    rsi.add_hline(y=70, line_dash="dot", annotation_text="Over Bought Signal", annotation_position="bottom right", line_color='red')
                    rsi.add_trace(go.Scatter(x=candles_df.index, y=candles_df['close'], name='Crypto Price Close', opacity=0.3), secondary_y=True)
                    rsi.update_yaxes(title_text="<b>RSI</b> Relative Strength Indictor", secondary_y=False)
                    rsi.update_yaxes(title_text="<b>Crypto Price</b> Close", secondary_y=True)
                    rsi.update_xaxes(title="Date")
                    rsi.update_layout(title="Crypto RSI Graph")

                    st.plotly_chart(candles, use_container_width = True)
                    st.plotly_chart(macd, title="Crypto RSI Graph", use_container_width = True)
                    st.plotly_chart(rsi, title="Crypto RSI Graph", use_container_width = True)
