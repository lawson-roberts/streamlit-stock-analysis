# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 10:10:39 2021

@author: ROBERTLJ
"""
import pandas as pd
import numpy as np
from pandas import json_normalize
from selenium.webdriver.support.expected_conditions import element_selection_state_to_be
#import matplotlib.pyplot as plt
#import seaborn as sns
import streamlit as st
import base64
import plotly.express as px
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import io
from math import floor

##stock packages
import yfinance as yfin
import finnhub

## web crawling packages
import time
from datetime import date
from datetime import datetime
import datetime
import requests
from lxml import html
import csv

##Setting Layout to Wide
st.set_page_config(layout="wide")

##functions

def get_table_download_link(df, filename='download', message='Download csv result file'):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv" >{message}</a>'
    return href

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

##importing files needed for web app
ticker_selection = pd.read_csv('data/tickers_only.csv')
filler_df = pd.DataFrame({"full": ['Please Search for a Stock'],"url": 'Please Search for a Stock', "ticker": 'Please Search for a Stock', "company_name": 'Please Search for a Stock'})

today = date.today()
year_ago = today - datetime.timedelta(days=365)
unixtime_today = time.mktime(today.timetuple())
unixtime_year = time.mktime(year_ago.timetuple())

ticker_selection = filler_df.append(ticker_selection)

col1, col2 = st.beta_columns(2)

## Streamlit
with col1:
    st.write("""
             # Superior Returns Stock Exploration Application
             """)
    st.write("""## Data Sources:""")
    st.write("""1.) yfinance python package""")
    st.write("""2.) https://stockanalysis.com/stocks/ used for crawling avaialble stock tickers""")

#with col2:
    #bull = Image.open('image.jpg')
    #st.image(bull, caption='Superior Returns', use_column_width=True) #use_column_width=True, width = 100)
    ## Need to make container and columns for Filters
st.write("## Filters")
filter_expander = st.beta_expander(" ", expanded=True)

with filter_expander:
    col3, col4 = st.beta_columns(2)
    
    with col3:
        pick_ticker = st.selectbox("Select Stock Ticker", ticker_selection["full"].unique())
        #pick_ticker_all = pick_ticker
        st.write("You have selected", pick_ticker)

    if pick_ticker == "Please Search for a Stock":
        pass
    else:
    
        with col4:
            st.write(pick_ticker[0])
            ticker_row_selected = ticker_selection.loc[ticker_selection["full"] == pick_ticker]
            ticker_desc = ticker_row_selected['ticker'].unique()
            ticker_desc = ticker_desc[0]
            st.write("Ticker Symbol", ticker_desc, ".")
            ticker_url = ticker_row_selected['url'].unique()
            ticker_url = ticker_url[0]
            st.write("Ticker Url", ticker_url)
            ticker = yfin.Ticker(ticker_desc)
            logo = ticker.info
            logo = json_normalize(logo)
            logo = logo['logo_url']
            logo = logo[0]
            response = requests.get(logo)
            image_bytes = io.BytesIO(response.content)
            img = Image.open(image_bytes)
            st.image(img)

if pick_ticker == "Please Search for a Stock":
        pass
else:

    st.write("## Price Performance")
    price_movement_expander = st.beta_expander(' ', expanded=True)

    ##making options to show different graphs
    period_list = {'Period':['1 Week', '1 Month', '3 Months', '6 Months', '1 Year'], 'Period_value':[5, 23, 69, 138, 250]}
    period_dict = pd.DataFrame(period_list)

    with price_movement_expander:
        col5, col6 = st.beta_columns((1,3))

        with col5:
            period_selection = st.selectbox("Select Time Period", period_dict['Period'].unique())
            period_row_selected = period_dict.loc[period_dict["Period"] == period_selection]
            period_desc = period_row_selected['Period_value'].unique()
            period_desc = period_desc[0]
            
            chart_selection = st.radio("Pick Which Stock Price Analysis you would like to look at", ("Candles", "MACD (Moving Average Convergence Divergence)", "RSI (Relative Strength Indictor)", "All"))

        with col6:
            # Setup client
            finnhub_client = finnhub.Client(api_key="c3qcjnqad3i9vt5tl68g")
            res = finnhub_client.stock_candles(ticker_desc, 'D', int(unixtime_year), int(unixtime_today))
            price_data = pd.DataFrame(res)
            price_data.columns = ['close', 'high', 'low', 'open', 'status', 'timestamp', 'volume']
            price_data['date'] = pd.to_datetime(price_data['timestamp'], unit='s')
            price_data.index = price_data['date']
            price_data = price_data.drop(columns=['date'])
            price_data['RSI'] = RSI(price_data['close'], 14)
            price_data['30_ma'] = price_data['close'].rolling(30).mean()
            price_data['30_st_dev'] = price_data['close'].rolling(30).std()
            price_data['Upper Band'] = price_data['30_ma'] + (price_data['30_st_dev'] * 2)
            price_data['Upper Band'] = price_data['30_ma'] + (price_data['30_st_dev'] * 2)
            slow = 26
            fast = 12
            smooth = 9
            exp1 = price_data['close'].ewm(span = fast, adjust = False).mean()
            exp2 = price_data['close'].ewm(span = slow, adjust = False).mean()
            price_data['macd'] = exp1 - exp2
            price_data['signal'] = price_data['macd'].ewm(span = smooth, adjust = False).mean()
            price_data['hist'] = price_data['macd'] - price_data['signal']
            price_data['macd_buy'] = np.where(price_data['macd'] > price_data['signal'], 1, 0)
            price_data['macd_sell'] = np.where(price_data['macd'] < price_data['signal'], 1, 0)
            price_data = price_data.tail(period_desc)
        
            if chart_selection == "Candles":
            
                # Create Candlestick Chart
                candles = go.Figure(data=[go.Candlestick(x=price_data.index,
                    open=price_data['open'],
                    high=price_data['high'],
                    low=price_data['low'],
                    close=price_data['close'])])
                candles.add_trace(go.Scatter(x=price_data.index, y=price_data['close'], name='Stock Close Price', opacity=0.5))
                candles.update_yaxes(title="Stock Price")
                candles.update_xaxes(title="Date")
                candles.update_layout(title="Daily Stock Pricing")
                st.plotly_chart(candles, use_container_width = True)

            elif chart_selection == "MACD (Moving Average Convergence Divergence)":
                # Create MACD Chart
                macd = make_subplots(specs=[[{"secondary_y": True}]])

                #macd = go.Figure(data=[go.Candlestick(x=price_data.index, open=price_data['open'], high=price_data['high'], low=price_data['low'], close=price_data['close'])], secondary_y=False)
                #macd.add_trace(go.Scatter(x=price_data.index, y=price_data['close'], name='Stock Close Price', opacity=0.5), secondary_y=False)
                macd.add_trace(go.Scatter(x=price_data.index, y=price_data['signal'], name='MACD Signal'), secondary_y=True)
                macd.add_trace(go.Scatter(x=price_data.index, y=price_data['macd'], name='MACD Formula'), secondary_y=True)
                # Set y-axes titles
                macd.update_yaxes(title_text="<b>Candles</b> Stock Price Data", secondary_y=False)
                macd.update_yaxes(title_text="<b>MACD</b> Signals", secondary_y=True)
                macd.update_xaxes(title="Date")
                macd.update_layout(title="Stock MACD Graph")
                st.plotly_chart(macd, title="Stock RSI Graph", use_container_width = True)
                st.markdown('**Note: In general the guidance is when these two lines cross this should signal some action to be taken. When the MACD Signal > MACD Formula Line you should sell the stock based on this technical. And vice versa.**')
            
            elif chart_selection == "RSI (Relative Strength Indictor)":
                # Create RSI Chart
                #rsi = go.Figure()
                rsi = make_subplots(specs=[[{"secondary_y": True}]])

                rsi.add_trace(go.Scatter(x=price_data.index, y=price_data['RSI'], name='RSI Value'), secondary_y=False)
                rsi.add_hline(y=30, line_dash="dot", annotation_text="Under Bought Signal", annotation_position="bottom right", line_color='green')
                rsi.add_hline(y=70, line_dash="dot", annotation_text="Over Bought Signal", annotation_position="bottom right", line_color='red')
                rsi.add_trace(go.Scatter(x=price_data.index, y=price_data['close'], name='Stock Price Close', opacity=0.3), secondary_y=True)
                rsi.update_yaxes(title_text="<b>RSI</b> Relative Strength Indictor", secondary_y=False)
                rsi.update_yaxes(title_text="<b>Stock Price</b> Close", secondary_y=True)
                rsi.update_xaxes(title="Date")
                rsi.update_layout(title="Stock RSI Graph")
                st.plotly_chart(rsi, title="Stock RSI Graph", use_container_width = True)

            else:
                # Create Candlestick Chart
                candles = go.Figure(data=[go.Candlestick(x=price_data.index,
                    open=price_data['open'],
                    high=price_data['high'],
                    low=price_data['low'],
                    close=price_data['close'])])
                candles.add_trace(go.Scatter(x=price_data.index, y=price_data['close'], name='Stock Close Price', opacity=0.5))
                candles.update_yaxes(title="Stock Price")
                candles.update_xaxes(title="Date")
                candles.update_layout(title="Daily Stock Pricing")

                # Create MACD Chart
                macd = make_subplots(specs=[[{"secondary_y": True}]])

                #macd = go.Figure(data=[go.Candlestick(x=price_data.index, open=price_data['open'], high=price_data['high'], low=price_data['low'], close=price_data['close'])], secondary_y=False)
                macd.add_trace(go.Candlestick(x=price_data.index, open=price_data['open'], high=price_data['high'], low=price_data['low'], close=price_data['close']), secondary_y=False)
                macd.add_trace(go.Scatter(x=price_data.index, y=price_data['signal'], name='MACD Signal'), secondary_y=True)
                macd.add_trace(go.Scatter(x=price_data.index, y=price_data['macd'], name='MACD Formula'), secondary_y=True)
                # Set y-axes titles
                macd.update_yaxes(title_text="<b>Candles</b> Stock Price Data", secondary_y=False)
                macd.update_yaxes(title_text="<b>MACD</b> Signals", secondary_y=True)
                macd.update_xaxes(title="Date")
                macd.update_layout(title="Stock MACD Graph")

                # Create RSI Chart
                #rsi = go.Figure()
                rsi = make_subplots(specs=[[{"secondary_y": True}]])

                rsi.add_trace(go.Scatter(x=price_data.index, y=price_data['RSI'], name='RSI Value'), secondary_y=False)
                rsi.add_hline(y=30, line_dash="dot", annotation_text="Under Bought Signal", annotation_position="bottom right", line_color='green')
                rsi.add_hline(y=70, line_dash="dot", annotation_text="Over Bought Signal", annotation_position="bottom right", line_color='red')
                rsi.add_trace(go.Scatter(x=price_data.index, y=price_data['close'], name='Stock Price Close', opacity=0.3), secondary_y=True)
                rsi.update_yaxes(title_text="<b>RSI</b> Relative Strength Indictor", secondary_y=False)
                rsi.update_yaxes(title_text="<b>Stock Price</b> Close", secondary_y=True)
                rsi.update_xaxes(title="Date")
                rsi.update_layout(title="Stock RSI Graph")

                st.plotly_chart(candles, use_container_width = True)
                st.plotly_chart(macd, title="Stock RSI Graph", use_container_width = True)
                st.plotly_chart(rsi, title="Stock RSI Graph", use_container_width = True)

    st.write("## Analyst Recommendations")
    recommendations_expander = st.beta_expander(" ", expanded=True)

    with recommendations_expander:

        ticker = yfin.Ticker(ticker_desc)
        recommendations = ticker.recommendations
        recommendations = recommendations[recommendations['Action'] == 'main']
        recommendations = recommendations[recommendations.index > "2018-01-01"]
        recommendations_agg = pd.DataFrame(recommendations.groupby(['To Grade', 'Firm']).size()).reset_index()
        recommendations_agg.columns = ['Grade', 'Firm', 'Count of Analyst']
        st.plotly_chart(px.bar(recommendations_agg, x="Grade" , y="Count of Analyst", color='Firm', title="Recommendations by Firm since 2018"), use_container_width=True)
    
        col7, col8 = st.beta_columns(2)
        with col7:
            st.dataframe(recommendations_agg)
        with col8:
            recommendations

    url = 'https://www.marketwatch.com/investing/stock/' + ticker_desc
    request = requests.get(url)
    data = html.fromstring(request.text)

    # Xpath
    news =[]

    for i in data.xpath("//div[contains(@class,'element element--article')]"):
        stock = i.xpath('div/h3/a/text()')
        url = i.xpath('div/h3/a/@href')
        #year = i.xpath('span[2]/text()')   
        #print(title, url, year)
        news.append([stock, url])

    news_df = pd.DataFrame(news)
    news_df[0]=news_df[0].str[0]
    news_df[1]=news_df[1].str[0]
    news_df = news_df.rename(columns = {0: 'title', 1: 'url'})
    news_df['title'] = news_df['title'].astype('str')
    news_df['title'] = news_df.title.str.strip()
    news_df = news_df[news_df["title"].str.contains("nan")==False]



    st.write("## News about", pick_ticker)
    news_exapnder = st.beta_expander(" ", expanded=True)

    with news_exapnder:
        col9, col10 = st.beta_columns(2)

        with col9:
            news_df
