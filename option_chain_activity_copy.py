import os
from turtle import color, title
import pandas as pd
import numpy as np
from pandas import json_normalize
import json
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

## sentiment analysis
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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

def get_option_chain(ticker_desc, group):
    ## running stock option scraping
    #base1 = "https://api.nasdaq.com/api/quote/"
    #base2 = "/option-chain?assetclass=stocks&fromdate=all&todate=undefined&excode=oprac&callput=callput&money=all&type=all"
    #url = base1 + str(ticker_desc) + base2

    #payload={}

    #headers = {'server':'Kestrel', 'Accept':'application/json, text/plain, */*','Accept-Encoding': 'gzip, deflate, br', 'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36'}
    #headers = {'User-Agent': 'PostmanRuntime/7.28.4','Accept': '*/*','Accept-Encoding': 'gzip, deflate, br'}

    #response = requests.get(url, headers=headers, data=payload)

    #response_text = json.loads(response.text)

    ## reading option chain dictionary
    dictionary_loc = "data/option_data_group" + str(group) + "_dict.json"
    #print(dictionary_loc)
    ## old dict
    #response = open(r'data/option_data_group_everything_dict.json',)

    ##new dict
    response = open(dictionary_loc,)

    # returns JSON object as a dictionary
    response_text_dict = json.load(response)

    ## properly filtering and formatting new dictionary to be parsed through...probably better way to filter through dictionary that I couldnt figure out.
    response_text = pd.DataFrame(response_text_dict)
    response_text = response_text[response_text['ticker'] == ticker_desc]

    selected_response_dict = response_text.set_index('ticker').to_dict()
        
    ## parsing through option data to find most recent price
    price = json_normalize(selected_response_dict['option_data'][ticker_desc]['data'])
    #price = json_normalize(response_text['data'])
    price = pd.DataFrame(price['lastTrade'], columns=['lastTrade'])
    price[['last', 'trade', 'price', 'as', 'of', 'sep', 'day', 'year']]= price["lastTrade"].str.split(" ", expand = True)
    price_new = price['price'].str[1:][0]
    price_new = float(price_new)

    ## parsing through option data to find the activitiy
    option_data = json_normalize(selected_response_dict['option_data'][ticker_desc]['data']['table'], 'rows')
    #option_data = json_normalize(response_text['data']['table'], 'rows')
    option_data = option_data.drop(columns = ['drillDownURL'])
    option_data['expirygroup'] = option_data['expirygroup'].replace('', np.nan).ffill()
    option_data['expirygroup'] = pd.to_datetime(option_data['expirygroup'])
    option_data = option_data.dropna(subset = ['strike'])

    calls = option_data[['expirygroup', 'expiryDate', 'c_Last', 'c_Change', 'c_Bid', 'c_Ask', 'c_Volume', 'c_Openinterest', 'c_colour', 'strike']].copy()
    calls = calls.rename(columns = {'c_Last': 'Last', 'c_Change': 'Change', 'c_Bid': 'Bid', 'c_Ask': 'Ask', 'c_Volume': 'Volume', 'c_Openinterest': 'Openinterest', 'c_colour': 'colour'})
    calls['type'] = "Call"
    calls['strike'] = calls['strike'].astype(float)
    calls['money_category'] = np.where(calls['strike'] <= price_new, "In the Money", "Out of the Money")

    puts = option_data[['expirygroup', 'expiryDate', 'p_Last', 'p_Change', 'p_Bid', 'p_Ask', 'p_Volume', 'p_Openinterest', 'p_colour', 'strike']].copy()
    puts = puts.rename(columns = {'p_Last': 'Last', 'p_Change': 'Change', 'p_Bid': 'Bid', 'p_Ask': 'Ask', 'p_Volume': 'Volume', 'p_Openinterest': 'Openinterest', 'p_colour': 'colour'})
    puts['type'] = "Put"
    puts['strike'] = puts['strike'].astype(float)
    puts['money_category'] = np.where(puts['strike'] >= price_new, "In the Money", "Out of the Money")

    option_data_new = calls.append(puts)
    option_data_new = option_data_new.replace('--', 0)
    option_data_new['last_price'] = round(price_new, 2)
    option_data_new['last_price'] = option_data_new['last_price'].astype(float)
    option_data_new['Last'] = option_data_new['Last'].astype(float)
    option_data_new['Change'] = option_data_new['Change'].astype(float)
    option_data_new['Bid'] = option_data_new['Bid'].astype(float)
    option_data_new['Ask'] = option_data_new['Ask'].astype(float)
    option_data_new['Volume'] = option_data_new['Volume'].astype(float)
    option_data_new['Openinterest'] = option_data_new['Openinterest'].astype(float)
    option_data_new['strike'] = option_data_new['strike'].astype(float)
    option_data_new['new_date']= option_data_new['expirygroup']
    
    ## some feature engineering
    option_data_new['money_out_the_door_bid'] = option_data_new['strike'] + option_data_new['Bid']
    option_data_new['money_out_the_door_ask'] = option_data_new['strike'] + option_data_new['Ask']
    option_data_new['percent_change_to_break_even'] = ((option_data_new['money_out_the_door_ask'] - option_data_new['last_price']) / option_data_new['last_price'])*100
    #option_data_new['expirygroup'] = option_data_new['expirygroup'].astype(str)

    maxStrikeValue = option_data_new['strike'].max()
    minStrikeValue = option_data_new['strike'].min()
    twenty_fifth_per = np.percentile(option_data_new['strike'], 25)
    seventy_fifth_per = np.percentile(option_data_new['strike'], 75)

    first_date = option_data_new['expirygroup'].head(1)
    last_date = option_data_new['expirygroup'].tail(1)

    start_date = pd.to_datetime(first_date.unique()[0])
    end_date = pd.to_datetime(last_date.unique()[0])

    return option_data, calls, puts, option_data_new, maxStrikeValue, minStrikeValue, twenty_fifth_per, seventy_fifth_per, start_date, end_date, price_new

def app():
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

    ##importing files needed for web app
    ticker_selection = pd.read_csv('data/tickers_only.csv')
    filler_df = pd.DataFrame({"full": ['Please Search for a Stock'],"url": 'Please Search for a Stock', "ticker": 'Please Search for a Stock', "company_name": 'Please Search for a Stock'})
    ticker_selection = filler_df.append(ticker_selection)

    col1, col2 = st.beta_columns(2)

    ## Streamlit
    with col1:
        st.write("""
             # Superior Returns Stock Option Chain Analysis
             """)
        st.write("""## Data Sources:""")
        st.write("""1.) https://www.nasdaq.com/market-activity/stocks/""")
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
            pick_ticker = st.selectbox("Select Stock Ticker", ticker_selection["full"].unique())
            #pick_ticker_all = pick_ticker
            st.write("You have selected", pick_ticker)

        if pick_ticker == "Please Search for a Stock":
            pass
        else:
    
            with col4:
                #st.write(pick_ticker[0])
                ticker_row_selected = ticker_selection.loc[ticker_selection["full"] == pick_ticker]
                ## group info
                group_selected = ticker_row_selected['group']
                group_selected = group_selected.iloc[0]
                group_selected = int(group_selected)
                group_selected_var = str(group_selected)
                st.write(group_selected_var)

                ## ticker info
                try:
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
                except Exception as e:
                    print("Error:", e)

    if pick_ticker == "Please Search for a Stock":
        pass
    else:

        st.write("## Option Chain Activity for", pick_ticker)
        options_expander = st.beta_expander(" ", expanded=True)

        with options_expander:
            st.write("""https://www.nerdwallet.com/article/investing/options-trading-definitions used for understanding option chain data terms.""")
            st.write("### Options Filters:")
            option_data, calls, puts, option_data_new, maxStrikeValue, minStrikeValue, twenty_fifth_per, seventy_fifth_per, start_date, end_date, price_new = get_option_chain(ticker_desc, group_selected_var)
            
            date_selection = pd.DataFrame(option_data_new['expirygroup'])
            dummy_date_selector = pd.DataFrame({'expirygroup': ['Please Select a Date']})
            date_selection_new = dummy_date_selector.append(date_selection)
            date_slider = st.slider('Select date range', start_date.date(), end_date.date(), (start_date.date(), end_date.date()))
            option_strike_price_slider = st.slider("What Strike Prices would you like included?", float(minStrikeValue), float(maxStrikeValue), (float(twenty_fifth_per), float(seventy_fifth_per)))
            low_strike = option_strike_price_slider[0]
            high_strike = option_strike_price_slider[1]
            lowDate = date_slider[0]
            highDate = date_slider[1]
            option_data_new['expirygroup'] = pd.to_datetime(option_data_new['expirygroup']).dt.date
            st.write("### Most Recent Closing Price was...", price_new)

            date_mask1 = (option_data_new['expirygroup'] >= lowDate) & (option_data_new['expirygroup'] <= highDate)
            option_data_new = option_data_new.loc[date_mask1]

            strike_mask1 = (option_data_new['strike'] >= low_strike) & (option_data_new['strike'] <= high_strike)
            option_data_new = option_data_new.loc[strike_mask1]
            
            ## bid vs ask visual
            # bid_agg = option_data_new.groupby('expirygroup').agg(money_out_the_door= ('money_out_the_door_bid', 'mean')).reset_index()
            # bid_agg['type'] = 'Bid'

            # ask_agg = option_data_new.groupby('expirygroup').agg(money_out_the_door= ('money_out_the_door_ask', 'mean')).reset_index()
            # ask_agg['type'] = 'Ask'

            # bid_ask_agg = ask_agg.append(bid_agg)

            ## % change to break even agg by calls and puts
            break_even_df_expiry_group = option_data_new.groupby(['expirygroup', 'type']).agg({'percent_change_to_break_even': 'mean'}).reset_index()

            bar_fig_exp_group = px.bar(break_even_df_expiry_group, x="expirygroup", y="percent_change_to_break_even",
             color="type", barmode = 'group', title='% Change Needed to Break Even - by Expiration Date')
            st.plotly_chart(bar_fig_exp_group, use_container_width=True)

            ## % change to break even agg by calls and puts
            break_even_df_strike = option_data_new.groupby(['strike', 'type']).agg({'percent_change_to_break_even': 'mean'}).reset_index()

            bar_fig_strike = px.bar(break_even_df_strike, x="strike", y="percent_change_to_break_even",
             color="type", barmode = 'group', title='% Change Needed to Break Even - by Strike Price')
            st.plotly_chart(bar_fig_strike, use_container_width=True)            

            st.write(option_data_new)
        
