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
    option_data_new['Last'] = option_data_new['Last'].astype(float)
    option_data_new['Change'] = option_data_new['Change'].astype(float)
    option_data_new['Bid'] = option_data_new['Bid'].astype(float)
    option_data_new['Ask'] = option_data_new['Ask'].astype(float)
    option_data_new['Volume'] = option_data_new['Volume'].astype(float)
    option_data_new['Openinterest'] = option_data_new['Openinterest'].astype(float)
    option_data_new['strike'] = option_data_new['strike'].astype(float)
    option_data_new['new_date']= option_data_new['expirygroup']
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
    
    finnhub_client = finnhub.Client(api_key="c3qcjnqad3i9vt5tl68g")

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

            #st.write(option_data.astype('object'))

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
            st.write(lowDate)
            highDate = date_slider[1]
            st.write(highDate)
            option_data_new['expirygroup'] = pd.to_datetime(option_data_new['expirygroup']).dt.date
            st.write("### Most Recent Closing Price was...", price_new)

            date_mask1 = (option_data_new['expirygroup'] >= lowDate) & (option_data_new['expirygroup'] <= highDate)
            option_data_new = option_data_new.loc[date_mask1]

            strike_mask1 = (option_data_new['strike'] >= low_strike) & (option_data_new['strike'] <= high_strike)
            option_data_new = option_data_new.loc[strike_mask1]

            calls_clean = option_data_new[option_data_new['type'] == 'Call']
            puts_clean = option_data_new[option_data_new['type'] == 'Put']

            option_data = option_data.replace('--', 0)
            option_data['c_Volume'] = option_data['c_Volume'].astype(float)
            option_data['p_Volume'] = option_data['p_Volume'].astype(float)
            option_data['c_Openinterest'] = option_data['c_Openinterest'].astype(float)
            option_data['p_Openinterest'] = option_data['p_Openinterest'].astype(float)
            option_data['strike'] = option_data['strike'].astype(float)
            option_data['expirygroup'] = pd.to_datetime(option_data['expirygroup']).dt.date
            date_mask2 = (option_data['expirygroup'] >= lowDate) & (option_data['expirygroup'] <= highDate)
            option_data = option_data.loc[date_mask2]

            strike_mask2 = (option_data['strike'] >= low_strike) & (option_data['strike'] <= high_strike)
            option_data = option_data.loc[strike_mask2]

            option_data_executed_volume_graph = pd.DataFrame(option_data.groupby('strike').agg({'c_Volume': 'sum', 'p_Volume': 'sum'})).reset_index()
            option_data_executed_volume_graph['call/put_ratio_Volume'] = option_data_executed_volume_graph['c_Volume'] / option_data_executed_volume_graph['p_Volume']

            option_data_open_interest_graph = pd.DataFrame(option_data.groupby('strike').agg({'c_Openinterest': 'sum', 'p_Openinterest': 'sum'})).reset_index()
            option_data_open_interest_graph['call/put_ratio_Openinterest'] = option_data_open_interest_graph['c_Openinterest'] / option_data_open_interest_graph['p_Openinterest']

            # Create Volume / Openinterest Chart
            option_ratios_graph = make_subplots(specs=[[{"secondary_y": True}]])

            option_ratios_graph.add_trace(go.Scatter(x=option_data_executed_volume_graph['strike'], y=option_data_executed_volume_graph['call/put_ratio_Volume'], name='call/put_ratio_Volume'), secondary_y=False)
            option_ratios_graph.add_trace(go.Scatter(x=option_data_open_interest_graph['strike'], y=option_data_open_interest_graph['call/put_ratio_Openinterest'], name='call/put_ratio_Openinterest'), secondary_y=True)
            # Set y-axes titles
            option_ratios_graph.update_yaxes(title_text="call/put_ratio_<b>Volume</b>", secondary_y=False)
            option_ratios_graph.update_yaxes(title_text="call/put_ratio_<b>Openinterest</b>", secondary_y=True)
            option_ratios_graph.update_xaxes(title="Strike Price")
            option_ratios_graph.update_layout(title="Stock Option Chain Ratio's")
            st.plotly_chart(option_ratios_graph, use_container_width=True)
            st.write("1.) Ratio used for chart above is based off said metrics calls / the same metrics puts. Trying to identify if there are any trends of people being call vs put heavy.")
            st.write("2.) Blue line is the indicator for Volume of options executed, Red line is the indicator for Openinterst in the market not yet executed.")

            ## creat volume and openinterst graphs
            #option_line_fig = make_subplots(rows=1, cols=2, column_titles=('Calls', 'Puts'))
            #fig = make_subplots(rows=1, cols=2, subplot_titles=("Calls", "Puts"))
            #calls_volume1 = px.line(calls_clean, x="strike", y="Volume", title="Volume")
            #calls_openinterest1 = px.line(calls_clean, x="strike", y="Openinterest", title="Openinterest")

            #puts_volume1 = px.line(puts_clean, x="strike", y="Volume", title="Volume")
            #puts_openinterest1 = px.line(puts_clean, x="strike", y="Openinterest", title="Openinterest")

            #trace_calls_volume = calls_volume1['data'][0]
            #trace_calls_openinterest = calls_openinterest1['data'][0]
            #trace_puts_volume = puts_volume1['data'][0]
            #trace_puts_openinterest = puts_openinterest1['data'][0]

            #option_line_fig.add_trace(trace_calls_volume, row=1, col=1)
            #option_line_fig.add_trace(trace_calls_openinterest, row=1, col=1)
            #option_line_fig.add_trace(trace_puts_volume, row=1, col=2)
            #option_line_fig.add_trace(trace_puts_openinterest, row=1, col=2)
            #option_line_fig.update_layout(title="Open Interest and Volume by Strike Price")
            #st.plotly_chart(option_line_fig, use_container_width=True)

            ### tried to group data, but removes the ability to see which exirpy date the data is assocaited with.

            #option_data_volume = pd.DataFrame(option_data_new.groupby(['strike', 'type']).agg({'Volume': 'sum'})).reset_index()
            #st.write(option_data_volume.astype('object'))
            #st.plotly_chart(px.bar(option_data_volume, x="strike", y="Volume", color="type", title="Volume"))

            
            st.write("## Volume and Open Interest by strike price")
            col9, col10 = st.beta_columns(2)
            with col9:
                st.plotly_chart(px.bar(option_data_new, x="strike", y="Volume", color="type", hover_data=['Openinterest', 'expiryDate'], barmode = 'stack', title="Volume"))
            
            with col10:
                st.plotly_chart(px.bar(option_data_new, x="strike", y="Openinterest", color="type", hover_data=['Volume', 'expiryDate'], barmode = 'stack', title="Openinterest"))
            
            
            st.plotly_chart(px.bar(option_data_new, x="expirygroup", y="Volume", color="type", hover_data=['Openinterest', 'strike'], barmode = 'stack', title="Volume of Exercised Options"), use_container_width=True)
            st.plotly_chart(px.bar(option_data_new, x="expirygroup", y="Openinterest", color="type", hover_data=['Volume', 'strike'], barmode = 'stack', title="Openinterest of unexecuted options"), use_container_width=True)
            
            st.write('Open Interest by Date, looking to see where the option activity is happening.')
            bar_fig = make_subplots(rows=1, cols=2, column_titles=('Calls', 'Puts'))
            #fig = make_subplots(rows=1, cols=2, subplot_titles=("Calls", "Puts"))
            bar1 = px.bar(calls_clean, x="expirygroup", y="Openinterest", title="Calls")
            bar2 = px.bar(puts_clean, x="expirygroup", y="Openinterest", title="Puts")

            trace3 = bar1['data'][0]
            trace4 = bar2['data'][0]
            bar_fig.add_trace(trace3, row=1, col=1)
            bar_fig.add_trace(trace4, row=1, col=2)
            #fig.update_layout(title="Open Interest by Strike Price, size by volume of options that have been exercised")
            st.plotly_chart(bar_fig, use_container_width=True)
            
            
            
            #st.write('Open Interest by Strike Price, size by volume of options that have been exercised')
            scatter_fig = make_subplots(rows=1, cols=2, column_titles=('Calls', 'Puts'))
            #fig = make_subplots(rows=1, cols=2, subplot_titles=("Calls", "Puts"))
            scatter1 = px.scatter(calls_clean, x="strike", y="Openinterest", size ="Volume", title="Calls")
            scatter2 = px.scatter(puts_clean, x="strike", y="Openinterest", size ="Volume", title="Puts")

            trace5 = scatter1['data'][0]
            trace6 = scatter2['data'][0]
            scatter_fig.add_trace(trace5, row=1, col=1)
            scatter_fig.add_trace(trace6, row=1, col=2)
            scatter_fig.update_layout(title="Open Interest by Strike Price, size by volume of options that have been exercised")
            st.plotly_chart(scatter_fig, use_container_width=True)
            st.write(option_data_new.astype('object'))
            st.markdown(get_table_download_link(option_data_new), unsafe_allow_html=True)