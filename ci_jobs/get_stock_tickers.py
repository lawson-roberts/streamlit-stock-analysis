import requests
import time
import pandas as pd
import numpy as np
from lxml import html
import csv
import finnhub

"""
## This script is for scraping available stock tickers. Having a list available to choose from will increase user expereince by enabling easier searching of companies.
"""
def get_tickers():

    """
    ## This function is for cleaning the original list of tickers. There are some items from the list that are either not legit tickers that we remove here.
    ## The last part of this function is creating a dataframe that we then use to create some additional features to help with searching.
    """

    # Setup client
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise ValueError("FINNHUB_API_KEY environment variable is not set.")
    finnhub_client = finnhub.Client(api_key=api_key)

    symbols_df = pd.DataFrame(finnhub_client.stock_symbols('US'))
    symbols_df_short = symbols_df[['displaySymbol', 'description']]
    
    ## creating a couple new columns to create a link and full searchable stock name plus ticker to help with app search.
    symbols_df_short['full'] = symbols_df_short['displaySymbol'] +  " - " + symbols_df_short['description']
    symbols_df_short['url'] = "https://stockanalysis.com/stocks/" + symbols_df_short['displaySymbol'] + "/"
    symbols_df_short = symbols_df_short.rename(columns = {'displaySymbol': 'ticker', 'description': 'company_name'})
    symbols_df_short['group'] = symbols_df_short.groupby(np.arange(len(symbols_df_short.index))//(len(symbols_df_short)/10),axis=0).ngroup()+1
    symbols_df_short['full'] = symbols_df_short['full'].astype('str')
    symbols_df_short['ticker'] = symbols_df_short.ticker.str.strip()
    print("Big DataFrame Cleaned...")

    ## create csv to use for streamlit app
    symbols_df_short.to_csv(r'data/tickers_only.csv')
    print("DataFrame saved as csv :)")

    return symbols_df_short

symbols_df_short = get_tickers()
