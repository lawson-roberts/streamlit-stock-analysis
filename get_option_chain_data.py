import pandas as pd
import numpy as np
from pandas import json_normalize
import json
#from selenium.webdriver.support.expected_conditions import element_selection_state_to_be
#import matplotlib.pyplot as plt
import base64
#import matplotlib.pyplot as plt
import io
from math import floor

## web crawling packages
import time
from datetime import date
from datetime import datetime
import datetime
import requests
from lxml import html
import csv


def gather_tickers():
    ##importing files needed for web app
    ticker_selection = pd.read_csv('data/tickers_only.csv')
    tickers = ticker_selection['ticker']
    
    response_list = []
    error_list = []
    
    for i in tickers:
        
        try:
            time.sleep(2)
        
            base1 = "https://api.nasdaq.com/api/quote/"
            base2 = "/option-chain?assetclass=stocks&fromdate=all&todate=undefined&excode=oprac&callput=callput&money=all&type=all"
            url = base1 + str(i) + base2

            payload={}

            #headers = {'server':'Kestrel', 'Accept':'application/json, text/plain, */*','Accept-Encoding': 'gzip, deflate, br', 'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36'}
            headers = {'User-Agent': 'PostmanRuntime/7.28.4','Accept': '*/*','Accept-Encoding': 'gzip, deflate, br'}

            response = requests.get(url, headers=headers, data=payload)

            response_text = json.loads(response.text)
            response_list.append(response_text)
            print("Loaded Option Chain Data for:", i)

        except Exception as e:
            error_list.append([i, e])
            print("Error:", e)

    option_chain_df = pd.DataFrame(response_list, columns = ['ticker', 'option_data'])
    option_chain_df.to_csv('data/option_chain_data.csv')
    error_df = pd.DataFrame(error_list, columns = ['ticker', 'error_desc'])
    error_df.to_csv('data/option_chain_errors.csv')
    return option_chain_df, error_df

option_chain_df, error_df = gather_tickers()