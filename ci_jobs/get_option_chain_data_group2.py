import pandas as pd
import numpy as np
from pandas import json_normalize
import json
#from selenium.webdriver.support.expected_conditions import element_selection_state_to_be
#import matplotlib.pyplot as plt
import base64
#import matplotlib.pyplot as plt
import io
import os
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
    ticker_selection = pd.read_csv(r'data/tickers_only.csv')
    ticker_selection = ticker_selection[ticker_selection['group'] == 2]
    tickers = ticker_selection['ticker']
    
    response_dict = {"ticker":[],"option_data":[]}
    error_list = []

    start = time.time()

    print("Starting to gather option data...")
    
    for i in tickers:
        
        time.sleep(1)
        
        try:
            base1 = "https://api.nasdaq.com/api/quote/"
            base2 = "/option-chain?assetclass=stocks&fromdate=all&todate=undefined&excode=oprac&callput=callput&money=all&type=all"
            url = base1 + str(i) + base2

            payload={}

            #headers = {'server':'Kestrel', 'Accept':'application/json, text/plain, */*','Accept-Encoding': 'gzip, deflate, br', 'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36'}
            headers = {'User-Agent': 'PostmanRuntime/7.28.4','Accept': '*/*','Accept-Encoding': 'gzip, deflate, br'}

            response = requests.get(url, headers=headers, data=payload)

            response_text = json.loads(response.text)
            response_dict['ticker'].append(i)
            response_dict['option_data'].append(response_text)
            print("Loaded Option Chain Data for:", i)

        except Exception as e:
            error_list.append([i, e])
            print("Error:", e)

    # create json object from dictionary
    big_dict = json.dumps(response_dict)

    print("open file for writing, w")
    f = open("data/option_data_group2_dict.json","w")

    print("write json object to file")
    f.write(big_dict)

    print("close file")
    f.close()
    
    #big_df = pd.DataFrame(response_dict)
    #big_df.to_csv(r'data/option_chain_data_group1.csv')

    end = time.time()
    print("Gathering Stock Tickers Took...", round((end - start)/60, 2), "minutes")
    print("Complete!")

    return big_dict

big_dict = gather_tickers()