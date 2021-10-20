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
    start = time.time()
    print("Starting to gather option data...")
    group1 = pd.read_csv(r'data\option_chain_data_group1.csv')
    group2 = pd.read_csv(r'data\option_chain_data_group2.csv')
    group3 = pd.read_csv(r'data\option_chain_data_group3.csv')
    group4 = pd.read_csv(r'data\option_chain_data_group4.csv')
    group5 = pd.read_csv(r'data\option_chain_data_group5.csv')
    group6 = pd.read_csv(r'data\option_chain_data_group6.csv')
    group7 = pd.read_csv(r'data\option_chain_data_group7.csv')
    group8 = pd.read_csv(r'data\option_chain_data_group8.csv')
    group9 = pd.read_csv(r'data\option_chain_data_group9.csv')
    group10 = pd.read_csv(r'data\option_chain_data_group10.csv')

    all_tickers = group1.append([group2, group3, group4, group5, group6, group7, group8, group9, group10])
    
    all_tickers.to_csv(r'data/option_chain_data_all.csv')
    end = time.time()
    print("Gathering Stock Tickers Took...", round((end - start)/60, 2), "minutes")
    print("Complete!")

    return all_tickers

response_df = gather_tickers()