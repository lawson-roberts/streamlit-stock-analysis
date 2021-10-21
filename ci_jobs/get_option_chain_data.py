import pandas as pd
import numpy as np
from pandas import json_normalize
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
import json

def gather_tickers():
    ##importing files needed for web app
    start = time.time()
    print("Starting to gather option data...")

    # Opening JSON file
    one = open('data/option_data_group1_dict.json',)
    # returns JSON object as a dictionary
    one_response_dict = json.load(one)
    print("First File Loaded...")

    two = open('data/option_data_group2_dict.json',)
    two_response_dict = json.load(two)
    print("Second File Loaded...")

    three = open('data/option_data_group3_dict.json',)
    three_response_dict = json.load(three)
    print("Third File Loaded...")

    four = open('data/option_data_group4_dict.json',)
    four_response_dict = json.load(four)
    print("Fourth File Loaded...")

    five = open('data/option_data_group5_dict.json',)
    five_response_dict = json.load(five)
    print("Fifth File Loaded...")

    six = open('data/option_data_group6_dict.json',)
    six_response_dict = json.load(six)
    print("Sixth File Loaded...")

    seven = open('data/option_data_group7_dict.json',)
    seven_response_dict = json.load(seven)
    print("Seventh File Loaded...")

    eight = open('data/option_data_group8_dict.json',)
    eight_response_dict = json.load(eight)
    print("Eight File Loaded...")

    nine = open('data/option_data_group9_dict.json',)
    nine_response_dict = json.load(nine)
    print("Ninth File Loaded...")

    ten = open('data/option_data_group10_dict.json',)
    ten_response_dict = json.load(ten)
    print("Tenth File Loaded...")

    print("Combining Dictionary")
    all_tickers = {**one, **two, **three, **four, **five, **six, **seven, **eight, **nine, **ten}

    print("open file for writing, w")
    f = open("option_chain_data_all.json","w")

    print("write json object to file")
    f.write(all_tickers)

    print("close file")
    f.close()

    #all_tickers = group1.append([group2, group3, group4, group5, group6, group7, group8, group9, group10])
    #all_tickers.to_csv(r'data/option_chain_data_all.csv')
    end = time.time()
    print("Gathering Stock Tickers Took...", round((end - start)/60, 2), "minutes")
    print("Complete!")

    return all_tickers

response_df = gather_tickers()