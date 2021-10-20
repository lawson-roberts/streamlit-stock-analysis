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

def split_into_bins():

    tickers = pd.read_csv("data/tickers_only.csv")

    tickers['cut'] = pd.qcut(len(tickers), 10).value_counts()