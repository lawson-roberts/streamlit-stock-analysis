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

## sentiment analysis
import re
import nltk
nltk.download('vader_lexicon')
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

import home
import stock_trends
import stock_analysis
import crypto
import option_chain_activity
import option_chain_activity_copy

PAGES = {
    "Home": home,
    "Stock Analysis": stock_analysis,
    "Stock Trends": stock_trends,
    "Crypto": crypto,
    "Option Chain Analysis": option_chain_activity_copy
}

##Setting Streamlit Settings
st.set_page_config(layout="wide")
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
