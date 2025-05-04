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

def app():

    #st.title('Stock Trends')
    st.write("""
             # Coming Soon!
             """)

    construction = Image.open('construction.jpg')
    st.image(construction, caption='Page Under Construction...Come Back Later', width=680)
    #st.write("""## Data Sources:""")
    #st.write("""1.) yfinance python package""")
