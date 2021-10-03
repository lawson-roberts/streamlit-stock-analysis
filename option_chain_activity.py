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
from datetime import date
import datetime
import time

##Crypto packages
import finnhub


#def app():


    ##Setting Streamlit Settings
    #st.set_page_config(layout="wide")

    ##importing files needed for web app

    #today = date.today()
    #year_ago = today - datetime.timedelta(days=365)
    #unixtime_today = time.mktime(today.timetuple())
    #unixtime_year = time.mktime(year_ago.timetuple())
    
    #finnhub_client = finnhub.Client(api_key="c3qcjnqad3i9vt5tl68g")