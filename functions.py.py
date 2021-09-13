import streamlit as st
import base64
from PIL import Image

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

PAGES = {
    "Home": home,
    "Stock Analysis": stock_analysis,
    "Stock Trends": stock_trends
}

##Setting Streamlit Settings
st.set_page_config(layout="wide")
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()