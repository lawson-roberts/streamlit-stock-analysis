import requests
import time
import pandas as pd
import numpy as np
from lxml import html
import csv
import chromedriver_autoinstaller
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.select import Select

"""
## This script is for scraping available stock tickers. Having a list available to choose from will increase user expereince by enabling easier searching of companies.
"""

def get_tickers():

    """
    ## This function is for scraping the available stock tickers from the website https://stockanalysis.com/stocks/.
    ## We will be using a web scraping technique "xpath". This is essentially reading a websites html code and getting the elements you want.
    """

    web = "https://stockanalysis.com/stocks/"
    #driver = webdriver.Chrome(r'C:\Users\rober\Anaconda3\bin\chromedriver')
    driver_path = chromedriver_autoinstaller.install()

    driver = webdriver.Chrome(driver_path)
    driver.get(web)
    sel = Select(driver.find_element_by_xpath('//select[@name="perpage"]'))
    sel.select_by_visible_text("10000")
    print("Selected All Tickers")

    time.sleep(5)

    ticker_list = []
    company_name_list = []
    industry_list = []

    ## starting to find elements
    ticker = driver.find_elements_by_xpath('//*[@id="main"]/div/div/div[2]/table/tbody/tr/td[1]/a')
    for a in range(len(ticker)):
        ticker_list.append(ticker[a].text)

    company_name = driver.find_elements_by_xpath('//*[@id="main"]/div/div/div[2]/table/tbody/tr/td[2]')
    for b in range(len(company_name)):
        company_name_list.append(company_name[b].text)

    industry = driver.find_elements_by_xpath('//*[@id="main"]/div/div/div[2]/table/tbody/tr/td[3]')
    for c in range(len(industry)):
        industry_list.append(industry[c].text)

    ## Creating dataframes so I can join this all together
    ticker_df = pd.DataFrame(ticker_list)
    print("Ticker DataFrame Created...")
    company_name_df = pd.DataFrame(company_name_list)
    print("Company Name DataFrame Created...")
    industry_df = pd.DataFrame(industry_list)
    print("Industry DataFrame Created...")
    big_df = pd.concat([ticker_df, company_name_df, industry_df], axis=1)
    print("Big DataFrame Created...")

    return big_df

def clean_tickers(big_df):

    """
    ## This function is for cleaning the original list of tickers. There are some items from the list that are either not legit tickers that we remove here.
    ## The last part of this function is creating a dataframe that we then use to create some additional features to help with searching.
    """

    ## creating a couple new columns to create a link and full searchable stock name plus ticker to help with app search.
    big_df.columns = ['ticker', 'company_name', 'industry']
    big_df['full'] = big_df['ticker'] + " - " + big_df['company_name']
    big_df['url'] = "https://stockanalysis.com/stocks/" + big_df['ticker'] + "/"
    big_df['full'] = big_df['full'].astype('str')
    big_df['ticker'] = big_df.ticker.str.strip()
    big_df['group'] = big_df.groupby(np.arange(len(big_df.index))//(len(big_df)/10),axis=0).ngroup()+1
    print("Big DataFrame Cleaned...")

    ## create csv to use for streamlit app
    big_df.to_csv(r'data/tickers_only_test.csv')
    print("DataFrame saved as csv :)")

    return big_df

big_df = get_tickers()
ticker_df = clean_tickers(big_df)