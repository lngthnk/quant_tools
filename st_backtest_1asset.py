# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 13:02:36 2023

@author: phata
"""
import streamlit as st
import pandas as pd

import helper.binance_api as ba
import helper.yahoo_api as ya

df_of_symbols_set = pd.read_excel('C:/Users/phata/Documents/GitHub/quant_tools/helper/set_symbol_list.xlsx')
list_of_symbols_set = df_of_symbols_set['symbol'].to_list()


    
##########
def get_list_of_symbols(exchange: str):
    if exchange == "Binance":
        return ba.list_of_symbols
    elif exchange == "SET":
        return list_of_symbols_set
    else:
        raise Exception("Bug")

def get_data_1d(exchange: str, symbol: str, str_start_date = None, str_end_date = None):
    if exchange == "Binance":
        return ba.get_data_1d(symbol)
    elif exchange == "SET":
        return ya.get_data_1d(symbol)
    else:
        raise Exception("Bug")
##########

add_sidebar \
    = st.sidebar.selectbox('Select tool', 
                           ('Technical Backtester',
                            'Dummy'
                            )
                           )

if add_sidebar == 'Technical Backtester':
    
    st.write('Technical Backtester')
    
    exchange \
        = st.selectbox(
            "Exchange",
            ("Binance", "SET")
            )
    
    if exchange == "Binance":
        list_of_symbols = ba.list_of_symbols
    elif exchange == "SET":
        list_of_symbols = list_of_symbols_set
    else:
        raise Exception("Bug")
    
    symbol \
        = st.selectbox(
            "Symbol",
            list_of_symbols
            )
    
     
    st.button('Run: currently doesn\'t work', on_click=None)
    
    # Get df
    df = get_data_1d(exchange, symbol)
    
    st.write(df)
    
    # Indicator
    st.write('Long when')
    col11, col12, col13 = st.columns(3)
    with col11: 
        indicator_1 = st.selectbox('Indicator 1',('SMA',), key = 'indicator_1')
    with col12: 
        sign_1 = st.selectbox('Sign',('>', '<'), key = 'sign_1')
    with col13: 
        value_1 = st.selectbox('Value', range(10, 201, 10), key = 'value_1')
        
    st.write('Short when')
    col21, col22, col23 = st.columns(3)
    with col21: 
        indicator_2 = st.selectbox('Indicator 2',('SMA',), key = 'indicator_2')
    with col22: 
        sign_2 = st.selectbox('Sign',('>', '<'), key = 'sign_2')
    with col23: 
        value_2 = st.selectbox('Value', range(10, 201, 10), key = 'value_2')