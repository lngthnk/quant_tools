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
@st.experimental_memo
def get_list_of_symbols(exchange: str):
    if exchange == "Binance":
        return ba.list_of_symbols
    elif exchange == "SET":
        return list_of_symbols_set
    else:
        raise Exception("Bug")

@st.experimental_memo
def get_data_1d(exchange: str, symbol: str, str_start_date = None, str_end_date = None):
    if exchange == "Binance":
        return ba.get_data_1d(symbol)
    elif exchange == "SET":
        return ya.get_data_1d(symbol)
    else:
        raise Exception("Bug")  
##########


st.write('Technical Backtester')

sb_exchange \
    = st.selectbox("Exchange",
                   ("Binance", "SET")
                   )

if sb_exchange == "Binance":
    list_of_symbols = ba.list_of_symbols
elif sb_exchange == "SET":
    list_of_symbols = list_of_symbols_set
else:
    raise Exception("Bug")

sb_symbol \
    = st.selectbox(
        "Symbol",
        list_of_symbols
        )
    
# Get df
df = get_data_1d(sb_exchange, sb_symbol)

st.write(df)

# sb_trade_direction
sb_trade_direction \
    = st.selectbox("Trade Direction",
                   ("Long", "Short")
                   )

# sb_and_or_or
sb_and_or_or \
    = st.selectbox("AND or OR",
                   ("AND", "OR")
                   )

# Indicator
st.write(sb_trade_direction + ' when')

if 'n_indicators' not in st.session_state:
    st.session_state.n_indicators = 1

col1, col2, col3 = st.columns(3)
for i in range(st.session_state.n_indicators):
    with col1: 
        indicator_i = st.selectbox(f'Indicator {i}',('SMA',), key = f'indicator_{i}')
    with col2: 
        sign_i = st.selectbox('Sign',('>', '<'), key = f'sign_{i}')
    with col3: 
        value_i = st.selectbox('Value', range(10, 201, 10), key = f'value_{i}')

# Add indicator
def add_indicator():
    st.session_state.n_indicators += 1
st.button('Add Indicator', on_click=add_indicator)
    
# sb_trade_other_direction
if sb_trade_direction == "Long":
    sb_trade_other_direction_label = 'Allow shorting otherwise?'
else:
    sb_trade_other_direction_label = 'Allow longing otherwise?'
sb_trade_other_direction \
    = st.checkbox(sb_trade_other_direction_label)
    
# if 'backtest_results' not in st.session_state:
#     st.session_state.backtest_results = 'No results yet'

# def backtest_run():
#     st.session_state.backtest_results = 'Results here'
    
    
# st.button('Run', on_click=backtest_run)

# st.write('Results:', st.session_state.backtest_results)