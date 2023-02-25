# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:51:33 2023

@author: phata
"""

import streamlit as st
import datetime as dt
import pandas as pd

# Import local lib
import helper.binance_api as ba

# Read-only key
api_key \
    = 'EWvN6DTAdPgnuIMfPlqnB72P1uAROB6pwjnFNpfML7vf7EehaSgvvqTsnTQFUcDJ'
api_secret \
    = 'TcylGKbkCroyYvNTCQFoQhZOq0abvts81BWfpMZIb9Fo6Wm1sXtKGVGzXJaRtmWz'

client \
    = ba.Client(api_key, api_secret)

exchange_info \
    = client.get_exchange_info()
    
list_of_symbols_info \
    = exchange_info['symbols']


list_of_quote_assets \
    = sorted(list(dict.
                  fromkeys([dict['quoteAsset'] 
                            for dict 
                            in list_of_symbols_info]
                           )))

quote_asset = st.selectbox('Quote Asset', list_of_quote_assets)

list_of_base_assets_given_quote_asset \
    = sorted(list(dict.fromkeys([dict['baseAsset'] 
                                 for dict 
                                 in list_of_symbols_info 
                                 if dict['quoteAsset'] == quote_asset]
                                )))
    
base_asset = st.selectbox('Available Base Asset', list_of_base_assets_given_quote_asset)

symbol = base_asset + quote_asset

st.write('Selected symbol: ' + symbol)



klines \
    = client.get_historical_klines(symbol, '1d')
li_dates \
    = [li[0] for li in klines]
ts_start, ts_end \
    = li_dates[0], li_dates[-1]
dt_start, dt_end \
    = (dt.datetime.utcfromtimestamp(ts_start/1000), 
       dt.datetime.utcfromtimestamp(ts_end/1000))
year_start, year_end \
    = dt_start.year, dt_end.year
year \
    = st.slider('Choose year', year_start, year_end, year_start)

klines_of_a_year = client.get_historical_klines(symbol, '1d', start_str = dt.datetime(year, 1, 1).strftime('%Y-%m-%d'), end_str = dt.datetime(year, 12, 31).strftime('%Y-%m-%d'))


column_names = ['open_time', 
                'open', 
                'high', 
                'low', 
                'close', 
                'volume', 
                'close_time', 
                'quote_asset_volume', 
                'number_of_trades', 
                'taker_buy_base_asset_volume', 
                'taker_buy_quote_asset_volume', 
                'unused_field_ignore']

df = pd.DataFrame(klines_of_a_year,columns=column_names)

st.write(df)