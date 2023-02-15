# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:27:23 2023

@author: phata
"""

import pandas as pd

# pip install python-binance
from binance.client import Client

# Read-only key
api_key \
    = 'EWvN6DTAdPgnuIMfPlqnB72P1uAROB6pwjnFNpfML7vf7EehaSgvvqTsnTQFUcDJ'
api_secret \
    = 'TcylGKbkCroyYvNTCQFoQhZOq0abvts81BWfpMZIb9Fo6Wm1sXtKGVGzXJaRtmWz'

client \
    = Client(api_key, api_secret)

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

quote_asset \
    = 'USDT'

list_of_base_assets_given_quote_asset \
    = sorted(list(dict.fromkeys([dict['baseAsset'] 
                                 for dict 
                                 in list_of_symbols_info 
                                 if dict['quoteAsset'] == quote_asset]
                                )))

list_of_symbols_given_quote_asset \
    = sorted(list(dict.
                  fromkeys([dict['symbol'] 
                            for dict 
                            in list_of_symbols_info
                            if dict['quoteAsset'] == quote_asset]
                           )))

list_of_symbols \
    = sorted(list(dict.
                  fromkeys([dict['symbol'] 
                            for dict 
                            in list_of_symbols_info]
                           )))

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

def get_data_1d(symbol: str, str_start_date = None, str_end_date = None):
    klines = client.get_historical_klines(symbol, 
                                          '1d', 
                                          start_str = str_start_date, 
                                          end_str = str_end_date)
    # Ignore last row to make [start_date,end_date)
    df = pd.DataFrame(klines, columns = column_names)[0:-1]
    df['open_time'] = pd.to_datetime(df['open_time'],unit='ms')
    df = df.rename(columns = {'open_time': 'date'})
    df = df.set_index('date')
    return df[['open', 'high', 'low', 'close', 'volume']]





