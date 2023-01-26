# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:27:23 2023

@author: phata
"""

import datetime as dt
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

symbol = 'BTCUSDT'

klines_1yr = client.get_historical_klines(symbol, '1d', start_str = dt.datetime(2022, 1, 1).strftime('%Y-%m-%d'), end_str = dt.datetime(2023, 1, 1).strftime('%Y-%m-%d'))



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

df = pd.DataFrame(klines,columns=column_names)







