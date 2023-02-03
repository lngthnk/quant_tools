# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 09:29:32 2023

@author: phata
"""

# SET from yf
# Already adjusted for dividends

import yfinance as yf

def get_data_1d(symbol):
    ticker = yf.Ticker(symbol + '.BK')
    df = ticker.history(period="max")[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = df.columns.str.lower()
    df.index = df.index.rename('date')
    return df
