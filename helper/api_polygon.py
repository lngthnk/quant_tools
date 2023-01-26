# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:51:42 2023

@author: phata
"""

from polygon import RESTClient


api_key = 'D8MlxgaM5AnoaGLIF8gT3Si1agzcU88r'
client = RESTClient(api_key=api_key)

ticker = "AAPL"

# List Aggregates (Bars)
bars = client.get_aggs(ticker=ticker, multiplier=1, timespan="day", from_="2022-01-01", to="2023-01-01")
for bar in bars:
    print(bar)
list_tickers = client.list_tickers(market='stocks')
for ticker in list_tickers:
    print(ticker)