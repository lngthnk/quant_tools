import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import riskfolio as rp

import helper.yahoo_api as ya

st.set_page_config(
        page_title="Covariance Methods and Optimal Portfolio",
)

df_of_symbols_set = pd.read_excel('https://github.com/phat-ap/quant_tools/blob/main/helper/set_symbol_list.xlsx?raw=true')
list_of_symbols_set = df_of_symbols_set['symbol'].to_list()

# function: a df of OHLC and volume
@st.cache_data
def get_data_1d_SET(symbol: str, int_start_year = None, int_end_year = None): 
    df = ya.get_data_1d(symbol)
    if int_start_year is not None: df = df[df.index.year >= int_start_year]
    if int_end_year is not None: df = df[df.index.year <= int_end_year]
    return df[df > 0] # Quick fix for negative prices

'''
### Choose at least 2 stocks
'''
cols = st.columns(5)
disabled = False
for i in range(5):
    if i > 0:
        if st.session_state[f'stock_{i-1}'] == '':
            disabled = True
    cols[i%5].selectbox(f'Stock {i+1}', 
                        [''] + list_of_symbols_set, 
                        key = f'stock_{i}', 
                        disabled = disabled
                        )
cols = st.columns(5)
for i in range(5, 10):
    if i > 0:
        if st.session_state[f'stock_{i-1}'] == '':
            disabled = True
    cols[i%5].selectbox(f'Stock {i+1}', 
                        [''] + list_of_symbols_set, 
                        key = f'stock_{i}', 
                        disabled = disabled
                        )
    
li_stocks = [st.session_state[f'stock_{i}'] 
             for i in range(10) 
             if st.session_state[f'stock_{i}'] != '']
# st.write(li_stocks)

if len(li_stocks) >= 1:
    '''
    ### Start and end date
    '''
    df = yf.download([str + '.BK' for str in li_stocks])['Close']
    df.columns = df.columns.str.rstrip('.BK')
    df_returns = df.pct_change().dropna()
    
    st.write(df_returns)
    
    # From https://nbviewer.org/github/dcajasn/Riskfolio-Lib/blob/master/examples/Tutorial%2034.ipynb
    
    port = rp.Portfolio(returns=df_returns)
    method_mu='hist' # Method to estimate expected returns based on historical data.
    method_covs = ['hist', 'ledoit', 'oas', 'shrunk', 'gl', 'ewma1',
                   'ewma2','jlogo', 'fixed', 'spectral', 'shrink',
                   'gerber1', 'gerber2']
    model='Classic'
    rm = 'MV'
    obj = 'Sharpe'
    hist = True
    rf = 0
    l = 0

    w_s = pd.DataFrame([])
    
    for i in method_covs:
        port.assets_stats(method_mu=method_mu, method_cov=i, d=0.94)
        w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
        w_s = pd.concat([w_s, w], axis=1)
            
    w_s.columns = method_covs
    
    st.write(w_s.style.format("{:.2%}").background_gradient(cmap='YlGn'))
    