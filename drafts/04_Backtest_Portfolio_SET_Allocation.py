import streamlit as st

import pandas as pd

# import helper.binance_api as ba
import helper.yahoo_api as ya

st.set_page_config(
    page_title='Backtest Portfolio SET Allocation'
)

# Data functions
df_of_symbols_set = pd.read_excel('C:/Users/phata/Documents/GitHub/quant_tools/helper/set_symbol_list.xlsx')
list_of_symbols_set = df_of_symbols_set['symbol'].to_list()

# Choose stocks
if 'n_stocks' not in st.session_state:
    st.session_state.n_stocks = 1
    
# Initialize vars

if 'df' not in st.session_state:
    st.session_state.df = None

# Add and remove stocks
def add_stock():
    st.session_state.n_stocks += 1
def remove_last_stock():
    st.session_state.n_stocks -= 1

col1, col2, col3, col4, col5 = st.columns(5)
col1.markdown('**Asset Allocation**')
col2.markdown('**Stock Symbol**')
col3.markdown('**Portfolio #1 (%)**')
col4.markdown('**Portfolio #2 (%)**')
col5.markdown('**Portfolio #3 (%)**')

li_selected_symbols = []
li_portfolio_1 = []
li_portfolio_2 = []
li_portfolio_3 = []

for i_row in range(0, st.session_state.n_stocks):
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.write(f'Asset {i_row + 1}')
    
    li_selected_symbols.append(
        col2
        .selectbox(
            "Symbol", 
            list_of_symbols_set, 
            key = f'symbol_{i_row + 1}', 
            label_visibility = 'collapsed'
            )
        )
    
    li_portfolio_1.append(
        col3
        .number_input(
            'Portfolio 1', 
            min_value=0, 
            max_value=100,
            format='%i', 
            key=f'portfolio_1_asset_{i_row + 1}', 
            label_visibility='collapsed')
        )

    li_portfolio_2.append(
        col4
        .number_input(
            'Portfolio 2', 
            min_value=0, 
            max_value=100,
            format='%i', 
            key=f'portfolio_2_asset_{i_row + 1}', 
            label_visibility='collapsed')
        )

    li_portfolio_3.append(
        col5
        .number_input(
            'Portfolio 3', 
            min_value=0, 
            max_value=100,
            format='%i', 
            key=f'portfolio_3_asset_{i_row + 1}', 
            label_visibility='collapsed')
        )
    
col1, col2 = st.columns(2)
col1.button('Add more', on_click=add_stock)
col2.button('Remove latest', on_click=remove_last_stock)

# Get data as df
@st.cache_data
def get_data_1d_SET(symbol: str):
    series = ya.get_data_1d(symbol).close.rename(symbol)
    return series[series > 0] # Quick fix for negative prices
@st.cache_data
def get_df(li_symbols, int_start_year = None, int_end_year = None):
    df = pd.concat([get_data_1d_SET(symbol) for symbol in li_symbols], axis = 1)
    if int_start_year is not None: df = df[df.index.year >= int_start_year]
    if int_end_year is not None: df = df[df.index.year <= int_end_year]
    st.session_state.df = df
    return df
#

from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage

st.button('Run', on_click=get_df, args=(li_selected_symbols,))

if st.session_state.df is not None:
    st.write(st.session_state.df)

mu = mean_historical_return(st.session_state.df)
S = CovarianceShrinkage(st.session_state.df).ledoit_wolf()

# Plots (Doesn't use the weights)
import matplotlib.pyplot as plt
from pypfopt import plotting
fig, ax = plt.subplots()
plotting.plot_covariance(S, plot_correlation=True, show_tickers=True, ax=ax)
st.pyplot(fig)
plt.show()
