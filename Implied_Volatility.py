import numpy as np
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
import tf_quant_finance as tff 
option_price = tff.black_scholes.option_price
implied_vol = tff.black_scholes.implied_vol
import yfinance as yf
from datetime import datetime
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Option Pricing & Implied Volatility"
)

'''
# Option Pricing & Implied Volatility

This webapp compares the implied volatility of options of stocks under the S&P 500 given by YahooFinance with closed-form calculations using tf_quant_finance\'s black_scholes.implied_vol function. The interest rate per year defaults at 5%.

Deviation of the market implied volatility from the model\'s calculation may be an opportunity for investors to study in depth, if there are arbitrage opportunities to be exploited.
 
'''

@st.cache_data
def get_ticker(symbol: str):
    return yf.Ticker(symbol)

df_of_symbols_sp500 = (pd
                       .read_excel('https://github.com/phat-ap/quant_tools/blob/main/helper/sp500_symbol_list.xlsx?raw=true')
                       .drop(['#', 'Weight', 'Price', 'Chg', '% Chg'], axis = 1)
                       .set_index('Symbol'))


symbol = st.selectbox('Select Stock', df_of_symbols_sp500.sort_index().index)
ticker = get_ticker(symbol)

str_expiry_date = st.selectbox('Select Expiry Date', ticker.options)

df_calls, df_puts = ticker.option_chain(str_expiry_date)

for df in [df_calls, df_puts]:
    df['mid_price'] = (df['bid'] + df['ask']) / 2
df_calls = df_calls[['strike', 'mid_price', 'impliedVolatility']]
df_puts = df_puts[['strike', 'mid_price', 'impliedVolatility']]

dt_expiry_date = datetime.strptime(str_expiry_date, "%Y-%m-%d")
expiry = (dt_expiry_date - datetime.utcnow()).total_seconds() / 31536000

# Parameters
initial_volatilities = 1
rate = st.number_input('Interest rate per annum (%)', value = 5) / 100

# Input from df
strikes = df_calls['strike'].values
prices = df_calls['mid_price'].values
expiries = expiry
discount_factors = np.exp(-rate * expiries)

# Current value of assets.
spots = ticker.history(period='1d')['Close'][0]
# Forward value of assets at expiry.
forwards = spots / discount_factors

# Find the implied vols beginning at initial_volatilities.
implied_vols_calls = implied_vol(
    prices=df_calls['mid_price'].values,
    strikes=df_calls['strike'].values,
    expiries=expiry,
    forwards=forwards,
    discount_factors=discount_factors,
    is_call_options=True,
    initial_volatilities=initial_volatilities,
    validate_args=True,
    tolerance=1e-9,
    max_iterations=200,
    name=None,
    dtype=None)
implied_vols_puts = implied_vol(
    prices=df_puts['mid_price'].values,
    strikes=df_puts['strike'].values,
    expiries=expiry,
    forwards=forwards,
    discount_factors=discount_factors,
    is_call_options=False,
    initial_volatilities=initial_volatilities,
    validate_args=True,
    tolerance=1e-9,
    max_iterations=200,
    name=None,
    dtype=None)

st.markdown(f"<h3 style='text-align: center; '>{symbol}: {df_of_symbols_sp500.loc[symbol].Company}</h3>", unsafe_allow_html = True)

import plotly.graph_objects as go

col1, col2 = st.columns(2)

fig = go.Figure()

fig.add_trace(
    go.Scatter(x=df_calls.strike.values, y=df_calls.impliedVolatility.values,
    name='Yahoo Finance'))

fig.add_trace(
    go.Scatter(x=df_calls.strike.values, y=implied_vols_calls,
    name='TensorFlow'))

fig.update_layout(title_text="Calls", title_x=0.5, xaxis_title="Strike Price", yaxis_title="Implied Volatility ",)
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))
col1.plotly_chart(fig, use_container_width=True)

fig = go.Figure()


fig.add_trace(
    go.Scatter(x=df_puts.strike.values, y=df_puts.impliedVolatility.values,
    name='Yahoo Finance'))

fig.add_trace(
    go.Scatter(x=df_puts.strike.values, y=implied_vols_puts,
    name='TensorFlow'))


fig.update_layout(title_text="Puts", title_x=0.5, xaxis_title="Strike Price", yaxis_title="Implied Volatility ",)
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))
col2.plotly_chart(fig, use_container_width=True)