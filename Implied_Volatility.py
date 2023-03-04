import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import tf_quant_finance as tff 
option_price = tff.black_scholes.option_price
implied_vol = tff.black_scholes.implied_vol
import yfinance as yf
from datetime import datetime

import streamlit as st

@st.cache_data
def get_ticker(symbol: str):
    return yf.Ticker(symbol)

symbol = st.selectbox('Select Stock', ('MSFT', 'APPL'))
ticker = get_ticker(symbol)

str_expiry_date = st.selectbox('Select Expiry Data', ticker.options)

df_calls, df_puts = ticker.option_chain(str_expiry_date)

for df in [df_calls, df_puts]:
    df['mid_price'] = (df['bid'] + df['ask']) / 2
df_calls = df_calls[['strike', 'mid_price', 'impliedVolatility']]
df_puts = df_puts[['strike', 'mid_price', 'impliedVolatility']]

dt_expiry_date = datetime.strptime(str_expiry_date, "%Y-%m-%d")
expiry = (dt_expiry_date - datetime.utcnow()).total_seconds() / 31536000





# Parameters
initial_volatilities = 1
rate = 0.05

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




import plotly.graph_objects as go

col1, col2 = st.columns(2)

fig = go.Figure()

fig.add_trace(
    go.Scatter(x=df_calls.strike.values, y=df_calls.impliedVolatility.values,
    name='Yahoo Finance'))

fig.add_trace(
    go.Scatter(x=df_calls.strike.values, y=implied_vols_calls,
    name='TensorFlow'))

fig.update_layout(title_text="Calls")
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


fig.update_layout(title_text="Puts")
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))
col2.plotly_chart(fig, use_container_width=True)