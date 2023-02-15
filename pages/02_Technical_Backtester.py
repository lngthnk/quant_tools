import streamlit as st

st.set_page_config(
    page_title="Technical Backtester"
)

import pandas as pd
import numpy as np
from stockstats import wrap

# import helper.binance_api as ba
import helper.yahoo_api as ya

df_of_symbols_set = pd.read_excel('https://github.com/phat-ap/quant_tools/blob/main/helper/set_symbol_list.xlsx?raw=true')
list_of_symbols_set = df_of_symbols_set['symbol'].to_list()

##########
@st.cache_data
def get_list_of_symbols(exchange: str):
    if exchange == "Binance":
        pass
        # return ba.list_of_symbols
    elif exchange == "SET":
        return list_of_symbols_set
    else:
        raise Exception("Bug")

@st.cache_data
def get_data_1d(exchange: str, symbol: str):
    if exchange == "Binance":
        pass
        # return ba.get_data_1d(symbol).astype(float)
    elif exchange == "SET":
        return ya.get_data_1d(symbol).astype(float)
    else:
        raise Exception("Bug")  
        
# Fix neg. Filter by date
@st.cache_data
def filter_df(df, int_start_year = None, int_end_year = None):
    if int_start_year is not None: df = df[df.index.year >= int_start_year]
    if int_end_year is not None: df = df[df.index.year <= int_end_year]
    df = wrap(df)
    return df[df > 0] # Quick fix for negative prices

@st.cache_data
def get_min_max_year(df):
    return df.index.year.min(), df.index.year.max()
##########

st.write('Technical Backtester')

sb_exchange \
    = st.selectbox("Exchange",
                   ("SET", )
                   )

if sb_exchange == "Binance":
    # list_of_symbols = ba.list_of_symbols
    pass 
elif sb_exchange == "SET":
    list_of_symbols = list_of_symbols_set
else:
    raise Exception("Bug")

sb_symbol \
    = st.selectbox(
        "Symbol",
        list_of_symbols
        )
    
df = get_data_1d(sb_exchange, sb_symbol)
min_year, max_year = get_min_max_year(df)

sb_start_year \
    = st.selectbox(
        "Start Year",
        list(range(min_year, max_year+1)),
        index=0
        )
sb_end_year \
    = st.selectbox(
        "End Year",
        list(range(int(sb_start_year), max_year+1)),
        index=int(max_year-int(sb_start_year))
        )

df = filter_df(df, int(sb_start_year), int(sb_end_year))
st.write(df)

# Data
#############
# Signals

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
st.write('*** Need to mute param_i_1 and indic_i_2 when rsi/macd is selected')

if 'n_indicators' not in st.session_state:
    st.session_state.n_indicators = 1

col1, col2, col3, col4, col5 = st.columns(5)
for i in range(st.session_state.n_indicators):
    with col1: 
        indic_1_i = st.selectbox(f'Indicator {i+1}, 1',('sma', 'ema', 'rsi', 'macds'), key = f'indic_1_{i}')
    with col2: 
        param_1_i = st.selectbox(f'Parameter {i+1}, 1', range(0, 101, 1), key = f'param_1_{i}')
    with col3: 
        sign_i = st.selectbox('Sign',('>=', '>', '==', '<', '<='), key = f'sign_{i}')
    with col4: 
        indic_2_i = st.selectbox(f'Indicator {i+1}, 2',('sma', 'ema', 'rsi', 'macds'), key = f'indic_2_{i}')
    with col5: 
        param_2_i = st.selectbox(f'Parameter {i+1}, 2', range(0, 101, 1), key = f'param_2_{i}')

# Add indicator
def add_indicator():
    st.session_state.n_indicators += 1
st.button('Add Indicator', on_click=add_indicator)

def remove_indicator():
    st.session_state.n_indicators -= 1
st.button('Remove Indicator', on_click=remove_indicator)
    
# sb_trade_other_direction
if sb_trade_direction == "Long":
    sb_trade_other_direction_label = 'Allow shorting otherwise?'
else:
    sb_trade_other_direction_label = 'Allow longing otherwise?'
sb_trade_other_direction \
    = st.checkbox(sb_trade_other_direction_label)
st.write('*** Haven\'t implemented!')

    
# Signals
#############
# Interpret
li_indic_1, li_param_1, li_sign, li_indic_2, li_param_2 \
    = map(list, zip(*[(st.session_state[f'indic_1_{i}'],
        st.session_state[f'param_1_{i}'],
        st.session_state[f'sign_{i}'],
        st.session_state[f'indic_2_{i}'],
        st.session_state[f'param_2_{i}']
        ) 
       for i in range(st.session_state.n_indicators)
       ]))
    

li_signals = []
for id_signal, (indic_1, param_1, sign, indic_2, param_2) \
    in enumerate(zip(li_indic_1, li_param_1, li_sign, li_indic_2, li_param_2)):
    if indic_1 in ['sma', 'ema']:
        signal_i = eval(f"df['close_{param_1}_{indic_1}']{sign}df['close_{param_2}_{indic_2}']")
        signal_i = signal_i.rename(f"signal_{id_signal}")
        li_signals.append(signal_i)
    elif indic_1 in ['rsi', 'macds']:
        signal_i = eval(f"df['{indic_1}']{sign}{param_2}")
        signal_i = signal_i.rename(f"signal_{id_signal}")
        li_signals.append(signal_i)

df_signal = pd.concat(li_signals, axis = 1)

# Combine all signal
condition = sb_and_or_or

signal = df_signal['signal_0']
for col in df_signal.columns:
    if condition == 'AND': signal = signal & df_signal[col]
    elif condition == 'OR': signal = signal | df_signal[col]

st.write(pd.concat([df_signal, signal.rename(condition)], axis = 1))

# Interpret
#############
# BT and Plot

df_returns = pd.concat([df['log-ret'].rename('bah'), 
                        (signal.shift(1).replace(False,0) * df['log-ret']).rename('strategy')
                       ],
                       axis=1)


import plotly.graph_objects as go
    
x = df_returns.cumsum().index

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x,
    y=df_returns.cumsum().bah,
    name = 'bah'
))
fig.add_trace(go.Scatter(
    x=x,
    y=df_returns.cumsum().strategy,
    name = 'strategy'
))

st.write('Perf compared to buy-and-hold')
st.write(fig)

# https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

import plotly.express as px

fig = px.imshow(correlation_from_covariance(df_signal.cov()),
                color_continuous_scale=px.colors.diverging.RdBu,
                color_continuous_midpoint=0,
                text_auto=True)

st.write('Corr between signals')
st.write(fig)

import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Histogram(x=df_returns.bah))
fig.add_trace(go.Histogram(x=(signal.shift(1).replace(False,np.nan) * df['log-ret'])))

# Overlay both histograms
fig.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)
st.write('Returns distribution compared to buy-and-hold')
st.write(fig)

# BT
def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3
def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1+r).cumsum() # edited
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1
def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    return r.std()*(periods_per_year**0.5)
def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def bt_statistics(r: pd.Series):
    return pd.DataFrame(
        {"Skewness": skewness(r), 
         "Annualize Vol": annualize_vol(r, 252), 
         "Sharpe Ratio": sharpe_ratio(r, 0, 252)}).iloc[-1]

st.write(pd.concat([bt_statistics(df_returns.bah).rename('bah'), bt_statistics(df_returns.strategy).rename('strategy')], axis = 1))