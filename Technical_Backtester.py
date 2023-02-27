import streamlit as st

st.set_page_config(
    page_title="Technical Backtester"
)

import pandas as pd
import numpy as np
from stockstats import wrap

import helper.yahoo_api as ya

'''
# Technical Backtester
'''

@st.cache_data
def func():
    df_of_symbols_set = pd.read_excel('https://github.com/phat-ap/quant_tools/blob/main/helper/set_symbol_list.xlsx?raw=true')
    list_of_symbols_set = df_of_symbols_set['symbol'].to_list()
    return df_of_symbols_set, list_of_symbols_set
df_of_symbols_set, list_of_symbols_set = func()

@st.cache_data
def func_dict_indicators():
    dict_indicators \
        = {'Last close': {'var_name': 'close',
                          'n_params': 0
                          },
           'Value': {'n_params': 1,
                     'default': 
                         {'value': 0}
                     },
           'SMA': {'var_name': 'sma',
                   'n_params': 1,
                   'default': 
                       {'window': 10}
                   },
           'EMA': {'var_name': 'ema',
                   'n_params': 1,
                   'default': 
                       {'window': 10}
                   },
           'RSI': {'var_name': 'rsi',
                   'n_params': 1,
                   'default': 
                       {'window': 14}
                   },
           'MACD': {'var_name': 'macd',
                    'n_params': 0,  # turned off
                    'default': 
                       {'short': 12,
                        'long': 26}
                   },
           'MACD signal': {'var_name': 'macds',
                    'n_params': 0,  # turned off
                    'default': 
                       {'short': 12,
                        'long': 26,
                        'signal': 9}
                   },
               
           }
    return dict_indicators
dict_indicators = func_dict_indicators()

'''
### Choose stock
'''
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

cols = st.columns(4)

sb_exchange \
    = cols[0].selectbox("Exchange",
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
    = cols[1].selectbox(
        "Symbol",
        list_of_symbols_set
        )
    
df = get_data_1d(sb_exchange, sb_symbol)
min_year, max_year = get_min_max_year(df)

sb_start_year \
    = cols[2].selectbox(
        "Start Year",
        list(range(min_year, max_year+1)),
        index=0
        )
sb_end_year \
    = cols[3].selectbox(
        "End Year",
        list(range(int(sb_start_year), max_year+1)),
        index=int(max_year-int(sb_start_year))
        )

df = filter_df(df, int(sb_start_year), int(sb_end_year))

'''
### Choose indicators
'''
sb_and_or_or \
    = st.selectbox("AND or OR",
                   ("AND", "OR")
                   )

if 'n_rules' not in st.session_state:
    st.session_state['n_rules'] = 1
# Add rule
def add_rule():
    st.session_state['n_rules'] += 1
st.button('Add a rule', on_click=add_rule)
# Remove rule
def remove_rule():
    st.session_state['n_rules'] -= 1
disabled_remove = True if st.session_state['n_rules'] == 1 else False
st.button('Remove last rule', on_click=remove_rule, disabled = disabled_remove)

# List of rules
dict_rules = {}    

for i_rule in range(st.session_state['n_rules']):
    st.write(f'Rule {i_rule+1}')
    if f'indic_{i_rule}a' not in st.session_state:
        st.session_state[f'indic_{i_rule}a'] = 'Last close'
    if f'indic_{i_rule}b' not in st.session_state:
        st.session_state[f'indic_{i_rule}b'] = 'Last close'
    st.session_state[f'n_param_{i_rule}_a'] \
        = dict_indicators[st.session_state[f'indic_{i_rule}a']]['n_params']
    st.session_state[f'n_param_{i_rule}_b'] \
        = dict_indicators[st.session_state[f'indic_{i_rule}b']]['n_params']
    
    cols = st.columns(3 + st.session_state[f'n_param_{i_rule}_a'] + st.session_state[f'n_param_{i_rule}_b'])
    
    dict_rules[i_rule] \
        = {}    
    
    dict_rules[i_rule]['indic_a'] \
        = cols[0].selectbox('Indicator a',
                            dict_indicators.keys(), 
                            key = f'indic_{i_rule}a')
    
    if st.session_state[f'n_param_{i_rule}_a'] > 0:
        for i in range(st.session_state[f'n_param_{i_rule}_a']):
            label = list(dict_indicators[st.session_state[f'indic_{i_rule}a']]['default'].keys())[i]
            default = dict_indicators[st.session_state[f'indic_{i_rule}a']]['default'][label]
            dict_rules[i_rule][f'param_a_{label}'] \
                = cols[i+1].number_input(label, value = default, format = '%d', key = f'param_{i_rule}a_{label}')
            
    dict_rules[i_rule]['sign'] \
        = cols[1+st.session_state[f'n_param_{i_rule}_a']].selectbox('Sign',
                                                              ('>=', '>', '==', '<', '<='), 
                                                              key = f'sign_{i_rule}')
    
    dict_rules[i_rule]['indic_b'] \
        = cols[2+st.session_state[f'n_param_{i_rule}_a']].selectbox('Indicator b',
                      dict_indicators.keys(), 
                      key = f'indic_{i_rule}b')
    
    if st.session_state[f'n_param_{i_rule}_b'] > 0:
        for i in range(st.session_state[f'n_param_{i_rule}_b']):
            label = list(dict_indicators[st.session_state[f'indic_{i_rule}b']]['default'].keys())[i]
            default = dict_indicators[st.session_state[f'indic_{i_rule}b']]['default'][label]
            dict_rules[i_rule][f'param_b_{label}'] \
                = cols[i+3+st.session_state[f'n_param_{i_rule}_a']].number_input(label, value = default, format = '%d', key = f'param_{i_rule}b_{label}')
    
    # st.write(dict_rules[i_rule])

# Read rules
def run_rule_indic(StockDataFrame, dict_rules, i_rule, ab):
    if dict_rules[i_rule][f'indic_{ab}'] == 'Last close':
        return StockDataFrame['close'].rename(f'{i_rule}{ab}')
    if dict_rules[i_rule][f'indic_{ab}'] == 'Value':
        return dict_rules[i_rule][f'param_{ab}_value']
    if dict_rules[i_rule][f'indic_{ab}'] in ['SMA', 'EMA']:
        str_var_name = dict_indicators[dict_rules[i_rule][f'indic_{ab}']]['var_name']
        str_window = str(dict_rules[i_rule][f'param_{ab}_window'])
        return StockDataFrame[f'close_{str_window}_{str_var_name}'].rename(f'{i_rule}{ab}')
    if dict_rules[i_rule][f'indic_{ab}'] == 'RSI':
        return StockDataFrame['rsi_' + str(dict_rules[i_rule][f'param_{ab}_window'])].rename(f'{i_rule}{ab}')
    if dict_rules[i_rule][f'indic_{ab}'] in ['MACD', 'MACD signal']:
        return StockDataFrame[dict_indicators[dict_rules[i_rule][f'indic_{ab}']]['var_name']].rename(f'{i_rule}{ab}')
    else:
        pass
    
def run_rule(StockDataFrame, dict_rules, i_rule):
    if dict_rules[i_rule]['sign'] == '>=':
        signal = run_rule_indic(StockDataFrame, dict_rules, i_rule, 'a') >= run_rule_indic(StockDataFrame, dict_rules, i_rule, 'b')
    if dict_rules[i_rule]['sign'] == '>':
        signal = run_rule_indic(StockDataFrame, dict_rules, i_rule, 'a') > run_rule_indic(StockDataFrame, dict_rules, i_rule, 'b')
    if dict_rules[i_rule]['sign'] == '==':
        signal = run_rule_indic(StockDataFrame, dict_rules, i_rule, 'a') == run_rule_indic(StockDataFrame, dict_rules, i_rule, 'b')
    if dict_rules[i_rule]['sign'] == '<':
        signal = run_rule_indic(StockDataFrame, dict_rules, i_rule, 'a') < run_rule_indic(StockDataFrame, dict_rules, i_rule, 'b')
    if dict_rules[i_rule]['sign'] == '<=':
        signal = run_rule_indic(StockDataFrame, dict_rules, i_rule, 'a') <= run_rule_indic(StockDataFrame, dict_rules, i_rule, 'b')
    else:
        pass
    return signal.rename(f'{i_rule}')
df_signals = pd.concat([run_rule(df, dict_rules, i_rule) 
                        for i_rule 
                        in range(st.session_state['n_rules'])
                        ],
                       axis = 1
                       )
# st.write(df_signals)

# Combine rules

signal = df_signals['0']
for col in df_signals.columns:
    if sb_and_or_or == 'AND': signal = signal & df_signals[col]
    elif sb_and_or_or == 'OR': signal = signal | df_signals[col]
# st.write(pd.concat([df_signals, signal.rename(sb_and_or_or)], axis = 1))

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

fig = px.imshow(correlation_from_covariance(df_signals.cov()),
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