import streamlit as st

st.set_page_config(
    page_title="Technical Backtester"
)

import pandas as pd
import numpy as np
from stockstats import wrap

import helper.yahoo_api as ya

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
                    'n_params': 2,
                    'default': 
                       {'short': 12,
                        'long': 26}
                   },
           'MACD signal': {'var_name': 'macds',
                    'n_params': 3,
                    'default': 
                       {'short': 12,
                        'long': 26,
                        'signal': 9}
                   },
               
           }
    return dict_indicators
dict_indicators = func_dict_indicators()



st.write('Technical Backtester')

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
    
    dict_rules[i_rule][f'indic_a'] \
        = cols[0].selectbox(f'Indicator a',
                            dict_indicators.keys(), 
                            key = f'indic_{i_rule}a')
    
    if st.session_state[f'n_param_{i_rule}_a'] > 0:
        for i in range(st.session_state[f'n_param_{i_rule}_a']):
            label = list(dict_indicators[st.session_state[f'indic_{i_rule}a']]['default'].keys())[i]
            default = dict_indicators[st.session_state[f'indic_{i_rule}a']]['default'][label]
            dict_rules[i_rule][f'param_a_{label}'] \
                = cols[i+1].number_input(label, value = default, format = '%d', key = f'param_{i_rule}a_{label}')
            
    dict_rules[i_rule][f'sign'] \
        = cols[1+st.session_state[f'n_param_{i_rule}_a']].selectbox(f'Sign',
                                                              ('>=', '>', '==', '<', '<='), 
                                                              key = f'sign_{i_rule}')
    
    dict_rules[i_rule][f'indic_b'] \
        = cols[2+st.session_state[f'n_param_{i_rule}_a']].selectbox(f'Indicator {i_rule}b',
                      dict_indicators.keys(), 
                      key = f'indic_{i_rule}b')
    
    if st.session_state[f'n_param_{i_rule}_b'] > 0:
        for i in range(st.session_state[f'n_param_{i_rule}_b']):
            label = list(dict_indicators[st.session_state[f'indic_{i_rule}b']]['default'].keys())[i]
            default = dict_indicators[st.session_state[f'indic_{i_rule}b']]['default'][label]
            dict_rules[i_rule][f'param_b_{label}'] \
                = cols[i+3+st.session_state[f'n_param_{i_rule}_a']].number_input(label, value = default, format = '%d', key = f'param_{i_rule}b_{label}')
    
    st.write(dict_rules[i_rule])

'''
### Read rules
'''
StockDataFrame = df
def run_rule_indic_1(dict_rules, i_rule, ab):
    if dict_rules[i_rule][f'indic_{ab}'] == 'RSI':
        return StockDataFrame['rsi_' + str(dict_rules[i_rule][f'param_{ab}_window'])]
    if dict_rules[i_rule][f'indic_{ab}'] == 'MACD':
        StockDataFrame.MACD_EMA_SHORT = dict_rules[i_rule][f'param_{ab}_short']
        StockDataFrame.MACD_EMA_LONG = dict_rules[i_rule][f'param_{ab}_long']
        return StockDataFrame['macd']
    if dict_rules[i_rule][f'indic_{ab}'] == 'MACD signal':
        StockDataFrame.MACD_EMA_SHORT = dict_rules[i_rule][f'param_{ab}_short']
        StockDataFrame.MACD_EMA_LONG = dict_rules[i_rule][f'param_{ab}_long']
        StockDataFrame.MACD_EMA_SIGNAL = dict_rules[i_rule][f'param_{ab}_signal']
        return StockDataFrame['macds']
    else:
        pass
# งงงงงงง
st.write(df.MACD_EMA_SHORT, df.MACD_EMA_LONG, df.MACD_EMA_SIGNAL)
siga = run_rule_indic_1(dict_rules, 0, 'a')
st.write(df)
st.write(df.MACD_EMA_SHORT, df.MACD_EMA_LONG, df.MACD_EMA_SIGNAL)
sigb = run_rule_indic_1(dict_rules, 0, 'b')
st.write(sigb)
st.write(df.MACD_EMA_SHORT, df.MACD_EMA_LONG, df.MACD_EMA_SIGNAL)
