import streamlit as st

import pandas as pd

import helper.binance_api as ba
import helper.yahoo_api as ya

st.set_page_config(
    page_title="QuantCorner - Data Loader"
)

##########

df_of_symbols_set = pd.read_excel('C:/Users/phata/Documents/GitHub/quant_tools/helper/set_symbol_list.xlsx')
list_of_symbols_set = df_of_symbols_set['symbol'].to_list()

@st.experimental_memo
def get_list_of_symbols(exchange: str):
    if exchange == "Binance":
        return ba.list_of_symbols
    elif exchange == "SET":
        return list_of_symbols_set
    else:
        raise Exception("Bug")

@st.experimental_memo
def get_data_1d(exchange: str, symbol: str, str_start_date = None, str_end_date = None):
    if exchange == "Binance":
        return ba.get_data_1d(symbol)
    elif exchange == "SET":
        return ya.get_data_1d(symbol)
    else:
        raise Exception("Bug")  
        
##########

st.markdown('**Data Loader**')

col1, col2 \
    = st.columns(2)
sb_exchange \
    = col1.selectbox("Exchange",
                     ("Binance", "SET")
                     )
if sb_exchange == "Binance":
    list_of_symbols = ba.list_of_symbols
elif sb_exchange == "SET":
    list_of_symbols = list_of_symbols_set
else:
    raise Exception("Bug")
sb_symbol \
    = col2.selectbox("Symbol",
                     list_of_symbols
                     )   

# Get df
df = get_data_1d(sb_exchange, sb_symbol)

st.write(df)