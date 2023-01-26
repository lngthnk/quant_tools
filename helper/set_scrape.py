# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 09:29:32 2023

@author: phata
"""

# SET

import pandas as pd
from string import ascii_uppercase
import yfinance as yf

def get_df_of_symbols_given_prefix(prefix: str = 'NUMBER'):
    df_th \
        = pd.read_html("https://classic.set.or.th/set/commonslookup.do?language=th&country=TH&prefix="
                       + prefix)[0]
    df_en \
        = pd.read_html("https://classic.set.or.th/set/commonslookup.do?language=en&country=US&prefix="
                       + prefix)[0]
    df \
        = df_en.join(df_th).drop(['ชื่อย่อหลักทรัพย์','ตลาด'], axis = 1)
    df \
        = df.rename(columns = {'Symbol': 'symbol',
                               'Company/Security Name': 'company_or_security_name_en',
                               'Market': 'market',
                               'ชื่อเต็มบริษัท / หลักทรัพย์จดทะเบียน': 'company_or_security_name_th'})
    df \
         = df[['symbol', 'market', 'company_or_security_name_en', 'company_or_security_name_th']]
    return df

def get_df_of_symbols():
    df = pd.concat([get_df_of_symbols_given_prefix(prefix) 
                    for prefix in ['NUMBER'] + list(ascii_uppercase)],
                   ignore_index = True)
    return df

df_of_symbols = get_df_of_symbols()
list_of_symbols = df_of_symbols['symbol'].to_list()

def get_symbol_obs(symbol):
    ticker = yf.Ticker(symbol + '.BK')
    return ticker.history(period="max")

if __name__ == '__main__':
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        df_of_symbols = get_df_of_symbols()
        print(df_of_symbols)