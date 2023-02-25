import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import riskfolio as rp

import helper.yahoo_api as ya

st.set_page_config(
        page_title="Efficient Frontier on SET data",
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
    if i > 0 and st.session_state[f'stock_{i-1}'] == '':
        disabled = True
    cols[i%5].selectbox(f'Stock {i+1}', 
                        [''] + list_of_symbols_set, 
                        key = f'stock_{i}', 
                        disabled = disabled
                        )
cols = st.columns(5)
for i in range(5, 10):
    if i > 0 and st.session_state[f'stock_{i-1}'] == '':
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

if len(li_stocks) >= 2:
    '''
    ### Start and end date
    '''
    df = yf.download([str + '.BK' for str in li_stocks])['Close']
    df.columns = df.columns.str.rstrip('.BK')
    df_returns = df.pct_change().dropna()
    
   
    st.write('The data ranges from ',
             df_returns.index.min().strftime("%d %B, %Y"), 
             ', to ',
             df_returns.index.max().strftime("%d %B, %Y"),
             '.')
    cols = st.columns(2)
    cols[0].date_input('Start date',
                       value = df_returns.index.min(),
                       min_value=df_returns.index.min(),
                       max_value=df_returns.index.max(),
                       key = 'start_date')
    cols[1].date_input('End date',
                       value = df_returns.index.max(),
                       min_value=st.session_state.start_date,
                       max_value=df_returns.index.max(),
                       key = 'end_date')
    
    '''
    ### Optimal portfolio
    '''
    
    
    # Building the portfolio object
    port = rp.Portfolio(returns=df_returns)
    # Select method and estimate input parameters:
    method_mu = 'hist'
    method_cov = 'hist'
    port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)
    # Estimate optimal portfolio:
    model = 'Classic'
    rm = 'MV'
    obj = 'Sharpe'
    hist = True
    rf = 0
    l = 0 # Risk aversion factor, only useful when obj is 'Utility'
    w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist
                          )
    
    assert w is not None, 'Cannot construct optimal portfolio from the given stocks. Try changing or adding more stocks.'    

    
    fig, ax = plt.subplots()
    rp.plot_pie(w = w, 
                title = 'Sharpe Mean Variance', 
                others = 0.05, 
                nrow = 25, 
                cmap = "tab20",
                height = 6, 
                width = 10, 
                ax = ax
                )
    st.pyplot(fig)
    
    '''
    ### Efficient frontier
    '''
    points = 50 # Number of points of the frontier
    
    frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)
    
    
    label = 'Max Risk Adjusted Return Portfolio' # Title of point
    mu = port.mu # Expected returns
    cov = port.cov # Covariance matrix
    returns = port.returns # Returns of the assets
    
    fig, ax = plt.subplots()
    rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                     rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                     marker='*', s=16, c='r', height=6, width=10, ax=None)
    st.pyplot(fig)
    
    '''
    ### Efficient frontier composition
    '''
    fig, ax = plt.subplots()
    rp.plot_frontier_area(w_frontier=frontier, cmap="tab20", height=6, width=10, ax=None)
    st.pyplot(fig)
    
    '''
    ### Risk Contributions
    '''
    dict_rm = {'Standard Deviation': 'MV',
               'Square Root Kurtosis': 'KT',
               'Mean Absolute Deviation': 'MAD',
               'Gini Mean Difference': 'GMD',
               'Semi Standard Deviation': 'MSV',
               'Square Root Semi Kurtosis': 'SKT',
               'First Lower Partial Moment (Omega Ratio)': 'FLPM',
               'Second Lower Partial Moment (Sortino Ratio)': 'SLPM',
               'Conditional Value at Risk': 'CVaR',
               'Tail Gini': 'TG',
               'Entropic Value at Risk': 'EVaR',
               'Worst Realization (Minimax)': 'WR',
               'CVaR range of returns': 'CVRG',
               'Tail Gini range of returns': 'TGRG',
               'Range of returns': 'RG',
               'Maximum Drawdown of uncompounded cumulative returns (Calmar Ratio)': 'MDD',
               'Average Drawdown of uncompounded cumulative returns': 'ADD',
               'Drawdown at Risk of uncompounded cumulative returns': 'DaR',
               'Conditional Drawdown at Risk of uncompounded cumulative returns': 'CDaR',
               'Ulcer Index of uncompounded cumulative returns': 'UCI'
               }
    
    rm = st.selectbox('Select Risk Measure',
                      dict_rm.keys()
                      )
    
    fig, ax = plt.subplots()
    
    rp.plot_risk_con(w=w, cov=cov, returns=returns, rm=dict_rm[rm],
                     rf=0, alpha=0.05, color="tab:blue", height=6,
                     width=10, t_factor=252, ax=ax
                     )
    
    st.pyplot(fig)
