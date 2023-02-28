# -*- coding: utf-8 -*-

# Statistics from return series
# BT
import pandas as pd

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

def unannualized_return(r):
    """
    Total return from a pd.Series of arithmetic returns
    """
    compounded_growth = (1+r).cumprod().iloc[-1]-1
    return compounded_growth
def annualized_return(r, periods_per_year=252):
    """
    Annualized return from a pd.Series of arithmetic returns
    """
    n_periods = r.shape[0]
    return (1+unannualized_return(r))**(periods_per_year/n_periods)-1
def annualized_vol(r, periods_per_year=252):
    """
    Annualized volatility of a pd.Series of arithmetic returns
    """
    n_periods = r.shape[0]
    return r.std()*(periods_per_year/n_periods)**0.5

def sharpe_ratio(r, rf=0, periods_per_year=252):
    """
    Annualized Sharpe ratio of a pd.Series of arithmetic returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+rf)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualized_return(excess_ret, periods_per_year)
    ann_vol = annualized_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol
def sortino_ratio(r, rf=0, periods_per_year=252):
    """
    Annualized Sortino ratio of a pd.Series of arithmetic returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+rf)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualized_return(excess_ret, periods_per_year)
    ann_vol = annualized_vol(r[r<0], # Only difference from Sharpe
                             periods_per_year) 
    return ann_ex_ret/ann_vol

def maximum_drawdown(r: pd.Series):
    """Takes a time series of asset returns.
       returns a DataFrame with columns for
       the wealth index, 
       the previous peaks, and 
       the percentage drawdown
    """
    wealth_index = (1+r).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks)/previous_peaks
    return min(0, drawdown.min())
def calmar_ratio(r, rf=0, periods_per_year=252):
    """
    Annualized Sortino ratio of a pd.Series of arithmetic returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+rf)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualized_return(excess_ret, periods_per_year)
    return ann_ex_ret/abs(maximum_drawdown(r)) # Only difference from Sharpe

def btstats(r: pd.Series):
    r = r.dropna()
    return pd.Series(
        {"Return (%)": f'{unannualized_return(r):.2%}', 
         "Ann. Return (%)": f'{annualized_return(r):.2%}', 
         "Ann. Volatility (%)": f'{annualized_vol(r):.2%}',
         "Max. Drawdown": f'{sortino_ratio(r):.3f}',
         "Sharpe Ratio": f'{sharpe_ratio(r):.3f}',
         "Sortino Ratio": f'{sortino_ratio(r):.3f}',
         "Calmar Ratio": f'{calmar_ratio(r):.3f}',
         "Skewness": f'{skewness(r):.3f}',
         }
        )
