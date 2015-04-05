__author__ = 'jscastanoc'
import csv
import sys
import datetime as dt
import argparse

import pandas as pd
import pandas.tseries.offsets as pd_time
import numpy as np

import QSTK.qstkutil.qsdateutil as du # manipulation of time stamps
import QSTK.qstkutil.DataAccess as da # Quick access to norgate and yahoo data

def read_market(dt_start, dt_end, sym_file):
    dt_timeofday = dt.timedelta(hours=16)
    ldt_timestamps = du.getNYSEdays(dt_start,dt_end, dt_timeofday)
    c_dataobj = da.DataAccess('Yahoo')

    ls_symbols = c_dataobj.get_symbols_from_list(sym_file)
    ls_symbols.append('SPY')

    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ldf_data = c_dataobj.get_data(ldt_timestamps,ls_symbols,ls_keys)
    d_data = dict(zip(ls_keys,ldf_data))

    for c_key in ls_keys:
        #Forward propagation missing values
        d_data[c_key] = d_data[c_key].fillna(method='ffill')
        #Backward propagation missing values
        d_data[c_key] = d_data[c_key].fillna(method='bfill')
        #If it failed just fill with ones
        d_data[c_key] = d_data[c_key].fillna(1.0)
    return d_data, ldt_timestamps

def get_closePrices(ls_orders, data_source='Yahoo', mkt_benchmark='$SPX'):
    """
    From a given order list -as returned by read_orders- generate a pandas
    Dataframe with the closing prices of all the symbols contained in the
    order list from the earliest to the latest date listed in the order list.

    :param ls_orders: Order list as returned by read_orders()
    :param data_source: String indicating the source of the market data, as
        accepted by QSTK.qstkutil.DataAccess.DataAccess
    :param mkt_benchmark: String indicating the symbol used as market benchmark
    :return: Pandas dataframe containing the closing prices of the symbols
        ls_orders, ranging from the earliest to the latest listed date.
    """
    ldt_timestamps = get_dateInfo(ls_orders)
    c_dataobj = da.DataAccess(data_source)
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ls_symbols = ls_orders['SYMBOL'].values
    ls_symbols = list(ls_symbols)
    ls_symbols.append(mkt_benchmark)
    ls_symbols = list(set(ls_symbols))
    ldf_data = c_dataobj.get_data(ldt_timestamps,ls_symbols,ls_keys)
    market = dict(zip(ls_keys,ldf_data))
    market_close = pd.DataFrame(market['close'], index=ldt_timestamps)
    market_close = market_close.fillna(method='ffill')
    market_close = market_close.fillna(method='bfill')
    return market_close



def read_orders(orders_file):
    """
    Reads a cvs file where each line contains 'Buy' or 'Sell' order for
    a given data date.
    The format is yyyy,m,d,symbol,actions,qty
    Example:
    2011,2,2,XOM,Sell,4000,

    :param orders_file: name of the cvs file
    :return: pandas Dataframe with the fields DATE, SYMBOL, ACTION, QTY
    """
    ls_orders = pd.read_csv(orders_file,index_col=None,header=None,
                parse_dates={'DATE': [0,1,2]})
    ls_orders=ls_orders.drop([ls_orders.columns[4]],axis=1)
    ls_orders.columns=['DATE','SYMBOL','ACTION','QTY']
    return ls_orders

def get_dateInfo(ls_orders):
    """
    Get date vector for the dates of interest (earliest to latest).

    :param ls_orders: Order list as returned by read_orders()
    :return: consecutive list of dates from the earliest to the latest date
        contained in ls_orders
    """
    dt_start = pd.to_datetime(ls_orders['DATE'].values[0])
    dt_end = pd.to_datetime(ls_orders['DATE'].values[-1])
    dt_end += pd_time.DateOffset(days=1)
    dt_timeofday = dt.timedelta(hours=16)
    ldt_timestamps = du.getNYSEdays(dt_start,dt_end, dt_timeofday)
    return ldt_timestamps

def get_holdingMatrix(ls_orders, ldt_timestamps):
    """
    returns matrix with the amount of stocks for all the symbols contained
    in the order list, for the whole time interval of interest.

    :param ls_orders: Order list as returned by read_orders()
    :param ldt_timestamps: date vector as returned by get_dateInfo()
    :return:
    """
    ls_symbols = ls_orders['SYMBOL'].values
    ls_symbols = list(ls_symbols)
    d = {}
    for c_sym in ls_symbols:
        d[c_sym] = 0
    trade_matrix = pd.DataFrame(d, index=ldt_timestamps)
    hold_matrix = pd.DataFrame(d, index=ldt_timestamps)
    for c_day in enumerate(ls_orders['DATE']):
        c_dayEn = c_day[0]
        c_day = c_day[1] + pd_time.DateOffset(hours=16)
        c_symbol = ls_orders['SYMBOL'].ix[c_dayEn]
        if ls_orders['ACTION'].ix[c_dayEn].lower() == 'sell':
            sign = -1
        elif ls_orders['ACTION'].ix[c_dayEn].lower() == 'buy':
            sign = 1
        trade_matrix[c_symbol].loc[c_day] = trade_matrix[c_symbol].loc[c_day]+\
                                            sign*ls_orders['QTY'].ix[c_dayEn]
    for c_sym in trade_matrix:
        hold_matrix[c_sym] = np.cumsum(trade_matrix[c_sym])
    return hold_matrix

def get_priceMatrix(market, ls_orders):
    """
    gets stock prices for the symbols and dates of interest (those contained in
    in the order list ls_orders

    :param market: dataframe of market closing prices as returned by
        get_closePrices()
    :param ls_orders: Order list as returned by read_orders()
    :return:
    """
    ls_symbols = ls_orders['SYMBOL'].values
    ls_symbols = set(list(ls_symbols))
    ldt_timestamps = get_dateInfo(ls_orders)
    sym_toDrop = set(list(market.columns)) - ls_symbols
    price_matrix = market.drop(sym_toDrop,axis=1)
    price_matrix = price_matrix[list(ls_symbols)]
    return price_matrix

def get_cashFlow(cash_init, market, ls_orders):
    """
    Tracks the amount of cash available according to market prices, and order
    list. Negative cash is allowed

    :param cash_init: Cash available at day 0
    :param market: dataframe of market closing prices as returned by
        get_closePrices()
    :param ls_orders: Order list as returned by read_orders()
    :return: pandas timeseries with the cash for every day.
    """
    price_matrix = get_priceMatrix(market, ls_orders)
    ldt_timestamps = get_dateInfo(ls_orders)
    hold_matrix = get_holdingMatrix(ls_orders,ldt_timestamps)
    d = {}
    for c_sym in hold_matrix.columns:
        d[c_sym] = 0
    trade_matrix = pd.DataFrame(d, index=ldt_timestamps)
    for c_sym in hold_matrix.columns:
        tmp = np.insert(hold_matrix[c_sym].values,0,0)
        tmp = np.diff(tmp)
        hold_matrix[c_sym] = tmp
    cash_matrix = pd.Series(0,index=ldt_timestamps)
    cash_matrix.loc[ldt_timestamps[0]] = cash_init
    for c_day in enumerate(ls_orders['DATE']):
        c_dayEn = c_day[0]
        c_day = c_day[1] + pd_time.DateOffset(hours=16)
        c_symbol = ls_orders['SYMBOL'].ix[c_dayEn]
        if ls_orders['ACTION'].ix[c_dayEn].lower() == 'sell':
            sign = -1
        elif ls_orders['ACTION'].ix[c_dayEn].lower() == 'buy':
            sign = 1
        trade_matrix[c_symbol].loc[c_day] = trade_matrix[c_symbol].loc[c_day]+\
                                            sign*ls_orders['QTY'].ix[c_dayEn]
        price = -1*price_matrix[c_symbol].loc[c_day]*sign*\
                ls_orders['QTY'].ix[c_dayEn]
        cash_matrix.loc[c_day] =  cash_matrix.loc[c_day]+price
    cash_matrix = np.cumsum(cash_matrix)
    return cash_matrix

def get_portfolioVal(ls_orders,cash_init,market):
    """
    Calculates the daily value of the portfolio for between the earliest and
    latest date contained in ls_orders

    :param ls_orders: Order list as returned by read_orders()
    :param cash_init: float. Initial amount of money
    :param market: dataframe of market closing prices as returned by
        get_closePrices()
    :return: pandas time series with the daily value of the portfolio result of
        executing the orders listed in ls_orders
    """
    ldt_timestamps = get_dateInfo(ls_orders)
    hold_matrix = get_holdingMatrix(ls_orders, ldt_timestamps)
    price_matrix = get_priceMatrix(market,ls_orders)
    cash_matrix = get_cashFlow(cash_init,market,ls_orders)
    portfolio_val = hold_matrix*price_matrix
    portfolio_val = np.sum(portfolio_val,axis=1)+cash_matrix
    return portfolio_val

def write_valuesCSV(portfolio_values, name='values.csv'):
    """
    write the porfolio daily value in a csv file

    :param portfolio_value: daily portfolio value as returned by
    get_portfolioVal
    :param name: name of the saved file
    :return: 0
    """
    writer = csv.writer(open(name, 'wb'), delimiter=',')
    for row_index in portfolio_val.index:
        row_to_enter = [str(row_index.year), str(row_index.month), \
                        str(row_index.day), str(portfolio_val[row_index])]
        writer.writerow(row_to_enter)
    return 0

if __name__ == '__main__':
    """
    Simulates a portfolio using an order list generated by EventProfiler

    :param ci: Starting cash
                (default 100000)
    :param ol: Order list file generated by EventProfiler
                (default orders.csv)
    :param of: Output file to store Portfolio history
                (default values.csv)

    Call example:
        python MarketSim.py -ci 100000 -ol orders.csv -of values.csv
    """

    parser = argparse.ArgumentParser(description=""""Simulates a portfolio
                            using an order list generated by EventProfiler""")
    parser.add_argument('-ci', default=100000, type=float)
    parser.add_argument('-ol', default='orders.csv', type=str)
    parser.add_argument('-of', default='values.csv', type=str)

    args = parser.parse_args()

    cash_init = args.ci
    orders = args.ol
    out_name = args.of
    ls_orders = read_orders(orders)
    market_close = get_closePrices(ls_orders)
    portfolio_val = get_portfolioVal(ls_orders,cash_init,market_close)
    write_valuesCSV(portfolio_val,name=out_name)

