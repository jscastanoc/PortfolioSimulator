__author__ = 'jscastanoc'
import datetime as dt
import copy

import QSTK.qstkutil.qsdateutil as du # manipulation of time stamps
import QSTK.qstkutil.tsutil as tsu # Computes metrics for given time series
import QSTK.qstkutil.DataAccess as da # Quick access to norgate and yahoo data
import QSTK.qstkstudy.EventProfiler as ep

import matplotlib.pyplot as plt
import pandas as pd
import pandas.stats.moments as pd_stats
import numpy as np


def bollinger_bands(d_data, ldt_timestamps, ls_symbols=None, lookback = 20,
                    width = 1, plot_boll=False, ls_symbols_plot=None):
    if ls_symbols == None:
        ls_symbols = list(d_data.keys())
    df_close = copy.deepcopy(d_data)
    df_close = df_close[ls_symbols]

    df_mean_bollinger = copy.deepcopy(df_close) * np.NAN
    df_std_bollinger = copy.deepcopy(df_close) *np.NAN
    df_index_bollinger = copy.deepcopy(df_close) *np.NAN

    for c_sym in ls_symbols:
        df_mean_bollinger[c_sym] = pd_stats.rolling_mean(df_close[c_sym],lookback)
        df_std_bollinger[c_sym] = width*pd_stats.rolling_std(df_close[c_sym],lookback)
        df_index_bollinger[c_sym] = (df_close[c_sym] -
                                     df_mean_bollinger[c_sym])/df_std_bollinger[c_sym]

    if plot_boll:
        if ls_symbols_plot == None:
            if len(ls_symbols) <= 5:
                ls_symbols_plot = ls_symbols
            else:
                ls_symbols_plot = ls_symbols[0:5]
        fig = []
        for c_sym in ls_symbols_plot:
            fig.append(plt.figure())
            ax = fig[-1].add_subplot(211)
            ax.plot(ldt_timestamps,df_close[c_sym],'k')
            ax.plot(ldt_timestamps,df_mean_bollinger[c_sym],'b')
            ax.plot(ldt_timestamps,df_mean_bollinger[c_sym] - df_std_bollinger[c_sym],'b--')
            ax.plot(ldt_timestamps,df_mean_bollinger[c_sym] + df_std_bollinger[c_sym],'b--')
            ax.set_xlim((ldt_timestamps[0],ldt_timestamps[-1]))
            ax.get_xaxis().set_visible(False)
            ax.set_ylabel('Adj. Close')

            ax = fig[-1].add_subplot(212)
            ax.plot(ldt_timestamps, df_index_bollinger)
            ax.plot([ldt_timestamps[0], ldt_timestamps[-1]],
                    [1, 1])
            ax.plot([ldt_timestamps[0], ldt_timestamps[-1]],
                    [-1, -1])
            ax.set_xlim((ldt_timestamps[0],ldt_timestamps[-1]))
            ax.set_xlabel('Time')
            ax.set_ylabel('Bollinger Val.')
        plt.show()

    return df_index_bollinger


if __name__ == '__main__':
    dt_start = dt.datetime(2010,1,1)
    dt_end = dt.datetime(2010,12,31)
    symbol_file = "sp5002012"
    d_data, ldt_timestamps = MarketSim.read_market(dt_start,
                                                       dt_end, symbol_file)

    df_mean_boll, df_std_boll, df_index = bollinger_bands(d_data,
                                                ldt_timestamps, ['VZ'],
                                                plot_boll=True)



