import csv

import pandas as pd
import yfinance as yf
import numpy as np
import pandas_ta as ta
import talib as tal
from statistics import covariance
from datetime import datetime as dt
import quantstats as qs
pd.options.mode.chained_assignment = None  # default='warn'


desired_width = 320

pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)
# pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 10)
rf = 0.0451 # risk free rate for three month T-Bill -> https://ycharts.com/indicators/1_year_treasury_rate, https://www.investopedia.com/terms/s/sharperatio.asp


def add_Backtesting_Ratios(df,start,end):

    results = pd.DataFrame()
    temp2 = pd.DataFrame()

    benchmark = df.loc[(df["Date"] >= start) & (df["Date"] <= end) & (df["Ticker"] == "SPY")]
    benchmark_daily_returns = benchmark["Close"].pct_change().rename("bench_daily_ret").reset_index(drop=True)

    df = df.loc[(df["Date"] >= start) & (df["Date"] <= end)]
    for ticker in df["Ticker"].unique():

        temp = df[df["Ticker"] == ticker].reset_index(drop=True)
        t = temp["Close"].pct_change().rename("ticker_daily_ret").reset_index(drop=True)

        # Calculate the beta for the stock
        covar = pd.concat([t, benchmark_daily_returns], axis=1).cov()
        beta = covar.iloc[0,1] / covar.iloc[1,1]

        daily_std = t.std()
        annual_std = daily_std * np.sqrt(252)

        deannualized_rf = np.power(1 + rf, 1. / 252) - 1.
        nominator = t.mean() - deannualized_rf

        # Sharpe Ratio https://corporatefinanceinstitute.com/resources/wealth-management/sortino-ratio-2/
        temp2["Sharpe Ratio"] = [(nominator / daily_std) * np.sqrt(252)]

        # Treynor Ratio
        temp2["Treynor Ratio"] = [nominator / beta * 252]


        # Sortino Ratio
        downside_std = np.sqrt((t[t < 0] ** 2).sum() / len(t))
        temp2["Sortino Ratio"] = [nominator / downside_std * np.sqrt(252)]

        # Annulized returns
        temp2["Total Return"] = [((temp["Close"].iloc[-1] - temp["Close"].iloc[0]) / temp["Close"].iloc[0])]

        # Annulized Volatility
        temp2["Annulized Volatility"] = [annual_std]

        results = pd.concat([results, pd.DataFrame(
            {"Ticker": ticker, "Sharpe Ratio": temp2["Sharpe Ratio"], "Total Return": temp2["Total Return"],
             "Sortino Ratio": temp2["Sortino Ratio"], "Annulized Volatility": temp2["Annulized Volatility"], "Treynor Ratio": temp2["Treynor Ratio"]})],
                            ignore_index=True)

    return results

def add_percentage_gain(df):
    results = pd.DataFrame()
    for ticker in df["Ticker"].unique():
        temp = df[df["Ticker"] == ticker]
        temp["Pct change"] = temp["Adj Close"].pct_change()
        temp["cum Pct change"] = ((1 + temp["Pct change"]).cumprod()-1) * 100

        results = pd.concat([results, pd.DataFrame(
            {"Ticker": ticker, "Date": temp["Date"], "Pct change": temp["Pct change"],
             "cum Pct change": temp["cum Pct change"]})], ignore_index=True)

    return results

def add_indicators(df, period_k = None, period_d = None, rsiLength = None, WillrLength = None, MACDSlow = None,MACDFast = None, MACDSig = None):
    results = pd.DataFrame()

    df.ta.adjusted = "Close"
    for ticker in df["Ticker"].unique():
        ticker_df = df[df["Ticker"] == ticker]

        # Stochastic Oscilator
        if (period_d != None) and (period_k != None):
            ticker_df.ta.stoch(high='High', low='Low', k=period_k, d=period_d, append=True)

            results = pd.concat([results, pd.DataFrame(
            {"Ticker": ticker, "Date": ticker_df["Date"],
             f"STOCHk_{period_k}_{period_d}": ticker_df[f"STOCHk_{period_k}_{period_d}_3"],
             f"STOCHd_{period_k}_{period_d}": ticker_df[f"STOCHd_{period_k}_{period_d}_3"]})])


        # RSI
        if rsiLength != None:
            ticker_df.ta.rsi(close='Close', length=rsiLength, append=True, signal_indicators=True, xa=60, xb=40)

            results = pd.concat([results, pd.DataFrame(
                {"Ticker": ticker, "Date": ticker_df["Date"],f"RSI{rsiLength}": ticker_df[f"RSI_{rsiLength}"]})], ignore_index=True)

        # MACD
        if (MACDFast != None) and (MACDSig != None) and (MACDSlow != None):
            ticker_df.ta.macd(close='close', fast=MACDFast, slow=MACDSlow, signal=MACDSig, append=True)

            results = pd.concat([results, pd.DataFrame(
                {"Ticker": ticker, "Date": ticker_df["Date"],f"MACD_{MACDFast}_{MACDSlow}_{MACDSig}":ticker_df[f"MACD_{MACDFast}_{MACDSlow}_{MACDSig}"],
             f"MACDh_{MACDFast}_{MACDSlow}_{MACDSig}":ticker_df[f"MACDh_{MACDFast}_{MACDSlow}_{MACDSig}"],
             f"MACDs_{MACDFast}_{MACDSlow}_{MACDSig}":ticker_df[f"MACDs_{MACDFast}_{MACDSlow}_{MACDSig}"]}  )], ignore_index=True)

        # WILLR
        if WillrLength != None:
            ticker_df[f"WILLR{WillrLength}"] = tal.WILLR(ticker_df["High"], ticker_df["Low"], ticker_df["Close"], WillrLength)

            results = pd.concat([results, pd.DataFrame(
                {"Ticker": ticker, "Date": ticker_df["Date"],f"WILLR{WillrLength}":ticker_df[f"WILLR{WillrLength}"]})])



    print(results)
    return results

def read_sentiment_values(ticker):
    ticker = str.lower(ticker)
    df = pd.read_csv(f"data\\daily-average-{ticker}.csv")
    df = df.rename(columns={"created_at" : "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    return df




# download stock tickers from yfinance
stockList = ["MSFT", "META", "TTE", "AMZN", "XOM", "TSLA", "SPY", "AAPL", "GOOG", "MCD", "CVX"]
data = yf.download(stockList, start="2015-01-01", end="2023-01-11", group_by="Ticker")

# transform index level -> Tickers as column instead of level 0
df = data.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1).reset_index()
df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()



