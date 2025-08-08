import datetime as dt
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def dataloader_YF(
    PATH, start_date=dt.datetime(2004, 1, 1), end_date=dt.datetime(2022, 5, 20)
):
    files = os.listdir(PATH)
    dir_files = {}
    for file in files:
        ticker = file[:-4]
        df = pd.read_csv(os.path.join(PATH, file))
        # remove the beginning constant samples
        indices = (pd.to_datetime(df["Date"]) > start_date) & (
            pd.to_datetime(df["Date"]) < end_date
        )
        df = df[indices]
        # export to np arrays
        date_arr = pd.to_datetime(df["Date"]).to_numpy()
        open_arr = df["Open"].to_numpy()
        close_arr = df["Close"].to_numpy()
        high_arr = df["High"].to_numpy()
        low_arr = df["Low"].to_numpy()
        volume_arr = df["Volume"].to_numpy()
        dir_files[ticker] = [
            date_arr,
            open_arr,
            close_arr,
            high_arr,
            low_arr,
            volume_arr,
        ]
    return dir_files


def dataloader_SP500(
    year, PATH_org=os.path.join(".", "data", "YahooFinance", "SP500"), verbose=True
):
    # PATH = "./data/YahooFinance/SP500/"+str(year)
    PATH = os.path.join(PATH_org, str(year))
    files = os.listdir(PATH)
    dir_files = {}
    for file in files:
        try:
            ticker = file[:-18]
            df = pd.read_csv(os.path.join(PATH, file))
            # export to np arrays
            date_arr = pd.to_datetime(df["Date"]).to_numpy()
            open_arr = df["Open"].to_numpy()
            close_arr = df["Close"].to_numpy()
            high_arr = df["High"].to_numpy()
            low_arr = df["Low"].to_numpy()
            volume_arr = df["Volume"].to_numpy()
            if date_arr.shape[0] > 246:
                dir_files[ticker] = [
                    date_arr,
                    open_arr,
                    close_arr,
                    high_arr,
                    low_arr,
                    volume_arr,
                ]
            elif verbose:
                print(file, "is empty")
        except Exception as e:
            if verbose:
                print(file, "raised an error", e.message)
    return dir_files


def dataloader_SP500_yf(
    year, PATH_org=os.path.join(".", "data", "YahooFinance", "SP500"), verbose=True
):
    # PATH = "./data/YahooFinance/SP500/"+str(year)
    PATH = os.path.join(PATH_org, str(year))
    files = os.listdir(PATH)
    dir_files = {}
    for file in files:
        try:
            ticker = file[:-18]
            df = pd.read_csv(os.path.join(PATH, file))[2:]
            # export to np arrays
            date_arr = pd.to_datetime(df["Price"]).to_numpy()
            open_arr = df["Open"].to_numpy().astype(float)
            close_arr = df["Close"].to_numpy().astype(float)
            high_arr = df["High"].to_numpy().astype(float)
            low_arr = df["Low"].to_numpy().astype(float)
            volume_arr = df["Volume"].to_numpy().astype(float)
            if date_arr.shape[0] > 246:
                dir_files[ticker] = [
                    date_arr,
                    open_arr,
                    close_arr,
                    high_arr,
                    low_arr,
                    volume_arr,
                ]
            elif verbose:
                print(file, "is empty")
        except Exception:
            if verbose:
                print(file, "raised an error")
    return dir_files


def close_array(dir_files, plot=False):
    ticker_list = list(dir_files.keys())
    # suppose same length per ticker
    prices = np.zeros((len(ticker_list), dir_files[ticker_list[0]][0].shape[0]))
    for i in range(len(ticker_list)):
        date_arr, open_arr, close_arr = dir_files[ticker_list[i]]
        prices[i, : close_arr.shape[0]] = close_arr
        if plot:
            plt.plot(date_arr, close_arr, label=ticker_list[i])
    if plot:
        plt.title("Close prices")
        plt.legend()
        plt.show()
    return prices


def get_ticker_list(year):
    df = dataloader_SP500(year, verbose=False)
    return list(df.keys())


def get_domain_list(ticker_list):
    PATH = os.path.join(".", "data", "YahooFinance", "SP500", "domains.csv")
    df = pd.read_csv(PATH)
    domains = np.unique(df["GICS Sector"])
    domain_list = []
    for ticker in ticker_list:
        try:
            domain = df[df["Symbol"] == ticker]["GICS Sector"].values[0]
            domain_card = np.argmax(domains == domain)
        except Exception:
            domain_card = 12
        domain_list += [domain_card]
    return domain_list, domains


def close_SP500(year, ticker_list=None, verbose=True):
    df = dataloader_SP500(year, verbose=False)
    if ticker_list is None:
        ticker_list = list(df.keys())
    d = len(ticker_list)
    N = df[ticker_list[0]][2].shape[0]
    X = np.zeros((N, d))
    indices = []
    for i in range(d):
        ticker = ticker_list[i]
        samples = df[ticker][2]
        if samples.shape[0] == N and (samples == 0).sum() == 0:
            X[:, i] = samples
            indices += [i]
        elif verbose:
            print(ticker, "unmatching size")
    indices = np.array(indices).astype(int)
    return X[:, indices]


def close_volume_SP500(year, ticker_list=None, verbose=True):
    df = dataloader_SP500_yf(year, PATH_org=os.path.join(".", "data_yf"), verbose=False)
    if ticker_list is None:
        ticker_list = list(df.keys())
    d = len(ticker_list)
    N = df[ticker_list[0]][2].shape[0]
    X = np.zeros((N, d))
    V = np.zeros((N, d))
    indices = []
    for i in range(d):
        try:
            ticker = ticker_list[i]
            samples = df[ticker][2]
            vol = df[ticker][5]
            if samples.shape[0] == N and (samples == 0).sum() == 0:
                X[:, i] = samples
                V[:, i] = vol
                indices += [i]
            elif verbose:
                print(ticker, "unmatching size")
        except KeyError:
            if verbose:
                print(ticker, "unfound")
    indices = np.array(indices).astype(int)
    return X[:, indices], V[:, indices]


# Daily var estimation with Yang Zhang vol
def close_var_SP500(year, ticker_list=None, verbose=True):
    df = dataloader_SP500_yf(year, PATH_org=os.path.join(".", "data_yf"), verbose=False)
    if ticker_list is None:
        ticker_list = list(df.keys())
    d = len(ticker_list)
    N = df[ticker_list[0]][2].shape[0]
    X = np.zeros((N, d))
    V = np.zeros((N, d))
    indices = []
    for i in range(d):
        try:
            ticker = ticker_list[i]
            samples = df[ticker][2]
            o = np.log(df[ticker][1])
            h = np.log(df[ticker][3])
            lo = np.log(df[ticker][4])
            c = np.log(df[ticker][2])

            # Overnight (close_t-1 -> open_t)
            c_shift = np.roll(c, 1)
            r_o = o - c_shift
            r_o[0] = 0

            # Open-to-close intraday
            r_c = c - o

            # Rogers–Satchell part
            rs = (h - o) * (h - c) + (lo - o) * (lo - c)

            # Variances
            sigma_o2 = r_o.var()
            sigma_c2 = r_c.var()
            sigma_rs = rs.mean()

            k = 0.34 / (1.34 + (len(df) + 1) / (len(df) - 1))
            vol = sigma_o2 + k * sigma_c2 + (1 - k) * sigma_rs

            if samples.shape[0] == N and (samples == 0).sum() == 0:
                X[:, i] = samples
                V[:, i] = vol
                indices += [i]
            elif verbose:
                print(ticker, "unmatching size")
        except KeyError:
            if verbose:
                print(ticker, "unfound")
    indices = np.array(indices).astype(int)
    return X[:, indices], V[:, indices]
