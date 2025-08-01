import numpy as np
import pandas as pd
import torch
import os
import datetime as dt
import warnings
import time

from weighted_LWO_estimator import wLWO_estimator_torch, wLWO_estimator_torch2
from GIS_ewma import GIS_ewma_torch, GIS_ewma_prec_torch
from QIS_ewma import QIS_ewma_torch, QIS_ewma_prec_torch
from LIS_ewma import LIS_ewma_torch, LIS_ewma_prec_torch
from ANS_ewma import analytical_shrinkage_ewma_torch, analytical_shrinkage_prec_ewma_torch

from markowitz import GMV_P_torch, GMV_torch
from dataloader import get_domain_list, close_SP500
import matplotlib.pyplot as plt

class EWMA_model(torch.nn.Module):
    def __init__(self, a = 1., lag = 24, estimator = 'S'):
        super().__init__()
        self.a = torch.nn.Parameter(a*torch.ones(1).type(torch.float64))
        self.lag = lag
        self.estimator = estimator
    
    def forward(self, Y):
        n, p = Y.shape
        res_s = torch.zeros(Y.shape[0]//20-self.lag, dtype=Y.dtype)
        res_e = torch.zeros(Y.shape[0]//20-self.lag, dtype=Y.dtype)
        for i in range(self.lag, Y.shape[0]//20):
            Y_train = Y[max(0, (i-self.lag)*20):i*20]
            Y_test = Y[i*20:(i+1)*20]

            alpha = -torch.log(self.a)*Y_train.shape[0]
            beta = alpha/(1-torch.exp(-alpha))
            w = self.a**torch.fliplr(torch.ones(Y_train.shape[0]).cumsum(axis=0)[None,:]-1)[0,:]
            w = w/w.sum(axis=0)*Y_train.shape[0]
            wsq = torch.sqrt(w)
            W = torch.diag(w)
            Wsq = torch.sqrt(W)

            Y_train_ewma_mean = (w[:,None]*Y_train).sum(axis=0)[None,:]/w.sum(axis=0)
            # Y_train_ewma_mean = (W @ Y_train).sum(axis=0)[None,:]/w.sum(axis=0)
            Y_train_ewma = Y_train - Y_train_ewma_mean

            P, Sigma = None, None
            if self.estimator == 'S':
                Sigma = Y_train_ewma.T @ (w[:,None]*Y_train_ewma)/Y_train_ewma.shape[0]/(1-(w**2/Y_train_ewma.shape[0]**2).sum(axis=0))
            if self.estimator == 'Var':
                S_ewma = Y_train_ewma.T @ (w[:,None]*Y_train_ewma)/Y_train_ewma.shape[0]/(1-(w**2/Y_train_ewma.shape[0]**2).sum(axis=0))
                Sigma = S_ewma*torch.eye(p)
            if self.estimator == 'LWO':
                Sigma = wLWO_estimator_torch(Y_train, w/Y_train.shape[0], assume_centered = False)
            if self.estimator == 'ANS':
                # try:
                P = analytical_shrinkage_prec_ewma_torch(wsq[:,None]*Y_train_ewma, alpha, assume_centered = True)
                # except:
                #     Sigma = analytical_shrinkage_ewma_torch(wsq[:,None]*Y_train_ewma, alpha, assume_centered = True)
            if self.estimator == 'GIS':
                try:
                    P = GIS_ewma_prec_torch(wsq[:,None]*Y_train_ewma, alpha, assume_centered = True)
                except:
                    Sigma = GIS_ewma_torch(wsq[:,None]*Y_train_ewma, alpha, assume_centered = True)
            if self.estimator == 'QIS':
                try:
                    P = QIS_ewma_prec_torch(wsq[:,None]*Y_train_ewma, alpha, assume_centered = True)
                except:
                    Sigma = QIS_ewma_torch(wsq[:,None]*Y_train_ewma, alpha, assume_centered = True)                
            if self.estimator == 'LIS':
                try:
                    P = LIS_ewma_prec_torch(wsq[:,None]*Y_train_ewma, alpha, assume_centered = True)
                except:
                    Sigma = LIS_ewma_torch(wsq[:,None]*Y_train_ewma, alpha, assume_centered = True)
                
            if P is None:
                e, s, IC = GMV_torch(Y_test, Sigma, wb=torch.ones(p)/p)
            else:
                e, s, IC = GMV_P_torch(Y_test, P, wb=torch.ones(p)/p)
            res_s[i-self.lag] = s
            res_e[i-self.lag] = e

        vol = torch.sqrt(res_s.mean(axis=0)*252)
        mu = res_e.mean(axis=0)/20*252
        sharpe = mu/vol
        return vol, mu, sharpe
    
    def get_a(self):
        return self.a

class EWMA_model2(torch.nn.Module):
    def __init__(self, a = 1., b = 1., lag = 24, estimator = 'S'):
        super().__init__()
        self.a = torch.nn.Parameter(a*torch.ones(1).type(torch.float64))
        self.b = torch.nn.Parameter(b*torch.ones(1).type(torch.float64))
        self.lag = lag
        self.estimator = estimator
    
    def forward(self, Y):
        n, p = Y.shape
        res_s = torch.zeros(Y.shape[0]//20-self.lag, dtype=Y.dtype)
        res_e = torch.zeros(Y.shape[0]//20-self.lag, dtype=Y.dtype)
        for i in range(self.lag, Y.shape[0]//20):
            Y_train = Y[max(0, (i-self.lag)*20):i*20]
            Y_test = Y[i*20:(i+1)*20]

            alpha = -torch.log(self.b)*Y_train.shape[0]
            beta = alpha/(1-torch.exp(-alpha))
            w = self.a**torch.fliplr(torch.ones(Y_train.shape[0]).cumsum(axis=0)[None,:]-1)[0,:]
            w = w/w.sum(axis=0)*Y_train.shape[0]
            wsq = torch.sqrt(w)
            W = torch.diag(w)
            Wsq = torch.sqrt(W)

            Y_train_ewma_mean = (w[:,None]*Y_train).sum(axis=0)[None,:]/w.sum(axis=0)
            # Y_train_ewma_mean = (W @ Y_train).sum(axis=0)[None,:]/w.sum(axis=0)
            Y_train_ewma = Y_train - Y_train_ewma_mean

            P, Sigma = None, None
            if self.estimator == 'S':
                Sigma = Y_train_ewma.T @ (w[:,None]*Y_train_ewma)/Y_train_ewma.shape[0]/(1-(w**2/Y_train_ewma.shape[0]**2).sum(axis=0))
            if self.estimator == 'Var':
                S_ewma = Y_train_ewma.T @ (w[:,None]*Y_train_ewma)/Y_train_ewma.shape[0]/(1-(w**2/Y_train_ewma.shape[0]**2).sum(axis=0))
                Sigma = S_ewma*torch.eye(p)
            if self.estimator == 'LWO':
                wX = self.b**torch.fliplr(torch.ones(Y_train.shape[0]).cumsum(axis=0)[None,:]-1)[0,:]
                wX = wX/wX.sum(axis=0)*Y_train.shape[0]
                Sigma = wLWO_estimator_torch2(Y_train, wX/Y_train.shape[0], w/Y_train.shape[0], assume_centered = False)
            if self.estimator == 'ANS':
                # try:
                P = analytical_shrinkage_prec_ewma_torch(wsq[:,None]*Y_train_ewma, alpha, assume_centered = True)
                # except:
                #     Sigma = analytical_shrinkage_ewma_torch(wsq[:,None]*Y_train_ewma, alpha, assume_centered = True)
            if self.estimator == 'GIS':
                try:
                    P = GIS_ewma_prec_torch(wsq[:,None]*Y_train_ewma, alpha, assume_centered = True)
                except:
                    Sigma = GIS_ewma_torch(wsq[:,None]*Y_train_ewma, alpha, assume_centered = True)
            if self.estimator == 'QIS':
                try:
                    P = QIS_ewma_prec_torch(wsq[:,None]*Y_train_ewma, alpha, assume_centered = True)
                except:
                    Sigma = QIS_ewma_torch(wsq[:,None]*Y_train_ewma, alpha, assume_centered = True)                
            if self.estimator == 'LIS':
                try:
                    P = LIS_ewma_prec_torch(wsq[:,None]*Y_train_ewma, alpha, assume_centered = True)
                except:
                    Sigma = LIS_ewma_torch(wsq[:,None]*Y_train_ewma, alpha, assume_centered = True)
                
            if P is None:
                e, s, IC = GMV_torch(Y_test, Sigma, wb=torch.ones(p)/p)
            else:
                e, s, IC = GMV_P_torch(Y_test, P, wb=torch.ones(p)/p)
            res_s[i-self.lag] = s
            res_e[i-self.lag] = e

        vol = torch.sqrt(res_s.mean(axis=0)*252)
        mu = res_e.mean(axis=0)/20*252
        sharpe = mu/vol
        return vol, mu, sharpe
    
    def get_a(self):
        return self.a
    
    def get_b(self):
        return self.b
     

def minimization(Y, a = 1., lag = 24, n_epochs = 10, lr  = 1e-2, estimator = 'S', verbose = True):
    model = EWMA_model(a = a, lag = lag, estimator = estimator)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    running_loss = []
    best_loss = np.inf
    best_a = a*torch.ones(1).type(torch.float64)

    model.train(True)
    for i in range(n_epochs):
        optimizer.zero_grad()
        vol, mu, sharpe = model(Y)
        loss = vol
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_a = model.get_a().detach()

        running_loss += [loss.item()]

        if verbose:
            if n_epochs <= 10 or i % (n_epochs // 10) == 0:
                print("Loss epoch", i, ":", loss.item())
    if verbose:
        print("Final loss:", best_loss)

    return model, best_a, np.array(running_loss)

def minimization2(Y, a = 1., b = 1., lag = 24, n_epochs = 10, lr  = 1e-2, estimator = 'S', verbose = True):
    model = EWMA_model2(a = a, b = b, lag = lag, estimator = estimator)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    running_loss = []
    best_loss = np.inf
    best_a = a*torch.ones(1).type(torch.float64)
    best_b = b*torch.ones(1).type(torch.float64)

    model.train(True)
    for i in range(n_epochs):
        optimizer.zero_grad()
        vol, mu, sharpe = model(Y)
        loss = vol
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_a = model.get_a().detach()
            best_b = model.get_b().detach()

        running_loss += [loss.item()]

        if verbose:
            if n_epochs <= 10 or i % (n_epochs // 10) == 0:
                print("Loss epoch", i, ":", loss.item())
    if verbose:
        print("Final loss:", best_loss)

    return model, best_a, best_b, np.array(running_loss)

def plot_vol(Y, a_tab, lag = 24, estimator_list = ['LWO']):
    vol_tab = []
    sharpe_tab = []
    for estimator in estimator_list:
        vol_tab += [[]]
        sharpe_tab += [[]]
        for a in a_tab:
            model = EWMA_model(a = a, lag = lag, estimator = estimator)
            vol, mu, sharpe = model(Y)
            vol_tab[-1] += [vol.detach().item()]
            sharpe_tab[-1] += [sharpe.detach().item()]
    vol_tab = np.array(vol_tab)
    sharpe_tab = np.array(sharpe_tab)

    plt.figure()
    for i in range(len(estimator_list)):
        plt.plot(np.array(a_tab), 100*vol_tab[i], label=estimator_list[i])
    plt.title("Vol")
    plt.xlabel("a")
    plt.ylabel("vol (%)")
    plt.legend()
    plt.show()

    plt.figure()
    for i in range(len(estimator_list)):
        plt.plot(np.array(a_tab), sharpe_tab[i], label=estimator_list[i])
    plt.title("Sharpe")
    plt.xlabel("a")
    plt.ylabel("Sharpe")
    plt.legend()
    plt.show()

    return 
    

if __name__ == "__main__":
    import importlib
    import matplotlib
    try:
        matplotlib.use('tkagg')
        matplotlib.pyplot.ion()
    except:
        pass

    try:
        #Y = torch.empty((500,25)).normal_()
        Y.shape
    except:
        beg_time = time.time()
        ticker_list = ['AAPL', 'ABC',     'ABMD',     'ABT',     'ADBE',     'ADI',     'ADM',     'ADP',     'ADSK',     'AEP',     'AES',     'AFL',     'AIG',     'AJG',     'ALB',     'ALK',     'ALL',     'AMAT',     'AMD',     'AME',     'AMGN',     'AON',     'AOS',     'APA',     'APD',     'APH',     'ATO',     'ATVI',     'AVB',     'AVY',     'AXP',     'AZO',     'BAC',     'BALL',     'BAX',     'BA',     'BBWI',     'BBY',     'BDX',     'BEN',     'BIIB',     'BIO',     'BKR',     'BK',     'BMY',     'BRO',     'BSX',     'BWA',     'CAG',     'CAH',     'CAT',     'CB',     'CCL',     'CDNS',     'CHD',     'CINF',     'CI',     'CLX',     'CL',     'CMA',     'CMCSA',     'CMI',     'CMS',     'CNP',     'COF',     'COO',     'COP',     'COST',     'CPB',     'CPRT',     'CPT',     'CSCO',     'CSX',     'CTAS',     'CTRA',     'CTXS',     'CVS',     'CVX',     'C',     'DD',     'DE',     'DHI',     'DHR',     'DISH',     'DIS',     'DLTR',     'DOV',     'DRE',     'DRI',     'DTE',     'DUK',     'DVA',     'DVN',     'DXC',     'D',     'EA',     'ECL',     'ED',     'EFX',     'EIX',     'EL',     'EMN',     'EMR',     'EOG',     'EQR',     'ESS',     'ES',     'ETN',     'ETR',     'EVRG',     'EXC',     'EXPD',     'FAST',     'FCX',     'FDX',     'FISV',     'FITB',     'FMC',     'FRT',     'F',     'GD',     'GE',     'GILD',     'GIS',     'GLW',     'GL',     'GPC',     'GWW',     'HAL',     'HAS',     'HBAN',     'HD',     'HES',     'HIG',     'HOLX',     'HON',     'HPQ',     'HRL',     'HSIC',     'HST',     'HSY',     'HUM',     'IBM',     'IDXX',     'IEX',     'IFF',     'INCY',     'INTC',     'INTU',     'IPG',     'IP',     'ITW',     'IT',     'IVZ',     'JBHT',     'JCI',     'JKHY',     'JNJ',     'JPM',     'J',     'KEY',     'KIM',     'KLAC',     'KMB',     'KO',     'KR',     'K',     'LEN',     'LHX',     'LH',     'LIN',     'LLY',     'LMT',     'LNC',     'LNT',     'LOW',     'LRCX',     'LUMN',     'LUV',     'L',     'MAA',     'MAS',     'MCD',     'MCHP',     'MCK',     'MCO',     'MDT',     'MGM',     'MHK',     'MKC',     'MLM',     'MMC',     'MMM',     'MNST',     'MOS',     'MO',     'MRK',     'MRO',     'MSFT',     'MSI',     'MS',     'MTB',     'MTCH',     'MU',     'NDSN',     'NEE',     'NEM',     'NI',     'NKE',     'NLOK',     'NOC',     'NSC',     'NTAP',     'NTRS',     'NUE',     'NVR',     'NWL',     'ODFL',     'OKE',     'OMC',     'ORCL',     'ORLY',     'OXY',     'O',     'PAYX',     'PCAR',     'PEAK',     'PEG',     'PENN',     'PEP',     'PFE',     'PGR',     'PG',     'PHM',     'PH',     'PKI',     'PNC',     'PNR',     'PNW',     'POOL',     'PPG',     'PPL',     'PSA',     'PTC',     'PVH',     'QCOM',     'RCL',     'REGN',     'REG',     'RE',     'RF',     'RHI',     'RJF',     'RMD',     'ROK',     'ROL',     'ROP',     'ROST',     'RTX',     'SBUX',     'SCHW',     'SEE',     'SHW',     'SIVB',     'SJM',     'SLB',     'SNA',     'SNPS',     'SO',     'SPGI',     'SPG',     'STE',     'STT',     'STZ',     'SWKS',     'SWK',     'SYK',     'SYY',     'TAP',     'TECH',     'TER',     'TFC',     'TFX',     'TGT',     'TJX',     'TMO',     'TRMB',     'TROW',     'TRV',     'TSCO',     'TSN',     'TT',     'TXN',     'TXT',     'TYL',     'T',     'UDR',     'UHS',     'UNH',     'UNP',     'USB',     'VFC',     'VLO',     'VMC',     'VNO',     'VRTX',     'VTRS',     'VZ',     'WAB',     'WAT',     'WBA',     'WDC',     'WEC',     'WELL',     'WFC',     'WHR',     'WMB',     'WMT',     'WM',     'WRB',     'WST',     'WY',     'XEL',     'XOM',     'XRAY',     'ZBRA',     'ZION']    
        domain_list, domains = get_domain_list(ticker_list)    
        domain_list = np.array(domain_list)    
        p = len(ticker_list)
        X = np.zeros((0,len(ticker_list)))
        for year in range(2010, 2022):
            X = np.concatenate([X,close_SP500(year, ticker_list=ticker_list, verbose = True)], axis=0)
        Y = np.log(X[1:]) - np.log(X[:-1])
        Y = torch.from_numpy(Y)
        end_time = time.time()
        print("Data loading:", end_time - beg_time)
    
    beg_time = time.time()
    #model, best_a, loss = minimization(Y, a = 0.99, lag = 48, n_epochs = 10, lr = 1e-3, estimator = 'ANS', verbose = True)
    #model, best_a, best_b, loss = minimization2(Y, a = 0.99, b = 0.99, lag = 48, n_epochs = 30, lr = 1e-2, estimator = 'ANS', verbose = True)
    end_time = time.time()
    print("Minimization:", end_time - beg_time)
    #print("Best a =", best_a)
    #print("Best b =", best_b)

    a_tab = list(np.linspace(0.92,1.02,60))
    plot_vol(Y, a_tab, lag = 48, estimator_list = ['LWO', 'S', 'GIS', 'QIS', 'ANS'])