import sys
import time

import numpy as np
import seaborn as sns
import torch
from ANS_ewma import (
    analytical_shrinkage_ewma_ridge_torch,
    analytical_shrinkage_ewma_torch,
    analytical_shrinkage_prec_ewma_ridge_torch,
    analytical_shrinkage_prec_ewma_torch,
)
from GIS_ewma import (
    GIS_ewma_prec_ridge_torch,
    GIS_ewma_prec_torch,
    GIS_ewma_ridge_torch,
    GIS_ewma_torch,
)
from LIS_ewma import (
    LIS_ewma_prec_ridge_torch,
    LIS_ewma_prec_torch,
    LIS_ewma_ridge_torch,
    LIS_ewma_torch,
)
from QIS_ewma import (
    QIS_ewma_prec_ridge_torch,
    QIS_ewma_prec_torch,
    QIS_ewma_ridge_torch,
    QIS_ewma_torch,
)
from weighted_LWO_estimator import wLWO_estimator_torch, wLWO_estimator_torch2

sys.path.insert(0, "./WeSpeR/code/WeSpeR_LD")
import matplotlib.pyplot as plt
from dataloader import close_SP500, get_domain_list

# from WeSpeR_LD import WeSpeR_LD
from markowitz import GMV_P_torch, GMV_torch
from WeSpeR.code.nl_formulas import nl_prec_shrinkage
from WeSpeR.code.WeSpeR_LD import WeSpeR_LD


class EWMA_model(torch.nn.Module):
    def __init__(self, a=1.0, q=0.05, lag=24, estimator="S"):
        super().__init__()
        self.a = torch.nn.Parameter(a * torch.ones(1).type(torch.float64))
        self.q = torch.nn.Parameter(q * torch.ones(1).type(torch.float64))
        self.lag = lag
        self.estimator = estimator

    def forward(self, Y):
        n, p = Y.shape
        res_s = torch.zeros(Y.shape[0] // 20 - self.lag, dtype=Y.dtype)
        res_e = torch.zeros(Y.shape[0] // 20 - self.lag, dtype=Y.dtype)
        for i in range(self.lag, Y.shape[0] // 20):
            Y_train = Y[max(0, (i - self.lag) * 20) : i * 20]
            y_min = torch.quantile(Y_train, self.q, dim=0)
            y_max = torch.quantile(Y_train, 1 - self.q, dim=0)
            Y_train = torch.max(torch.min(Y_train, y_max), y_min)
            Y_test = Y[i * 20 : (i + 1) * 20]

            alpha = -torch.log(self.a) * Y_train.shape[0]
            alpha / (1 - torch.exp(-alpha))
            w = (
                self.a
                ** torch.fliplr(
                    torch.ones(Y_train.shape[0]).cumsum(axis=0)[None, :] - 1
                )[0, :]
            )
            w = w / w.sum(axis=0) * Y_train.shape[0]
            wsq = torch.sqrt(w)

            Y_train_ewma_mean = (w[:, None] * Y_train).sum(axis=0)[None, :] / w.sum(
                axis=0
            )
            Y_train_ewma = Y_train - Y_train_ewma_mean

            P, Sigma = None, None
            if self.estimator == "S":
                Sigma = (
                    Y_train_ewma.T
                    @ (w[:, None] * Y_train_ewma)
                    / Y_train_ewma.shape[0]
                    / (1 - (w**2 / Y_train_ewma.shape[0] ** 2).sum(axis=0))
                )
            if self.estimator == "Var":
                S_ewma = (
                    Y_train_ewma.T
                    @ (w[:, None] * Y_train_ewma)
                    / Y_train_ewma.shape[0]
                    / (1 - (w**2 / Y_train_ewma.shape[0] ** 2).sum(axis=0))
                )
                Sigma = S_ewma * torch.eye(p)
            if self.estimator == "LWO":
                Sigma = wLWO_estimator_torch(
                    Y_train, w / Y_train.shape[0], assume_centered=False
                )
            if self.estimator == "ANS":
                try:
                    P = analytical_shrinkage_prec_ewma_torch(
                        wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                    )
                except ValueError:
                    Sigma = analytical_shrinkage_ewma_torch(
                        wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                    )
            if self.estimator == "GIS":
                try:
                    P = GIS_ewma_prec_torch(
                        wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                    )
                except ValueError:
                    Sigma = GIS_ewma_torch(
                        wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                    )
            if self.estimator == "QIS":
                try:
                    P = QIS_ewma_prec_torch(
                        wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                    )
                except ValueError:
                    Sigma = QIS_ewma_torch(
                        wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                    )
            if self.estimator == "LIS":
                try:
                    P = LIS_ewma_prec_torch(
                        wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                    )
                except ValueError:
                    Sigma = LIS_ewma_torch(
                        wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                    )
            if self.estimator == "WeSpeR":
                S = (
                    Y_train_ewma.T
                    @ (w[:, None] * Y_train_ewma)
                    / Y_train_ewma.shape[0]
                    / (1 - (w**2 / Y_train_ewma.shape[0] ** 2).sum(axis=0))
                )
                lambda_, U = torch.linalg.eigh(S)
                est = WeSpeR_LD(bias=True, assume_centered=True)
                est = est.fit_torch(
                    Y_train_ewma.detach(),
                    w.detach(),
                    y=None,
                    tau_init=None,
                    method="Adam",
                    n_epochs=30,
                    b=1,
                    assume_centered=True,
                    lr=5e-2,
                    momentum=0.0,
                    verbose=True,
                )
                # P = est.get_precision(d=None, wd=None,weights='ewma', w_args=[alpha])
                t_lambda = nl_prec_shrinkage(
                    lambda_.detach(),
                    est.tau_fit_.detach(),
                    torch.ones(est.tau_fit_.shape[0]) / est.tau_fit_.shape[0],
                    None,
                    None,
                    c=est.c,
                    weights="ewma",
                    w_args=[alpha.detach()],
                    method="root",
                    verbose=False,
                ).real.to(torch.float64)
                P = U @ torch.diag(t_lambda) @ U.T

            if P is None:
                P = torch.linalg.pinv(Sigma)
                e, s, IC = GMV_P_torch(Y_test, P, wb=torch.ones(p) / p)
            else:
                e, s, IC = GMV_P_torch(Y_test, P, wb=torch.ones(p) / p)
            res_s[i - self.lag] = s
            res_e[i - self.lag] = e

        vol = torch.sqrt(res_s.mean(axis=0) * 252)
        mu = res_e.mean(axis=0) / 20 * 252
        sharpe = mu / vol
        return vol, mu, sharpe

    def get_a(self):
        return self.a


class EWMA_model2(torch.nn.Module):
    def __init__(self, a=1.0, b=1.0, lag=24, estimator="S"):
        super().__init__()
        self.a = torch.nn.Parameter(a * torch.ones(1).type(torch.float64))
        self.b = torch.nn.Parameter(b * torch.ones(1).type(torch.float64))
        self.lag = lag
        self.estimator = estimator

    def forward(self, Y):
        n, p = Y.shape
        res_s = torch.zeros(Y.shape[0] // 20 - self.lag, dtype=Y.dtype)
        res_e = torch.zeros(Y.shape[0] // 20 - self.lag, dtype=Y.dtype)
        for i in range(self.lag, Y.shape[0] // 20):
            Y_train = Y[max(0, (i - self.lag) * 20) : i * 20]
            Y_test = Y[i * 20 : (i + 1) * 20]

            alpha = -torch.log(self.b) * Y_train.shape[0]
            alpha / (1 - torch.exp(-alpha))
            w = (
                self.a
                ** torch.fliplr(
                    torch.ones(Y_train.shape[0]).cumsum(axis=0)[None, :] - 1
                )[0, :]
            )
            w = w / w.sum(axis=0) * Y_train.shape[0]
            wsq = torch.sqrt(w)

            Y_train_ewma_mean = (w[:, None] * Y_train).sum(axis=0)[None, :] / w.sum(
                axis=0
            )
            Y_train_ewma = Y_train - Y_train_ewma_mean

            P, Sigma = None, None
            if self.estimator == "S":
                Sigma = (
                    Y_train_ewma.T
                    @ (w[:, None] * Y_train_ewma)
                    / Y_train_ewma.shape[0]
                    / (1 - (w**2 / Y_train_ewma.shape[0] ** 2).sum(axis=0))
                )
            if self.estimator == "Var":
                S_ewma = (
                    Y_train_ewma.T
                    @ (w[:, None] * Y_train_ewma)
                    / Y_train_ewma.shape[0]
                    / (1 - (w**2 / Y_train_ewma.shape[0] ** 2).sum(axis=0))
                )
                Sigma = S_ewma * torch.eye(p)
            if self.estimator == "LWO":
                wX = (
                    self.b
                    ** torch.fliplr(
                        torch.ones(Y_train.shape[0]).cumsum(axis=0)[None, :] - 1
                    )[0, :]
                )
                wX = wX / wX.sum(axis=0) * Y_train.shape[0]
                Sigma = wLWO_estimator_torch2(
                    Y_train,
                    wX / Y_train.shape[0],
                    w / Y_train.shape[0],
                    assume_centered=False,
                )
            if self.estimator == "ANS":
                try:
                    P = analytical_shrinkage_prec_ewma_torch(
                        wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                    )
                except ValueError:
                    Sigma = analytical_shrinkage_ewma_torch(
                        wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                    )
            if self.estimator == "GIS":
                try:
                    P = GIS_ewma_prec_torch(
                        wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                    )
                except ValueError:
                    Sigma = GIS_ewma_torch(
                        wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                    )
            if self.estimator == "QIS":
                try:
                    P = QIS_ewma_prec_torch(
                        wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                    )
                except ValueError:
                    Sigma = QIS_ewma_torch(
                        wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                    )
            if self.estimator == "LIS":
                try:
                    P = LIS_ewma_prec_torch(
                        wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                    )
                except ValueError:
                    Sigma = LIS_ewma_torch(
                        wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                    )

            if P is None:
                e, s, IC = GMV_torch(Y_test, Sigma, wb=torch.ones(p) / p)
            else:
                e, s, IC = GMV_P_torch(Y_test, P, wb=torch.ones(p) / p)
            res_s[i - self.lag] = s
            res_e[i - self.lag] = e

        vol = torch.sqrt(res_s.mean(axis=0) * 252)
        mu = res_e.mean(axis=0) / 20 * 252
        sharpe = mu / vol
        return vol, mu, sharpe

    def get_a(self):
        return self.a

    def get_b(self):
        return self.b


class EWMA_ridge_model(torch.nn.Module):
    def __init__(self, a=1.0, ridge=1e-5, lag=24, estimator="S"):
        super().__init__()
        self.a = torch.nn.Parameter(a * torch.ones(1).type(torch.float64))
        self.logridge = torch.nn.Parameter(
            torch.log(ridge * torch.ones(1).type(torch.float64))
        )
        self.lag = lag
        self.estimator = estimator

    def forward(self, Y):
        n, p = Y.shape
        res_s = torch.zeros(Y.shape[0] // 20 - self.lag, dtype=Y.dtype)
        res_e = torch.zeros(Y.shape[0] // 20 - self.lag, dtype=Y.dtype)
        for i in range(self.lag, Y.shape[0] // 20):
            Y_train = Y[max(0, (i - self.lag) * 20) : i * 20]
            Y_test = Y[i * 20 : (i + 1) * 20]

            alpha = -torch.log(self.a) * Y_train.shape[0]
            alpha / (1 - torch.exp(-alpha))
            w = (
                self.a
                ** torch.fliplr(
                    torch.ones(Y_train.shape[0]).cumsum(axis=0)[None, :] - 1
                )[0, :]
            )
            w = w / w.sum(axis=0) * Y_train.shape[0]
            wsq = torch.sqrt(w)

            Y_train_ewma_mean = (w[:, None] * Y_train).sum(axis=0)[None, :] / w.sum(
                axis=0
            )
            Y_train_ewma = Y_train - Y_train_ewma_mean

            P, Sigma = None, None
            if self.estimator == "S":
                Sigma = (
                    Y_train_ewma.T
                    @ (w[:, None] * Y_train_ewma)
                    / Y_train_ewma.shape[0]
                    / (1 - (w**2 / Y_train_ewma.shape[0] ** 2).sum(axis=0))
                )
                lambda_, U = torch.linalg.eigh(Sigma)
                P = U @ torch.diag(1 / (torch.exp(self.logridge) + lambda_)) @ U.T
            if self.estimator == "Var":
                S_ewma = (
                    Y_train_ewma.T
                    @ (w[:, None] * Y_train_ewma)
                    / Y_train_ewma.shape[0]
                    / (1 - (w**2 / Y_train_ewma.shape[0] ** 2).sum(axis=0))
                )
                lambda_ = torch.diag(S_ewma)
                P = torch.diag(1 / (torch.exp(self.logridge) + lambda_))
            if self.estimator == "LWO":
                Sigma = wLWO_estimator_torch(
                    Y_train, w / Y_train.shape[0], assume_centered=False
                )
                lambda_, U = torch.linalg.eigh(Sigma)
                P = U @ torch.diag(1 / (torch.exp(self.logridge) + lambda_)) @ U.T
            if self.estimator == "ANS":
                try:
                    P = analytical_shrinkage_prec_ewma_ridge_torch(
                        wsq[:, None] * Y_train_ewma,
                        alpha,
                        torch.exp(self.logridge),
                        assume_centered=True,
                    )
                except ValueError:
                    Sigma = analytical_shrinkage_ewma_ridge_torch(
                        wsq[:, None] * Y_train_ewma,
                        alpha,
                        torch.exp(self.logridge),
                        assume_centered=True,
                    )
            if self.estimator == "GIS":
                try:
                    P = GIS_ewma_prec_ridge_torch(
                        wsq[:, None] * Y_train_ewma,
                        alpha,
                        torch.exp(self.logridge),
                        assume_centered=True,
                    )
                except ValueError:
                    Sigma = GIS_ewma_ridge_torch(
                        wsq[:, None] * Y_train_ewma,
                        alpha,
                        torch.exp(self.logridge),
                        assume_centered=True,
                    )
            if self.estimator == "QIS":
                try:
                    P = QIS_ewma_prec_ridge_torch(
                        wsq[:, None] * Y_train_ewma,
                        alpha,
                        torch.exp(self.logridge),
                        assume_centered=True,
                    )
                except ValueError:
                    Sigma = QIS_ewma_ridge_torch(
                        wsq[:, None] * Y_train_ewma,
                        alpha,
                        torch.exp(self.logridge),
                        assume_centered=True,
                    )
            if self.estimator == "LIS":
                try:
                    P = LIS_ewma_prec_ridge_torch(
                        wsq[:, None] * Y_train_ewma,
                        alpha,
                        torch.exp(self.logridge),
                        assume_centered=True,
                    )
                except ValueError:
                    Sigma = LIS_ewma_ridge_torch(
                        wsq[:, None] * Y_train_ewma,
                        alpha,
                        torch.exp(self.logridge),
                        assume_centered=True,
                    )

            if P is None:
                P = torch.linalg.pinv(Sigma)
                e, s, IC = GMV_P_torch(Y_test, P, wb=torch.ones(p) / p)
            else:
                e, s, IC = GMV_P_torch(Y_test, P, wb=torch.ones(p) / p)
            res_s[i - self.lag] = s
            res_e[i - self.lag] = e

        vol = torch.sqrt(res_s.mean(axis=0) * 252)
        mu = res_e.mean(axis=0) / 20 * 252
        sharpe = mu / vol
        return vol, mu, sharpe

    def get_a(self):
        return self.a

    def get_ridge(self):
        return torch.exp(self.logridge)


def minimization(
    Y, a=1.0, q=0.05, lag=24, n_epochs=10, lr=1e-2, estimator="S", verbose=True
):
    model = EWMA_model(a=a, q=q, lag=lag, estimator=estimator)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    running_loss = []
    best_loss = np.inf
    best_a = a * torch.ones(1).type(torch.float64)
    best_q = 0.05 * torch.ones(1).type(torch.float64)

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
            best_q = model.q.detach()

        running_loss += [loss.item()]

        if verbose:
            if n_epochs <= 10 or i % (n_epochs // 10) == 0:
                print("Loss epoch", i, ":", loss.item())
    if verbose:
        print("Final loss:", best_loss)

    return model, best_a, best_q, np.array(running_loss)


def minimization2(
    Y, a=1.0, b=1.0, lag=24, n_epochs=10, lr=1e-2, estimator="S", verbose=True
):
    model = EWMA_model2(a=a, b=b, lag=lag, estimator=estimator)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    running_loss = []
    best_loss = np.inf
    best_a = a * torch.ones(1).type(torch.float64)
    best_b = b * torch.ones(1).type(torch.float64)

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


def minimization_ridge(
    Y, a=1.0, ridge=1e-5, lag=24, n_epochs=10, lr=1e-2, estimator="S", verbose=True
):
    model = EWMA_ridge_model(a=a, ridge=ridge, lag=lag, estimator=estimator)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    running_loss = []
    best_loss = np.inf
    best_a = a * torch.ones(1).type(torch.float64)
    best_ridge = ridge * torch.ones(1).type(torch.float64)

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
            best_ridge = model.get_ridge().detach()

        running_loss += [loss.item()]

        if verbose:
            if n_epochs <= 10 or i % (n_epochs // 10) == 0:
                print("Loss epoch", i, ":", loss.item())
    if verbose:
        print("Final loss:", best_loss)

    return model, best_a, best_ridge, np.array(running_loss)


def plot_vol(Y, a_tab, lag=24, estimator_list=["LWO"]):
    vol_tab = []
    sharpe_tab = []
    for estimator in estimator_list:
        vol_tab += [[]]
        sharpe_tab += [[]]
        for a in a_tab:
            model = EWMA_model(a=a, lag=lag, estimator=estimator)
            model.eval()
            vol, mu, sharpe = model(Y)
            vol_tab[-1] += [vol.detach().item()]
            sharpe_tab[-1] += [sharpe.detach().item()]
    vol_tab = np.array(vol_tab)
    sharpe_tab = np.array(sharpe_tab)

    plt.figure()
    for i in range(len(estimator_list)):
        plt.plot(np.array(a_tab), 100 * vol_tab[i], label=estimator_list[i])
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


def plot_spectrum(Y, a=0.99, lag=48, estimator="S"):
    eigenvalue_sets = []
    for i in range(lag, Y.shape[0] // 20):
        Y_train = Y[max(0, (i - lag) * 20) : i * 20]

        alpha = -torch.log(a) * Y_train.shape[0]
        alpha / (1 - torch.exp(-alpha))
        w = (
            a
            ** torch.fliplr(torch.ones(Y_train.shape[0]).cumsum(axis=0)[None, :] - 1)[
                0, :
            ]
        )
        w = w / w.sum(axis=0) * Y_train.shape[0]
        wsq = torch.sqrt(w)
        W = torch.diag(w)
        torch.sqrt(W)

        Y_train_ewma_mean = (w[:, None] * Y_train).sum(axis=0)[None, :] / w.sum(axis=0)
        Y_train_ewma = Y_train - Y_train_ewma_mean

        _P, Sigma = None, None
        if estimator == "S":
            Sigma = (
                Y_train_ewma.T
                @ (w[:, None] * Y_train_ewma)
                / Y_train_ewma.shape[0]
                / (1 - (w**2 / Y_train_ewma.shape[0] ** 2).sum(axis=0))
            )
        if estimator == "LWO":
            Sigma = wLWO_estimator_torch(
                Y_train, w / Y_train.shape[0], assume_centered=False
            )
        if estimator == "ANS":
            try:
                Sigma = torch.linalg.pinv(
                    analytical_shrinkage_prec_ewma_torch(
                        wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                    )
                )
            except ValueError:
                Sigma = analytical_shrinkage_ewma_torch(
                    wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                )
        if estimator == "GIS":
            try:
                Sigma = torch.linalg.pinv(
                    GIS_ewma_prec_torch(
                        wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                    )
                )
            except ValueError:
                Sigma = GIS_ewma_torch(
                    wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                )
        if estimator == "QIS":
            try:
                Sigma = torch.linalg.pinv(
                    QIS_ewma_prec_torch(
                        wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                    )
                )
            except ValueError:
                Sigma = QIS_ewma_torch(
                    wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                )
        if estimator == "LIS":
            try:
                Sigma = torch.linalg.pinv(
                    LIS_ewma_prec_torch(
                        wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                    )
                )
            except ValueError:
                Sigma = LIS_ewma_torch(
                    wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
                )
        if estimator == "WeSpeR":
            S = (
                Y_train_ewma.T
                @ (w[:, None] * Y_train_ewma)
                / Y_train_ewma.shape[0]
                / (1 - (w**2 / Y_train_ewma.shape[0] ** 2).sum(axis=0))
            )
            lambda_, U = torch.linalg.eigh(S)
            est = WeSpeR_LD(bias=True, assume_centered=True)
            est = est.fit_torch(
                Y_train_ewma.detach(),
                w.detach(),
                y=None,
                tau_init=None,
                method="Adam",
                n_epochs=30,
                b=1,
                assume_centered=True,
                lr=5e-2,
                momentum=0.0,
                verbose=True,
            )
            # P = est.get_precision(d=None, wd=None,weights='ewma', w_args=[alpha])
            t_lambda = nl_prec_shrinkage(
                lambda_.detach(),
                est.tau_fit_.detach(),
                torch.ones(est.tau_fit_.shape[0]) / est.tau_fit_.shape[0],
                None,
                None,
                c=est.c,
                weights="ewma",
                w_args=[alpha.detach()],
                method="root",
                verbose=False,
            ).real.to(torch.float64)
            Sigma = U @ torch.diag(1 / t_lambda) @ U.T

        lambda_ = torch.linalg.eigvalsh(Sigma)
        eigenvalue_sets += [1 / lambda_.numpy()]

    sorted_eigs = np.array([np.sort(eigs) for eigs in eigenvalue_sets])

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=sorted_eigs, inner="quartile")
    plt.title("Distribution of Eigenvalues of P at Each Rank, " + estimator)
    plt.xlabel("Eigenvalue Rank")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.show()


def plot_var(Y, a=0.99 * torch.ones(1, dtype=torch.float64), ridge_dict=None, lag=48):
    n, p = Y.shape
    lambda_tab = []
    tau_tab = []
    tauLWO_tab = []
    tauANS_tab = []
    tauQIS_tab = []
    for i in range(lag, Y.shape[0] // 20):
        Y_train = Y[max(0, (i - lag) * 20) : i * 20]
        Y_test = Y[i * 20 : (i + 1) * 20]

        alpha = -torch.log(a) * Y_train.shape[0]
        w = (
            a
            ** torch.fliplr(torch.ones(Y_train.shape[0]).cumsum(axis=0)[None, :] - 1)[
                0, :
            ]
        )
        w = w / w.sum(axis=0) * Y_train.shape[0]
        wsq = torch.sqrt(w)

        Y_train_ewma_mean = (w[:, None] * Y_train).sum(axis=0)[None, :] / w.sum(axis=0)
        Y_train_ewma = Y_train - Y_train_ewma_mean
        S_ewma = (
            Y_train_ewma.T
            @ (w[:, None] * Y_train_ewma)
            / Y_train_ewma.shape[0]
            / (1 - (w**2 / Y_train_ewma.shape[0] ** 2).sum(axis=0))
        )
        S_LWO = wLWO_estimator_torch(
            Y_train, w / Y_train.shape[0], assume_centered=False
        )
        P_ANS = analytical_shrinkage_prec_ewma_torch(
            wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
        )
        P_QIS = QIS_ewma_prec_torch(
            wsq[:, None] * Y_train_ewma, alpha, assume_centered=True
        )

        Y_test_mean = Y_test.mean(axis=0)[None, :]
        Y_test_centered = Y_test - Y_test_mean
        S_test = Y_test_centered.T @ Y_test_centered / (Y_test_centered.shape[0] - 1)

        lambda_, U = torch.linalg.eigh(S_ewma)
        tau = torch.diag(U.T @ S_test @ U)

        lambda_tab += [lambda_]
        tau_tab += [tau]
        tauLWO_tab += [torch.linalg.eigvalsh(S_LWO)]
        tauANS_tab += [torch.sort(1 / torch.linalg.eigvalsh(P_ANS))[0]]
        tauQIS_tab += [torch.sort(1 / torch.linalg.eigvalsh(P_QIS))[0]]

    lambda_tab = torch.concatenate(lambda_tab).numpy()
    tau_tab = torch.concatenate(tau_tab).numpy()
    tauLWO_tab = torch.concatenate(tauLWO_tab).numpy()
    tauANS_tab = torch.concatenate(tauANS_tab).numpy()
    tauQIS_tab = torch.concatenate(tauQIS_tab).numpy()
    lambda_inv = 1 / lambda_tab
    tau_inv = 1 / tau_tab
    tauLWO_inv = 1 / tauLWO_tab
    tauANS_inv = 1 / tauANS_tab
    tauQIS_inv = 1 / tauQIS_tab

    loglambda_dense = np.log(np.sort(np.random.choice(lambda_inv, 1000)))

    def kernel_regression_np(x_dense, x_obs, y_obs, bandwidth):
        weights = np.exp(
            -0.5 * ((x_dense[:, None] - x_obs[None, :]) / bandwidth) ** 2
        ) / (bandwidth * np.sqrt(2 * np.pi))
        smoothed_values = (weights * y_obs[None, :]).sum(axis=1) / weights.sum(axis=1)
        return smoothed_values

    bandwidth = 0.5
    logtau_smooth = kernel_regression_np(
        loglambda_dense, np.log(lambda_inv), np.log(tau_inv), bandwidth
    )
    logtauLWO_smooth = kernel_regression_np(
        loglambda_dense, np.log(lambda_inv), np.log(tauLWO_inv), bandwidth
    )
    logtauANS_smooth = kernel_regression_np(
        loglambda_dense, np.log(lambda_inv), np.log(tauANS_inv), bandwidth
    )
    logtauQIS_smooth = kernel_regression_np(
        loglambda_dense, np.log(lambda_inv), np.log(tauQIS_inv), bandwidth
    )
    logtauridgeS_smooth = kernel_regression_np(
        loglambda_dense,
        np.log(lambda_inv),
        np.log(1 / (ridge_dict["S"] + 1 / lambda_inv)),
        bandwidth,
    )
    logtauridgeLWO_smooth = kernel_regression_np(
        loglambda_dense,
        np.log(lambda_inv),
        np.log(1 / (ridge_dict["LWO"] + 1 / tauLWO_inv)),
        bandwidth,
    )
    logtauridgeANS_smooth = kernel_regression_np(
        loglambda_dense,
        np.log(lambda_inv),
        np.log(1 / (ridge_dict["ANS"] + 1 / tauANS_inv)),
        bandwidth,
    )
    logtauridgeQIS_smooth = kernel_regression_np(
        loglambda_dense,
        np.log(lambda_inv),
        np.log(1 / (ridge_dict["QIS"] + 1 / tauQIS_inv)),
        bandwidth,
    )

    plt.figure()
    plt.plot(loglambda_dense, logtau_smooth, label="Log of realized variance")
    plt.plot(loglambda_dense, logtauLWO_smooth, label="LWO")
    plt.plot(loglambda_dense, logtauANS_smooth, label="ANS")
    plt.plot(loglambda_dense, logtauQIS_smooth, label="QIS")
    plt.plot(loglambda_dense, logtauridgeS_smooth, label="Ridge S")
    plt.plot(loglambda_dense, logtauridgeLWO_smooth, label="Ridge LWO")
    plt.plot(loglambda_dense, logtauridgeANS_smooth, label="Ridge ANS")
    plt.plot(loglambda_dense, logtauridgeQIS_smooth, label="Ridge QIS")
    # plt.scatter(lambda_inv, tau_inv, color='red', label='Original data', zorder=5)
    plt.xlabel(r"$\log \lambda_{inv}$")
    plt.ylabel(r"$\log \tau_{inv}$")
    plt.title(r"Kernel regression of $\log \tau_{inv}$ vs $\log \lambda_{inv}$")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(np.exp(loglambda_dense), np.exp(logtau_smooth), label="Realized variance")
    plt.plot(np.exp(loglambda_dense), np.exp(logtauLWO_smooth), label="LWO")
    plt.plot(np.exp(loglambda_dense), np.exp(logtauANS_smooth), label="ANS")
    plt.plot(np.exp(loglambda_dense), np.exp(logtauQIS_smooth), label="QIS")
    plt.plot(np.exp(loglambda_dense), np.exp(logtauridgeS_smooth), label="Ridge S")
    plt.plot(np.exp(loglambda_dense), np.exp(logtauridgeLWO_smooth), label="Ridge LWO")
    plt.plot(np.exp(loglambda_dense), np.exp(logtauridgeANS_smooth), label="Ridge ANS")
    plt.plot(np.exp(loglambda_dense), np.exp(logtauridgeQIS_smooth), label="Ridge QIS")
    # plt.scatter(lambda_inv, tau_inv, color='red', label='Original data', zorder=5)
    plt.xlabel(r"$\lambda_{inv}$")
    plt.ylabel(r"$\tau_{inv}$")
    plt.title(r"Kernel regression of $\tau_{inv}$ vs $\lambda_{inv}$")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(np.exp(logtau_smooth), label="Realized variance")
    plt.plot(np.exp(logtauLWO_smooth), label="LWO")
    plt.plot(np.exp(logtauANS_smooth), label="ANS")
    plt.plot(np.exp(logtauQIS_smooth), label="QIS")
    plt.plot(np.exp(logtauridgeS_smooth), label="Ridge S")
    plt.plot(np.exp(logtauridgeLWO_smooth), label="Ridge LWO")
    plt.plot(np.exp(logtauridgeANS_smooth), label="Ridge ANS")
    plt.plot(np.exp(logtauridgeQIS_smooth), label="Ridge QIS")
    plt.plot(np.exp(loglambda_dense), label="S")
    # plt.scatter(lambda_inv, tau_inv, color='red', label='Original data', zorder=5)
    plt.xlabel(r"rank")
    plt.ylabel(r"$\tau_{inv}$")
    plt.title(r"Kernel regression of $\tau_{inv}$, ranked")
    plt.legend()
    plt.grid(True)
    plt.show(block=True)


if __name__ == "__main__":
    import matplotlib

    try:
        # matplotlib.use("tkagg")
        matplotlib.use("tkagg")
        matplotlib.pyplot.ion()
    except Exception:
        pass

    try:
        Y.shape
    except NameError:
        beg_time = time.time()
        ticker_list = [
            "AAPL",
            "ABC",
            "ABMD",
            "ABT",
            "ADBE",
            "ADI",
            "ADM",
            "ADP",
            "ADSK",
            "AEP",
            "AES",
            "AFL",
            "AIG",
            "AJG",
            "ALB",
            "ALK",
            "ALL",
            "AMAT",
            "AMD",
            "AME",
            "AMGN",
            "AON",
            "AOS",
            "APA",
            "APD",
            "APH",
            "ATO",
            "ATVI",
            "AVB",
            "AVY",
            "AXP",
            "AZO",
            "BAC",
            "BALL",
            "BAX",
            "BA",
            "BBWI",
            "BBY",
            "BDX",
            "BEN",
            "BIIB",
            "BIO",
            "BKR",
            "BK",
            "BMY",
            "BRO",
            "BSX",
            "BWA",
            "CAG",
            "CAH",
            "CAT",
            "CB",
            "CCL",
            "CDNS",
            "CHD",
            "CINF",
            "CI",
            "CLX",
            "CL",
            "CMA",
            "CMCSA",
            "CMI",
            "CMS",
            "CNP",
            "COF",
            "COO",
            "COP",
            "COST",
            "CPB",
            "CPRT",
            "CPT",
            "CSCO",
            "CSX",
            "CTAS",
            "CTRA",
            "CTXS",
            "CVS",
            "CVX",
            "C",
            "DD",
            "DE",
            "DHI",
            "DHR",
            "DISH",
            "DIS",
            "DLTR",
            "DOV",
            "DRE",
            "DRI",
            "DTE",
            "DUK",
            "DVA",
            "DVN",
            "DXC",
            "D",
            "EA",
            "ECL",
            "ED",
            "EFX",
            "EIX",
            "EL",
            "EMN",
            "EMR",
            "EOG",
            "EQR",
            "ESS",
            "ES",
            "ETN",
            "ETR",
            "EVRG",
            "EXC",
            "EXPD",
            "FAST",
            "FCX",
            "FDX",
            "FISV",
            "FITB",
            "FMC",
            "FRT",
            "F",
            "GD",
            "GE",
            "GILD",
            "GIS",
            "GLW",
            "GL",
            "GPC",
            "GWW",
            "HAL",
            "HAS",
            "HBAN",
            "HD",
            "HES",
            "HIG",
            "HOLX",
            "HON",
            "HPQ",
            "HRL",
            "HSIC",
            "HST",
            "HSY",
            "HUM",
            "IBM",
            "IDXX",
            "IEX",
            "IFF",
            "INCY",
            "INTC",
            "INTU",
            "IPG",
            "IP",
            "ITW",
            "IT",
            "IVZ",
            "JBHT",
            "JCI",
            "JKHY",
            "JNJ",
            "JPM",
            "J",
            "KEY",
            "KIM",
            "KLAC",
            "KMB",
            "KO",
            "KR",
            "K",
            "LEN",
            "LHX",
            "LH",
            "LIN",
            "LLY",
            "LMT",
            "LNC",
            "LNT",
            "LOW",
            "LRCX",
            "LUMN",
            "LUV",
            "L",
            "MAA",
            "MAS",
            "MCD",
            "MCHP",
            "MCK",
            "MCO",
            "MDT",
            "MGM",
            "MHK",
            "MKC",
            "MLM",
            "MMC",
            "MMM",
            "MNST",
            "MOS",
            "MO",
            "MRK",
            "MRO",
            "MSFT",
            "MSI",
            "MS",
            "MTB",
            "MTCH",
            "MU",
            "NDSN",
            "NEE",
            "NEM",
            "NI",
            "NKE",
            "NLOK",
            "NOC",
            "NSC",
            "NTAP",
            "NTRS",
            "NUE",
            "NVR",
            "NWL",
            "ODFL",
            "OKE",
            "OMC",
            "ORCL",
            "ORLY",
            "OXY",
            "O",
            "PAYX",
            "PCAR",
            "PEAK",
            "PEG",
            "PENN",
            "PEP",
            "PFE",
            "PGR",
            "PG",
            "PHM",
            "PH",
            "PKI",
            "PNC",
            "PNR",
            "PNW",
            "POOL",
            "PPG",
            "PPL",
            "PSA",
            "PTC",
            "PVH",
            "QCOM",
            "RCL",
            "REGN",
            "REG",
            "RE",
            "RF",
            "RHI",
            "RJF",
            "RMD",
            "ROK",
            "ROL",
            "ROP",
            "ROST",
            "RTX",
            "SBUX",
            "SCHW",
            "SEE",
            "SHW",
            "SIVB",
            "SJM",
            "SLB",
            "SNA",
            "SNPS",
            "SO",
            "SPGI",
            "SPG",
            "STE",
            "STT",
            "STZ",
            "SWKS",
            "SWK",
            "SYK",
            "SYY",
            "TAP",
            "TECH",
            "TER",
            "TFC",
            "TFX",
            "TGT",
            "TJX",
            "TMO",
            "TRMB",
            "TROW",
            "TRV",
            "TSCO",
            "TSN",
            "TT",
            "TXN",
            "TXT",
            "TYL",
            "T",
            "UDR",
            "UHS",
            "UNH",
            "UNP",
            "USB",
            "VFC",
            "VLO",
            "VMC",
            "VNO",
            "VRTX",
            "VTRS",
            "VZ",
            "WAB",
            "WAT",
            "WBA",
            "WDC",
            "WEC",
            "WELL",
            "WFC",
            "WHR",
            "WMB",
            "WMT",
            "WM",
            "WRB",
            "WST",
            "WY",
            "XEL",
            "XOM",
            "XRAY",
            "ZBRA",
            "ZION",
        ]
        domain_list, domains = get_domain_list(ticker_list)
        domain_list = np.array(domain_list)
        p = len(ticker_list)
        X = np.zeros((0, len(ticker_list)))
        for year in range(2010, 2022):
            X = np.concatenate(
                [X, close_SP500(year, ticker_list=ticker_list, verbose=True)], axis=0
            )
        Y = np.log(X[1:]) - np.log(X[:-1])
        Y = torch.from_numpy(Y)
        end_time = time.time()
        print("Data loading:", end_time - beg_time)

    beg_time = time.time()
    # model, best_a, best_q, loss = minimization(
    #     Y, a=0.99, q=0.05, lag=36, n_epochs=30, lr=2e-3, estimator="GIS", verbose=True
    # )
    # model, best_a, best_b, loss = minimization2(Y, a = 0.99, b = 0.99, \
    # lag = 48, n_epochs = 30, lr = 1e-3, estimator = 'LWO', verbose = True)
    # model, best_a, best_ridge, loss = minimization_ridge(Y, a = 0.99, \
    # ridge = 1.6012e-5, lag = 24, n_epochs = 10, lr = 1e-3, \
    # estimator = 'ANS', verbose = True)
    # end_time = time.time()
    # print("Minimization:", end_time - beg_time)
    # print("Best a =", best_a)
    # print("Best ridge =", best_ridge)
    # print("Best q =", best_q)
    # print("Best b =", best_b)

    a_tab = list(np.linspace(0.91, 1.01, 60))
    # plot_vol(Y, a_tab, lag = 48, estimator_list = ['LWO', 'S', 'GIS', 'QIS', 'ANS'])
    # plot_vol(Y, a_tab, lag = 48, estimator_list = ['WeSpeR'])

    # plot_spectrum(Y, a = 0.99*torch.ones(1, dtype=torch.float64), lag=48, \
    # estimator = 'WeSpeR')
    ridge_dict = {"S": 1.6648e-5, "LWO": 1.7472e-5, "ANS": 1.6012e-5, "QIS": 1.6012e-5}
    plot_var(
        Y, a=0.99 * torch.ones(1, dtype=torch.float64), ridge_dict=ridge_dict, lag=24
    )
