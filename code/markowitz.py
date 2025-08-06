import warnings

import numpy as np
import torch


def get_S(X):
    n, p = X.shape
    S = np.cov(X.T)
    return S


def get_IC(e):
    if e.std() != 0:
        IC = np.sqrt(252) * e.mean() / e.std()
    else:
        IC = 0
    return IC


def get_e(X, x):
    e = X @ x
    return e


def GMV(X_test, Sigma, wb):
    n, p = X_test.shape
    one = np.ones(p)
    try:
        x = (
            np.linalg.pinv(Sigma)
            @ one
            / (np.ones((1, p)) @ np.linalg.pinv(Sigma) @ one)
        )
    except Exception:
        x = one / p
    e = get_e(X_test, x)
    s = e.var()
    IC = get_IC(e)
    return e.sum(), s, IC


def get_IC_torch(e):
    if e.std() != 0:
        IC = np.sqrt(252) * e.mean(axis=0) / e.std(axis=0)
    else:
        IC = torch.zeors(1)
    return IC


def get_e_torch(X, x):
    e = X @ x
    return e


def GMV_torch(X_test, Sigma, wb):
    n, p = X_test.shape
    one = torch.ones(p, dtype=X_test.dtype)
    try:
        P = torch.linalg.pinv(Sigma)
        x = P @ one / (torch.ones((1, p), dtype=X_test.dtype) @ P @ one)
    except Exception:
        x = torch.ones(p, dtype=X_test.dtype, requires_grad=Sigma.requires_grad) / p
        warnings.warn("GMV: pinv failed:")
    e = get_e_torch(X_test, x)
    s = e.var()
    IC = get_IC_torch(e)
    return e.sum(axis=0), s, IC


def GMV_P_torch(X_test, P, wb):
    n, p = X_test.shape
    one = torch.ones(p, dtype=X_test.dtype)
    try:
        x = P @ one / (torch.ones((1, p), dtype=X_test.dtype) @ P @ one)
    except Exception:
        x = torch.ones(p, dtype=X_test.dtype, requires_grad=P.requires_grad) / p
        warnings.warn("GMV failed")
    e = get_e_torch(X_test, x)
    s = e.var()
    IC = get_IC_torch(e)
    return e.sum(axis=0), s, IC
