import numpy as np
import torch

def analytical_shrinkage_ewma(X, alpha, assume_centered = False):
    # X of shape (n,p), n >= 12
    beta = alpha/(1-np.exp(-alpha))
    n,p = X.shape
    if not assume_centered:
        X -= X.mean(axis=0)[np.newaxis,:]
        n -= 1
    sample = X.T @ X/n
    lambda_, u = np.linalg.eigh(sample)
    lambda_ = lambda_[max(0,p-n):p]
    L = np.repeat(lambda_[:,np.newaxis], min(p,n), axis=1)
    h = n**(-1/3)
    H = h*L.T
    x = (L - L.T)/H
    ftilde = (3/4/np.sqrt(5))*np.mean(np.maximum(1-x**2/5,0)/H,axis=1)
    Hftemp = (-3/10/np.pi)*x + (3/4/np.sqrt(5)/np.pi)*(1-x**2/5)*np.log(np.abs((np.sqrt(5)-x)/(np.sqrt(5)+x)))
    Hftemp[np.abs(x) == np.sqrt(5)] = (-3/10/np.pi)*x[np.abs(x) == np.sqrt(5)]
    Hftilde = (Hftemp/H).mean(axis=1)
    if p <= n:        
        m = np.pi*Hftilde + np.pi*ftilde*1j
        c = p/n
        theta = (np.exp(alpha*c*(1+lambda_*m))-1)/beta/c/(1-np.exp(-alpha+alpha*c*(1+lambda_*m)))
        s = (1+lambda_*m).imag/theta.imag
        dtilde = lambda_/s
    else:
        # TODO
        Hftilde0 = (1/np.pi)*(3/10/h**2+3/4/np.sqrt(5)/h*(1-1/5/h**2)*np.log((1+np.sqrt(5)*h)/(1-np.sqrt(5)*h)))*(1/lambda_).mean()
        dtilde0 = 1/(np.pi*(p-n)/n*Hftilde0)
        dtilde1 = lambda_/(np.pi**2*lambda_**2*(ftilde**2+Hftilde**2))
        dtilde = np.concatenate([dtilde0*np.ones((p-n)), dtilde1], axis=0)
    sigmatilde = u @ np.diag(dtilde) @ u.T
    return sigmatilde, dtilde

def analytical_shrinkage_prec_ewma(X, alpha, assume_centered = False):
    # X of shape (n,p), n >= 12
    beta = alpha/(1-np.exp(-alpha))
    n,p = X.shape
    if not assume_centered:
        X -= X.mean(axis=0)[np.newaxis,:]
        n -= 1
    sample = X.T @ X/n
    lambda_, u = np.linalg.eigh(sample)
    lambda_ = lambda_[max(0,p-n):p]
    L = np.repeat(lambda_[:,np.newaxis], min(p,n), axis=1)
    h = n**(-1/3)
    H = h*L.T
    x = (L - L.T)/H
    ftilde = (3/4/np.sqrt(5))*np.mean(np.maximum(1-x**2/5,0)/H,axis=1)
    Hftemp = (-3/10/np.pi)*x + (3/4/np.sqrt(5)/np.pi)*(1-x**2/5)*np.log(np.abs((np.sqrt(5)-x)/(np.sqrt(5)+x)))
    Hftemp[np.abs(x) == np.sqrt(5)] = (-3/10/np.pi)*x[np.abs(x) == np.sqrt(5)]
    Hftilde = (Hftemp/H).mean(axis=1)
    if p <= n:        
        m = np.pi*Hftilde + np.pi*ftilde*1j
        c = p/n
        theta = (np.exp(alpha*c*(1+lambda_*m))-1)/beta/c/(1-np.exp(-alpha+alpha*c*(1+lambda_*m)))
        s = (m*(1+lambda_*m)/theta).imag/m.imag
        dtilde = lambda_/s
    else:
        raise ValueError("p <= n necessary for precision nl analytical shrinkage estimation.")
    psitilde = u @ np.diag(dtilde) @ u.T
    return psitilde, dtilde
