import numpy as np
import torch

def wLWO_estimator_oracle(X, w, cov, assume_centered = False, S_r = None, est = False):
    X = X.T
    p, n = X.shape
    Id = np.eye(p)
    
    W = np.diag(w)
    
    if not assume_centered:
        X_mean = (X @ W).sum(axis=1)[:, None]/W.sum()
        X = X - X_mean
        S = X @ W @ X.T/(1-(W**2).sum())
    else:
        S = X @ W @ X.T
    
    try:
        if S_r == None:
            S_r = Id
    except ValueError:
        if (S_r**2).sum() == 0:
            S_r = Id
    S_r /= np.linalg.norm(S_r, ord = 'fro')/np.sqrt(p)
    
    mu = np.trace(S @ S_r.T)/p
    S2 = np.linalg.norm(S, ord='fro')**2 
    delta2_opt = S2/p - mu**2
    alpha2_opt = np.trace(S @ cov.T)/p - mu*np.trace(cov @ S_r.T)/p
    beta2_opt = -np.trace(S @ cov.T)/p*np.trace(S @ S_r.T)/p + S2/p*np.trace(S_r @ cov.T)/p
    S_opt = alpha2_opt/delta2_opt*S + beta2_opt/delta2_opt*S_r
      
    if est:
        return S_opt, np.array([alpha2_opt/delta2_opt, beta2_opt/delta2_opt])
    return S_opt


def wLWO_estimator(X, w, assume_centered = False, S_r = None, est = False):
    X = X.T
    if est:
      if not assume_centered:
          S_star, c = wLWO_estimator_unknown_mean(X, w, S_r, est)
      else:
          S_star, c = wLWO_estimator_known_mean(X, w, S_r, est)
      return S_star, c
    if not assume_centered:
        S_star = wLWO_estimator_unknown_mean(X, w, S_r, est)
    else:
        S_star = wLWO_estimator_known_mean(X, w, S_r, est)
    return S_star

def wLWO_estimator_known_mean(X, w, S_r = None, est = False):
    p, n = X.shape
    Id = np.eye(p)
    W = np.diag(w)
    S = X @ W @ X.T
    
    try:
        if S_r == None:
            S_r = Id
    except ValueError:
        if (S_r**2).sum() == 0:
            S_r = Id
    S_r /= np.linalg.norm(S_r, ord = 'fro')/np.sqrt(p)
    
    mu = np.trace(S @ S_r.T)/p
    delta2 = np.linalg.norm(S-mu*S_r, ord = 'fro')**2/p
    S2 = np.linalg.norm(S, ord = 'fro')**2/p
    beta2 = ((X**2).sum(axis=0)**2*w**2).sum()/p - S2*(w**2).sum()
    beta2 = beta2/(1-(w**2).sum())
    beta2 = min(beta2, delta2)
    
    shrinkage = beta2/delta2
    if est:
        c = np.array([1-shrinkage, shrinkage*mu])
        return shrinkage*mu*S_r + (1 - shrinkage)*S, c
    return shrinkage*mu*S_r + (1 - shrinkage)*S

def wLWO_estimator_unknown_mean(X, w, S_r = None, est = False):
    p, n = X.shape
    Id = np.eye(p)
    W = np.diag(w)
    X_mean = (X @ W).sum(axis=1)[:,None]/W.sum()
    X = X - X_mean
    S = X @ W @ X.T/(1-(W**2).sum())
    
    try:
        if S_r == None:
            S_r = Id
    except ValueError:
        if (S_r**2).sum() == 0:
            S_r = Id
    S_r /= np.linalg.norm(S_r, ord = 'fro')/np.sqrt(p)
    
    
    C = 1-(W**2).sum()

    C11 = (w**2*(1-w)**4 - w**6).sum() + (w**2).sum()*(w**4).sum()
    C22 = (w**2).sum()**3 - 3*(w**2).sum()*(w**4).sum() + 2*(w**6).sum() + (w**2).sum()*(2*w**2*(1-w)**2).sum() - 2*(w**4*(1-w)**2).sum()
    C33 = (w**2).sum()*(4*w**2*(1-w)**2).sum() - (4*w**4*(1-w)**2).sum() + 2*(w**2).sum()**3 - 6*(w**2).sum()*(w**4).sum() + 4*(w**6).sum()
    
    q1 = 2*(w**2*(1-w)).sum()**2 - 2*(w**4*(1-w)**2).sum()
    q2 = (w**2).sum()**2 - 4*(w**3).sum()*(w**2).sum() - (w**4).sum() + 4*(w**5).sum() + 2*(w**3).sum()**2 - ((w**2).sum()**2*w**2 - 4*w**4*(w**2).sum() - w**2*(w**4).sum() + 6*w**6).sum()
    q3 = -4*(w**2).sum()*(w**2*(1-w)).sum() + 4*(w**4*(1-w)).sum() + 4*(w**3).sum()*(w**2*(1-w)).sum() + (w**2).sum()*(4*w**3*(1-w)).sum() - 8*(w**5*(1-w)).sum()
    q4 = 2*(w*(1-2*w)**2*((w**2).sum()-w**2)).sum() - 2*(w**3*(1-2*w)**2).sum() - 2*(w**2*(1-2*w)**2*((w**2).sum() - 2*w**2)).sum()
    
    D11 = 2*(w**3*(1-w)**2).sum() - 2*(w**4*(1-w)**2).sum() +(w**4).sum() - 2*(w**5).sum() - (w**2).sum()*(w**4).sum() + 2*(w**6).sum()
    D22 = q1 + q2 + q3
    D33 = 2*q2 + q4 + (w**3).sum()**2 + 2*(w**2*(1-w)).sum()**2 + (w*(1-w)**2).sum()**2 - (w**2*(w**2+(1-w)**2)**2).sum()
    
    E11 = 2*(w**3*(1-w)**2).sum() + (w**4).sum() - 2*(w**5).sum() - 2*(w**4*(1-w)**2).sum() - (w**2).sum()*(w**4).sum() + 2*(w**6).sum()
    E22 = q2 + (w*(1-w)**2).sum()**2 + (w**3).sum()**2 + 2*(w*((1-w)**2+w**2)*((w**2).sum() - w**2 - (w**3).sum())).sum() - (w**2*(1-w)**4 + w**6 + 2*w**2*(w**2 + (1-w)**2)*((w**2).sum() - 2*w**2)).sum()
    E33 = 2*q2 + 4*(w**2*(1-w)).sum()**2 - 4*(w**4*(1-w)**2).sum() - 8*(w**2*(1-w)*((w**2).sum() - w**2 - (w**3).sum())).sum() + 8*(w**3*(1-w)*((w**2).sum()-2*w**2)).sum()
    
    P = np.array([[C11, C11 + D11, C11 + E11],
                  [C22, C22 + D22, C22 + E22],
                  [C33, C33 + D33, C33 + E33]])/C**2
    g = np.array([0,0,-1])[:, None]
    h = np.linalg.pinv(P) @ g
    
    a = h[0]
    b = h[1] + 1 
    c = h[2]  
    
    X2 = ((X**2).sum(axis=0)**2*w**2/C**2).sum()
    S2 = np.linalg.norm(S, ord='fro')**2 
    tS = np.trace(S)
    
    mu = np.trace(S @ S_r.T)/p
    delta2 = np.linalg.norm(S-mu*S_r, ord = 'fro')**2
    beta2 = a*X2 + b*S2 + c*tS**2
    beta2 = max(beta2, 0)
    beta2 = min(beta2, delta2)
    
    shrinkage = beta2/delta2
    if est:
        c = np.array([1-shrinkage, shrinkage*mu])
        return shrinkage*mu*S_r + (1 - shrinkage)*S, c
    return shrinkage*mu*S_r + (1 - shrinkage)*S


def wLWO_estimator_torch(X, w, assume_centered = False, S_r = None, est = False):
    X = X.T
    if est:
      if not assume_centered:
          S_star, c = wLWO_estimator_unknown_mean_torch(X, w, S_r, est)
      else:
          S_star, c = wLWO_estimator_known_mean_torch(X, w, S_r, est)
      return S_star, c
    if not assume_centered:
        S_star = wLWO_estimator_unknown_mean_torch(X, w, S_r, est)
    else:
        S_star = wLWO_estimator_known_mean_torch(X, w, S_r, est)
    return S_star

def wLWO_estimator_known_mean_torch(X, w, S_r = None, est = False):
    p, n = X.shape
    Id = torch.eye(p, dtype = X.dtype)
    W = torch.diag(w)
    S = X @ (X*w[None,:]).T
    
    try:
        if S_r == None:
            S_r = Id
    except ValueError:
        if (S_r**2).sum(axis=[0,1]) == 0:
            S_r = Id
    S_r /= torch.linalg.norm(S_r, ord = 'fro')/np.sqrt(p)
    
    mu = torch.trace(S @ S_r.T)/p
    delta2 = torch.linalg.norm(S-mu*S_r, ord = 'fro')**2/p
    S2 = torch.linalg.norm(S, ord = 'fro')**2/p
    beta2 = ((X**2).sum(axis=0)**2*w**2).sum(axis=0)/p - S2*(w**2).sum(axis=0)
    beta2 = beta2/(1-(w**2).sum(axis=0))
    beta2 = min(beta2, delta2)
    
    shrinkage = beta2/delta2
    if est:
        c = torch.tensor([1-shrinkage, shrinkage*mu])
        return shrinkage*mu*S_r + (1 - shrinkage)*S, c
    return shrinkage*mu*S_r + (1 - shrinkage)*S

def wLWO_estimator_unknown_mean_torch(X, w, S_r = None, est = False):
    p, n = X.shape
    Id = torch.eye(p, dtype = X.dtype)
    X_mean = (X*w[None,:]).sum(axis=1)[:,None]/w.sum(axis=0)
    X = X - X_mean
    S = X @ (X*w[None,:]).T/(1-(w**2).sum(axis=0))
    
    try:
        if S_r == None:
            S_r = Id
    except ValueError:
        if (S_r**2).sum(axis=[0,1]) == 0:
            S_r = Id
    S_r /= torch.linalg.norm(S_r, ord = 'fro')/np.sqrt(p)
    
    
    C = 1-(w**2).sum(axis=0)

    C11 = (w**2*(1-w)**4 - w**6).sum(axis=0) + (w**2).sum(axis=0)*(w**4).sum(axis=0)
    C22 = (w**2).sum(axis=0)**3 - 3*(w**2).sum(axis=0)*(w**4).sum(axis=0) + 2*(w**6).sum(axis=0) + (w**2).sum(axis=0)*(2*w**2*(1-w)**2).sum(axis=0) - 2*(w**4*(1-w)**2).sum(axis=0)
    C33 = (w**2).sum(axis=0)*(4*w**2*(1-w)**2).sum(axis=0) - (4*w**4*(1-w)**2).sum(axis=0) + 2*(w**2).sum(axis=0)**3 - 6*(w**2).sum(axis=0)*(w**4).sum(axis=0) + 4*(w**6).sum(axis=0)
    
    q1 = 2*(w**2*(1-w)).sum(axis=0)**2 - 2*(w**4*(1-w)**2).sum(axis=0)
    q2 = (w**2).sum(axis=0)**2 - 4*(w**3).sum(axis=0)*(w**2).sum(axis=0) - (w**4).sum(axis=0) + 4*(w**5).sum(axis=0) + 2*(w**3).sum(axis=0)**2 - ((w**2).sum(axis=0)**2*w**2 - 4*w**4*(w**2).sum(axis=0) - w**2*(w**4).sum(axis=0) + 6*w**6).sum(axis=0)
    q3 = -4*(w**2).sum(axis=0)*(w**2*(1-w)).sum(axis=0) + 4*(w**4*(1-w)).sum(axis=0) + 4*(w**3).sum(axis=0)*(w**2*(1-w)).sum(axis=0) + (w**2).sum(axis=0)*(4*w**3*(1-w)).sum(axis=0) - 8*(w**5*(1-w)).sum(axis=0)
    q4 = 2*(w*(1-2*w)**2*((w**2).sum(axis=0)-w**2)).sum(axis=0) - 2*(w**3*(1-2*w)**2).sum(axis=0) - 2*(w**2*(1-2*w)**2*((w**2).sum(axis=0) - 2*w**2)).sum(axis=0)
    
    D11 = 2*(w**3*(1-w)**2).sum(axis=0) - 2*(w**4*(1-w)**2).sum(axis=0) +(w**4).sum(axis=0) - 2*(w**5).sum(axis=0) - (w**2).sum(axis=0)*(w**4).sum(axis=0) + 2*(w**6).sum(axis=0)
    D22 = q1 + q2 + q3
    D33 = 2*q2 + q4 + (w**3).sum(axis=0)**2 + 2*(w**2*(1-w)).sum(axis=0)**2 + (w*(1-w)**2).sum(axis=0)**2 - (w**2*(w**2+(1-w)**2)**2).sum(axis=0)
    
    E11 = 2*(w**3*(1-w)**2).sum(axis=0) + (w**4).sum(axis=0) - 2*(w**5).sum(axis=0) - 2*(w**4*(1-w)**2).sum(axis=0) - (w**2).sum(axis=0)*(w**4).sum(axis=0) + 2*(w**6).sum(axis=0)
    E22 = q2 + (w*(1-w)**2).sum(axis=0)**2 + (w**3).sum(axis=0)**2 + 2*(w*((1-w)**2+w**2)*((w**2).sum(axis=0) - w**2 - (w**3).sum(axis=0))).sum(axis=0) - (w**2*(1-w)**4 + w**6 + 2*w**2*(w**2 + (1-w)**2)*((w**2).sum(axis=0) - 2*w**2)).sum(axis=0)
    E33 = 2*q2 + 4*(w**2*(1-w)).sum(axis=0)**2 - 4*(w**4*(1-w)**2).sum(axis=0) - 8*(w**2*(1-w)*((w**2).sum(axis=0) - w**2 - (w**3).sum(axis=0))).sum(axis=0) + 8*(w**3*(1-w)*((w**2).sum(axis=0)-2*w**2)).sum(axis=0)
    
    P = torch.zeros((3,3), dtype=X.dtype)
    P[0,0] = C11; P[0,1] = C11 + D11; P[0,2] = C11 + E11
    P[1,0] = C22; P[1,1] = C22 + D22; P[1,2] = C22 + E22
    P[2,0] = C33; P[2,1] = C33 + D33; P[2,2] = C33 + E33
    P = P/C**2

    g = torch.tensor([0,0,-1], dtype=X.dtype)[:, None]
    h = torch.linalg.pinv(P) @ g
    
    a = h[0]
    b = h[1] + 1 
    c = h[2]  
    
    X2 = ((X**2).sum(axis=0)**2*w**2/C**2).sum(axis=0)
    S2 = torch.linalg.norm(S, ord='fro')**2 
    tS = torch.trace(S)
    
    mu = torch.trace(S @ S_r.T)/p
    delta2 = torch.linalg.norm(S-mu*S_r, ord = 'fro')**2
    beta2 = a*X2 + b*S2 + c*tS**2
    beta2 = max(beta2, 0)
    beta2 = min(beta2, delta2)
    
    shrinkage = beta2/delta2
    if est:
        c = torch.tensor([1-shrinkage, shrinkage*mu])
        return shrinkage*mu*S_r + (1 - shrinkage)*S, c
    return shrinkage*mu*S_r + (1 - shrinkage)*S

def wLWO_estimator_torch2(X, wX, w, assume_centered = False, S_r = None, est = False):
    X = X.T
    if est:
      if not assume_centered:
          S_star, c = wLWO_estimator_unknown_mean_torch2(X, wX, w, S_r, est)
      else:
          S_star, c = wLWO_estimator_known_mean_torch2(X, wX, w, S_r, est)
      return S_star, c
    if not assume_centered:
        S_star = wLWO_estimator_unknown_mean_torch2(X, wX, w, S_r, est)
    else:
        S_star = wLWO_estimator_known_mean_torch2(X, wX, w, S_r, est)
    return S_star

def wLWO_estimator_known_mean_torch2(X, wX, w, S_r = None, est = False):
    p, n = X.shape
    Id = torch.eye(p, dtype = X.dtype)
    S = X @ (X*wX[None,:]).T
    
    try:
        if S_r == None:
            S_r = Id
    except ValueError:
        if (S_r**2).sum(axis=[0,1]) == 0:
            S_r = Id
    S_r /= torch.linalg.norm(S_r, ord = 'fro')/np.sqrt(p)
    
    mu = torch.trace(S @ S_r.T)/p
    delta2 = torch.linalg.norm(S-mu*S_r, ord = 'fro')**2/p
    S2 = torch.linalg.norm(S, ord = 'fro')**2/p
    beta2 = ((X**2).sum(axis=0)**2*w**2).sum(axis=0)/p - S2*(w**2).sum(axis=0)
    beta2 = beta2/(1-(w**2).sum(axis=0))
    beta2 = min(beta2, delta2)
    
    shrinkage = beta2/delta2
    if est:
        c = torch.tensor([1-shrinkage, shrinkage*mu])
        return shrinkage*mu*S_r + (1 - shrinkage)*S, c
    return shrinkage*mu*S_r + (1 - shrinkage)*S

def wLWO_estimator_unknown_mean_torch2(X, wX, w, S_r = None, est = False):
    p, n = X.shape
    Id = torch.eye(p, dtype = X.dtype)
    X_mean = (X*wX[None,:]).sum(axis=1)[:,None]/wX.sum(axis=0)
    X = X - X_mean
    S = X @ (X*wX[None,:]).T/(1-(wX**2).sum(axis=0))
    
    try:
        if S_r == None:
            S_r = Id
    except ValueError:
        if (S_r**2).sum(axis=[0,1]) == 0:
            S_r = Id
    S_r /= torch.linalg.norm(S_r, ord = 'fro')/np.sqrt(p)
    
    
    C = 1-(w**2).sum(axis=0)

    C11 = (w**2*(1-w)**4 - w**6).sum(axis=0) + (w**2).sum(axis=0)*(w**4).sum(axis=0)
    C22 = (w**2).sum(axis=0)**3 - 3*(w**2).sum(axis=0)*(w**4).sum(axis=0) + 2*(w**6).sum(axis=0) + (w**2).sum(axis=0)*(2*w**2*(1-w)**2).sum(axis=0) - 2*(w**4*(1-w)**2).sum(axis=0)
    C33 = (w**2).sum(axis=0)*(4*w**2*(1-w)**2).sum(axis=0) - (4*w**4*(1-w)**2).sum(axis=0) + 2*(w**2).sum(axis=0)**3 - 6*(w**2).sum(axis=0)*(w**4).sum(axis=0) + 4*(w**6).sum(axis=0)
    
    q1 = 2*(w**2*(1-w)).sum(axis=0)**2 - 2*(w**4*(1-w)**2).sum(axis=0)
    q2 = (w**2).sum(axis=0)**2 - 4*(w**3).sum(axis=0)*(w**2).sum(axis=0) - (w**4).sum(axis=0) + 4*(w**5).sum(axis=0) + 2*(w**3).sum(axis=0)**2 - ((w**2).sum(axis=0)**2*w**2 - 4*w**4*(w**2).sum(axis=0) - w**2*(w**4).sum(axis=0) + 6*w**6).sum(axis=0)
    q3 = -4*(w**2).sum(axis=0)*(w**2*(1-w)).sum(axis=0) + 4*(w**4*(1-w)).sum(axis=0) + 4*(w**3).sum(axis=0)*(w**2*(1-w)).sum(axis=0) + (w**2).sum(axis=0)*(4*w**3*(1-w)).sum(axis=0) - 8*(w**5*(1-w)).sum(axis=0)
    q4 = 2*(w*(1-2*w)**2*((w**2).sum(axis=0)-w**2)).sum(axis=0) - 2*(w**3*(1-2*w)**2).sum(axis=0) - 2*(w**2*(1-2*w)**2*((w**2).sum(axis=0) - 2*w**2)).sum(axis=0)
    
    D11 = 2*(w**3*(1-w)**2).sum(axis=0) - 2*(w**4*(1-w)**2).sum(axis=0) +(w**4).sum(axis=0) - 2*(w**5).sum(axis=0) - (w**2).sum(axis=0)*(w**4).sum(axis=0) + 2*(w**6).sum(axis=0)
    D22 = q1 + q2 + q3
    D33 = 2*q2 + q4 + (w**3).sum(axis=0)**2 + 2*(w**2*(1-w)).sum(axis=0)**2 + (w*(1-w)**2).sum(axis=0)**2 - (w**2*(w**2+(1-w)**2)**2).sum(axis=0)
    
    E11 = 2*(w**3*(1-w)**2).sum(axis=0) + (w**4).sum(axis=0) - 2*(w**5).sum(axis=0) - 2*(w**4*(1-w)**2).sum(axis=0) - (w**2).sum(axis=0)*(w**4).sum(axis=0) + 2*(w**6).sum(axis=0)
    E22 = q2 + (w*(1-w)**2).sum(axis=0)**2 + (w**3).sum(axis=0)**2 + 2*(w*((1-w)**2+w**2)*((w**2).sum(axis=0) - w**2 - (w**3).sum(axis=0))).sum(axis=0) - (w**2*(1-w)**4 + w**6 + 2*w**2*(w**2 + (1-w)**2)*((w**2).sum(axis=0) - 2*w**2)).sum(axis=0)
    E33 = 2*q2 + 4*(w**2*(1-w)).sum(axis=0)**2 - 4*(w**4*(1-w)**2).sum(axis=0) - 8*(w**2*(1-w)*((w**2).sum(axis=0) - w**2 - (w**3).sum(axis=0))).sum(axis=0) + 8*(w**3*(1-w)*((w**2).sum(axis=0)-2*w**2)).sum(axis=0)
    
    P = torch.zeros((3,3), dtype=X.dtype)
    P[0,0] = C11; P[0,1] = C11 + D11; P[0,2] = C11 + E11
    P[1,0] = C22; P[1,1] = C22 + D22; P[1,2] = C22 + E22
    P[2,0] = C33; P[2,1] = C33 + D33; P[2,2] = C33 + E33
    P = P/C**2

    g = torch.tensor([0,0,-1], dtype=X.dtype)[:, None]
    h = torch.linalg.pinv(P) @ g
    
    a = h[0]
    b = h[1] + 1 
    c = h[2]  
    
    X2 = ((X**2).sum(axis=0)**2*w**2/C**2).sum(axis=0)
    S2 = torch.linalg.norm(S, ord='fro')**2 
    tS = torch.trace(S)
    
    mu = torch.trace(S @ S_r.T)/p
    delta2 = torch.linalg.norm(S-mu*S_r, ord = 'fro')**2
    beta2 = a*X2 + b*S2 + c*tS**2
    beta2 = max(beta2, 0)
    beta2 = min(beta2, delta2)
    
    shrinkage = beta2/delta2
    if est:
        c = torch.tensor([1-shrinkage, shrinkage*mu])
        return shrinkage*mu*S_r + (1 - shrinkage)*S, c
    return shrinkage*mu*S_r + (1 - shrinkage)*S