import numpy as np
from scipy import linalg
import statsmodels.api as sm
import pandas as pd

# Classical estimate

def theta_classical(Y_t, Y_c):
    return np.mean(Y_t) - np.mean(Y_c)

def var_classical(sigma_t, sigma_c, n, m):
    return sigma_c*sigma_c/n + sigma_t*sigma_t/m

def m_star_classical(N, sigma_t, sigma_c):
    return N * sigma_t / (sigma_c + sigma_t)

# PPI CT

def mean_synthetic_control(fX_t):
    return np.mean(fX_t)

def rectifier(Y_c, fX_c):
    return np.mean(Y_c - fX_c)

def theta_PP(Y_t, Y_c, fX_t, fX_c):
    return np.mean(Y_t-fX_t) - rectifier(Y_c, fX_c)

def var_PP(sigma_t, sigma_f, sigma_delta, rho_t, n, m):
    return sigma_delta*sigma_delta/n + (sigma_t*sigma_t + sigma_f*sigma_f)/m - 2 * rho_t * sigma_t * sigma_f / m

def m_star_PP(N, sigma_delta, sigma_t, sigma_f, rho_t):
    den = sigma_delta + np.sqrt(sigma_t*sigma_t + sigma_f*sigma_f - 2 * rho_t * sigma_t * sigma_f)
    return N - N*sigma_delta/den

# PPI++ CT

def theta_PP_plus(Y_t, Y_c, fX_t, fX_c, lambda_, axis=0):
    return np.mean(Y_t, axis=axis) - lambda_ * np.mean(fX_t, axis=axis) - np.mean(Y_c, axis=axis) + lambda_ * np.mean(fX_c, axis=axis)

def var_PP_plus(sigma_t, sigma_f, sigma_c, rho_t, rho_c, n, m, lambda_):
    l2 = lambda_ * lambda_
    sf2 = sigma_f * sigma_f
    st2 = sigma_t * sigma_t
    sc2 = sigma_c * sigma_c
    return (l2 * sf2 + st2 - 2*lambda_*sigma_t*sigma_f * rho_t)/m + (l2 * sf2 + sc2 - 2*lambda_*sigma_c*sigma_f * rho_c)/n

def lambda_star(sigma_t, sigma_f, sigma_c, rho_t, rho_c, n, m):
    return (n*sigma_t*rho_t + m*sigma_c*rho_c)/(sigma_f * (m+n))

def var_theo_PP_plus(sigma_t, sigma_f, sigma_c, rho_t, rho_c, n, m, lambda_):
    return sigma_t*sigma_t/m + sigma_c*sigma_c/n - sigma_f*sigma_f*lambda_*lambda_*(1/n+1/m)

def corrcoef(a, b, axis=0):
    ea = np.mean(a, axis=axis)
    eb = np.mean(b, axis=axis)
    sa = np.std(a, axis=axis, ddof=1)
    sb = np.std(b, axis=axis, ddof=1)
    return np.sqrt(np.mean((a-ea)*(b-eb))/(sa*sb))

def classic_statistic(X, Y_t, Y_c, fX, N, sigma_t, sigma_c):
    m = int(np.round(m_star_classical(N, sigma_t, sigma_c)))
    
    sigma_t_e = np.std(Y_t, ddof=1)
    sigma_c_e = np.std(Y_c, ddof=1)
    
    n = N - m
    ate = theta_classical(Y_t, Y_c)
    v_ate = var_classical(sigma_t_e, sigma_c_e, n, m)
    return ate / np.sqrt(v_ate)

def PPI_statistic(X, Y_t_all, Y_c_all, fX, N, sigma_delta, sigma_t, sigma_f, rho_t):
    m = int(np.round(m_star_PP(N, sigma_delta, sigma_t, sigma_f, rho_t)))
    n = N-m
    X_t = X[:m]
    X_c = X[m:]
    Y_t = Y_t_all[:m]
    Y_c = Y_c_all[m:]
    fX_t = fX[:m]
    fX_c = fX[m:]
    
    sigma_t_e = np.std(Y_t, ddof=1)
    sigma_c_e = np.std(Y_c, ddof=1)
    sigma_f_e = np.std(fX, ddof=1)
    sigma_delta_e = np.std(Y_c - fX_c, ddof=1)
    
    rho_t_e = np.corrcoef(fX_t, Y_t)
    
    ate = theta_PP(Y_t, Y_c, fX_t, fX_c)
    v_ate = var_PP(sigma_t_e, sigma_f_e, sigma_delta_e, rho_t_e, n, m)
    return ate / np.sqrt(v_ate)

def PPI_plus_statistic(X, Y_t_all, Y_c_all, fX, N, sigma_c, sigma_t, sigma_f, rho_t, rho_c,
                       verbose=False, return_values=False,
                      ):
    l = []
    v = []
    for k in range(1, N):
        m = k
        n = N - k
        lambda_ = lambda_star(sigma_t, sigma_f, sigma_c, rho_t, rho_c, n, m)
        l.append(lambda_)
        var = var_PP_plus(sigma_t, sigma_f, sigma_c, rho_t, rho_c, n, m, lambda_)
        v.append(var)
    m = np.argmin(v) + 1
    lambda_ = l[m-1]
    if verbose:
        print(f"Lambda_star = {lambda_:.3} \t m_star = {m}")

    n = N-m
    X_t = X[:m]
    X_c = X[m:]
    Y_t = Y_t_all[:m]
    Y_c = Y_c_all[m:]
    fX_t = fX[:m]
    fX_c = fX[m:]
    
    sigma_t_e = np.std(Y_t, ddof=1)
    sigma_c_e = np.std(Y_c, ddof=1)
    sigma_f_e = np.std(fX, ddof=1) 
    rho_t_e = np.corrcoef(fX_t, Y_t)[0,1]
    rho_c_e = np.corrcoef(fX_c, Y_c)[0,1]
    
    ate = theta_PP_plus(Y_t, Y_c, fX_t, fX_c, lambda_)
    v_ate = var_PP_plus(sigma_t, sigma_f, sigma_c, rho_t, rho_c, n, m, lambda_)
    if return_values:
        return ate, v_ate, m, lambda_
    return ate / np.sqrt(v_ate)

def ancova2(Y, T, fX, X, N, use_covariates=True):
    if use_covariates:
        Z = np.vstack([np.ones(N), T, X, fX, T*X, T*fX])
    else:
        Z = np.vstack([np.ones(N), T, fX, T*fX])
    try:
        beta = linalg.inv(Z @ Z.T) @ Z @ Y
    except:
        beta = linalg.inv((Z @ Z.T)+np.eye(Z.shape[0])*1e-8) @ Z @ Y
    return beta

def var_ancova2(Y, T, fX, X, N, m):
    T_ = T.astype(bool)
    sigma_c = np.std(Y[~T_], ddof=1)
    sigma_t = np.std(Y[T_], ddof=1)
    if m==1:
        rho_c = 1
    else:
        rho_c = np.corrcoef(Y[~T_], fX[~T_])[0,1]
    rho_t = np.corrcoef(Y[T_], fX[T_])[0,1]
    n = N - m
    return sigma_c*sigma_c/n + sigma_t*sigma_t/m - n*m/N * (rho_c*sigma_c/n + rho_t*sigma_t/m)**2

def theta_ancova2_theo(Y_t, Y_c, fX_t, fX_c):
    rho_t = np.corrcoef(Y_t, fX_t)[0,1]
    rho_c = np.corrcoef(Y_c, fX_c)[0,1]
    sigma_c = np.std(Y_c, ddof=1)
    sigma_t = np.std(Y_t, ddof=1)
    sigma_fc = np.std(fX_c, ddof=1)
    sigma_ft = np.std(fX_t, ddof=1)
    lambda_t = sigma_t * rho_t / sigma_ft
    lambda_c = sigma_c * rho_c / sigma_fc
    return np.mean(Y_t) - lambda_t * np.mean(fX_t) - np.mean(Y_c) + lambda_c * np.mean(fX_c)

def ancova_sm(df, Y="Y", X=["X"], T="T", fX="fX", use_covariates=True, verbose=True, return_model=False):
    
    # Fit ANCOVA model
    if use_covariates:
        model = sm.formula.ols(f'{Y} ~ {" + ".join([T+"*"+x for x in X])}{T}*{fX}', data=df).fit()
    else:
        model = sm.formula.ols(f'{Y} ~ {T}*{fX}', data=df).fit()
    # Print model summary
    if verbose:
        print(model.summary())
    theta, std = model.params[T], model.bse[T]
    if return_model:
        return theta, std, model
    else:
        return theta, std

def compute_estimators(df, 
                       endpoint="Y", 
                       treatment="Treatment", 
                       prognostic="fX", 
                       covariates=["X"], 
                       names=["Classical", "PPI", "PPI++", "ANCOVAII"]):
    a = np.zeros(4)
    s = np.zeros(4)
    T = df[treatment].values
    Y = df[endpoint].values
    X = df[covariates].values
    fX = df[prognostic].values
    treated = (T==1)
    Y_c = Y[~treated]
    Y_t = Y[treated]
    X_c = X[~treated]
    X_t = X[treated]
    fX_c = fX[~treated]
    fX_t = fX[treated]
    N = len(T)
    m = T.sum()
    n = N - m
    
    # Values of sigma and rho on data
    sigma_c = np.std(Y_c, ddof=1)
    sigma_t = np.std(Y_t, ddof=1)
    sigma_f = np.std(fX, ddof=1)
    sigma_delta = np.std(Y_c-fX_c, ddof=1)
    rho_t = np.corrcoef(fX_t, Y_t)[1, 0]
    rho_c = np.corrcoef(fX_c, Y_c)[1, 0]
    
    a[0] = theta_classical(Y_t, Y_c) 
    s[0] = var_classical(sigma_t, sigma_c, n, m)
    
    # PPI :
    
    a[1] = theta_PP(Y_t, Y_c, fX_t, fX_c)
    s[1] = var_PP(sigma_t, sigma_f, sigma_delta, rho_t, n, m)
    
    # PPI ++ :
    
    lambda_ = lambda_star(sigma_t, sigma_f, sigma_c, rho_t, rho_c, n, m)
    theta = theta_PP_plus(Y_t, Y_c, fX_t, fX_c, lambda_)
    var = var_PP_plus(sigma_t, sigma_f, sigma_c, rho_t, rho_c, n, m, lambda_)
    
    a[2] = theta
    s[2] = var

    df_ = df.copy()
    for col in [treatment, prognostic]+covariates:
        df_[col] = df[col]-df[col].mean()
    theta, std = ancova_sm(df_, Y=endpoint, fX=prognostic, T=treatment, X=covariates, use_covariates=False, verbose=False)
    a[3] = theta
    s[3] = std*std
    return pd.DataFrame(a[np.newaxis,:], columns=names), pd.DataFrame(s[np.newaxis,:], columns=names)

class Estimator:

    def __init__(self):

        self.T = None
        self.Y = None
        self.X = None
        self.fX = None
        self.Y_c = None
        self.Y_t = None
        self.X_c = None
        self.X_t = None
        self.fX_c = None
        self.fX_t = None
        self.N = None
        self.m = None
        self.n = None
        self.sigma_c = None
        self.sigma_t = None
        self.sigma_f = None
        self.rho_t = None
        self.rho_c = None
        self.ATE = None
        self.var = None

    def fit(self,
            df,
            endpoint="Y", 
            treatment="Treatment",
            covariates=[],
            prognostic="fX"):
        self.T = df[treatment].values
        self.Y = df[endpoint].values
        self.X = df[covariates].values
        self.fX = df[prognostic].values
        treated = (self.T==1)
        self.Y_c = self.Y[~treated]
        self.Y_t = self.Y[treated]
        self.X_c = self.X[~treated]
        self.X_t = self.X[treated]
        self.fX_c = self.fX[~treated]
        self.fX_t = self.fX[treated]
        self.N = len(self.T)
        self.m = self.T.sum()
        self.n = self.N - self.m
        self.sigma_c = np.std(self.Y_c, ddof=1)
        self.sigma_t = np.std(self.Y_t, ddof=1)
        self.sigma_f = np.std(self.fX, ddof=1)
        self.rho_t = np.corrcoef(self.fX_t, self.Y_t)[1, 0]
        self.rho_c = np.corrcoef(self.fX_c, self.Y_c)[1, 0]

        # Compute ATE and Var
        self.ATE = self.compute_ATE()
        self.var = self.compute_var()

    #@abstractmethod
    def compute_ATE(self):
        """Returns the average treatment effect"""
        return 

    #@abstractmethod
    def compute_var(self):
        """Returns the variance of the ATE"""
        return
        

class PPI(Estimator):

    def __init__(self):
        super().__init__()
        self.lambda_ = None

    def compute_ATE(self):
        if self.lambda_ is None:
            self.lambda_ = lambda_star(self.sigma_t, self.sigma_f, self.sigma_c, self.rho_t, self.rho_c, self.n, self.m) 
        return theta_PP_plus(self.Y_t, self.Y_c, self.fX_t, self.fX_c, self.lambda_)

    def compute_var(self):
        if self.lambda_ is None:
            self.lambda_ = lambda_star(self.sigma_t, self.sigma_f, self.sigma_c, self.rho_t, self.rho_c, self.n, self.m)
        return var_PP_plus(self.sigma_t, self.sigma_f, self.sigma_c, self.rho_t, self.rho_c, self.n, self.m, self.lambda_)

    def mean_synthetic_control(self):
        return mean_synthetic_control(self.fX_t)

    def rectifier(self):
        return rectifier(self.Y_c, self.fX_c)
        