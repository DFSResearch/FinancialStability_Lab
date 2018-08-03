import numpy as np
from estimation_ytm.estimation_ytm import newton_estimation
from weight_scheme import weight
from ns_func import Z, D

def grad(beta, data, coupons_cf, streak_data, rho=0.2, weight_scheme='no_weight'):
    m = data['span']
    grad_z = []
    teta_m = m / beta[3]
    grad_z.append(1)
    grad_z.append((1 / teta_m) * (1 - np.exp(- teta_m)))
    grad_z.append((1 / teta_m) * (1 - np.exp(- teta_m)) - np.exp(- teta_m))

    teta_m_der = m / (beta[3] ** 2)
    grad_z.append(((beta[1] + beta[2]) *
            ((1 / m) * (1 - np.exp(-teta_m)) - np.exp(-teta_m) / beta[3])
            - beta[2] * np.exp(-teta_m) * teta_m_der))
    #calculatting Loss
    W = weight(beta, df=data, rho=rho, weight_scheme=weight_scheme)

    non_grad = W * (data['ytm'] / 100  - Z(m, beta))
    loss_grad = np.zeros_like(beta)
    for i in range(beta.shape[0]):
        loss_grad[i] =  -2 * (non_grad  * grad_z[i]).sum()
    return loss_grad

### Simple Yield Loss
def naive_yield_Loss(beta, df, coupons_cf, streak_data, rho=0.2,
                     weight_scheme='no_weight', tau=None):
    '''
    Parameters
    ---------------
    beta: array-like
        Nelson-Siegel's vector of parameters
    df: Pandas Dataframe
        Dataframe of bonds' data
    coupons_cf: Pandas Dataframe
        Dataframe containing bonds' payment cash flows
    streak_data: Pandas Dataframe
        Dataframe containing bonds' payment calendar
    rho: float from 0 to 1, default 0.2
        Weight of oldest deal - only used in
        'vol_time', 'full_vol_time', 'volume_kzt', 'complex_volume' weight schemes
    weight_scheme: str, default 'no_weight'
        weight function used to weight deals
    tau: float, default None
        Use this parameter if only you do 3-variablie minimization.
        Parameter is only used in grid search optimization
    '''
    #if tau is given, then beta is array
    if tau is not None:
        assert beta.shape[0] == 3
        beta = np.append(beta, [tau])
    #estimating price
    #calculatting Loss
    W = weight(beta, df=df, rho=rho, weight_scheme=weight_scheme)
    Loss = (W * np.square(df.ytm.values - Z(df.span / 365, beta))).sum()
    return Loss

def yield_Loss(beta, df, coupons_cf, streak_data, rho=0.2, weight_scheme='no_weight', tau=None):
    '''
    Parameters
    ---------------
    beta: array-like
        Nelson-Siegel's vector of parameters
    df: Pandas Dataframe
        Dataframe of bonds' data
    coupons_cf: Pandas Dataframe
        Dataframe containing bonds' payment cash flows
    streak_data: Pandas Dataframe
        Dataframe containing bonds' payment calendar
    rho: float from 0 to 1, default 0.2
        Weight of oldest deal - only used in
        'vol_time', 'full_vol_time', 'volume_kzt', 'complex_volume' weight schemes
    weight_scheme: str, default 'no_weight'
        weight function used to weight deals
    tau: float, default None
        Use this parameter if only you do 3-variablie minimization.
        Parameter is only used in grid search optimization
    '''
    #if tau is given, then beta is array
    if tau is not None:
        assert beta.shape[0] == 3
        beta = np.append(beta, [tau])
    ind = df.index
    #estimating price
    Price = (D(streak_data[ind], beta) * coupons_cf[ind]).sum().values
    if (Price <= 0).any():
        Loss = 1e100
    else:
        ytm_hat = np.array([newton_estimation(df.iloc[i], Price[i], coupons_cf, streak_data, maxiter=200)
                            for i in range(df.shape[0])])
        #calculatting Loss
        W = weight(beta, df=df, rho=rho, weight_scheme=weight_scheme)
        Loss = (W * np.square(df['ytm'].values - ytm_hat)).sum()
    return Loss

def price_Loss(beta, df, coupons_cf, streak_data,
               rho=0.2, weight_scheme='rev_span', tau=None):
    '''
    Parameters
    ---------------
    beta: array-like
        Nelson-Siegel's vector of parameters
    df: Pandas Dataframe
        Dataframe of bonds' data
    coupons_cf: Pandas Dataframe
        Dataframe containing bonds' payment cash flows
    streak_data: Pandas Dataframe
        Dataframe containing bonds' payment calendar
    rho: float from 0 to 1, default 0.2
        Weight of oldest deal - only used in
        'vol_time', 'full_vol_time', 'volume_kzt', 'complex_volume' weight schemes
    weight_scheme: str, default 'no_weight'
        weight function used to weight deals
    tau: float, default None
        Use this parameter if only you do 3-variablie minimization.
        Parameter is only used in grid search optimization
    '''
    #if tau is given, then beta is array
    if tau is not None:
        assert beta.shape[0] == 3
        beta = np.append(beta, [tau])
    ind = df.index
    Price = (D(streak_data[ind], beta) * coupons_cf[ind]).sum().values
    Loss = (weight(beta, df=df, rho=rho, weight_scheme=weight_scheme) *
            np.square(df.stand_price.values - Price)).sum()
    return Loss