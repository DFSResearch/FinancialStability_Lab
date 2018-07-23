### Estimating YTM and YTM filtering
import numpy as np
from scipy.optimize import newton, minimize_scalar
from . import ytm_approximation

#loss yield used for estimating real ytm from real deal prices
def loss_yield(bond, coupons_cf, streak_data):
    ind = bond.name
    delta = streak_data[ind].iloc[0]
    streak_ = np.arange(0, streak_data[ind].dropna().shape[0]) + delta * bond['annual_freq']
    def ytm_loss(ytm): 
        return np.square(
                (coupons_cf[ind].dropna() / ((1 + ytm / bond['annual_freq']) ** streak_)).sum() 
                - bond['stand_price']
                )
    try:
        ytm = minimize_scalar(ytm_loss, tol=10e-14)
    except Exception as e:
        print(e, ind)
    return ytm.x

#new ytm adds ytm column in dataset while saving older ytm as ytm_kase 
def new_ytm(df, coupons_cf, streak_data):
    '''
    Change ytm column of dataframe on estimated ytm, which is based on deals prices.
    For estimation uses Brent's algorithm to find a local minimum. 
    
    Parameters
    ------------
    df: Pandas DataFrame
        dataframe of bond's data
    coupons_cf: Pandas Dataframe
        Dataframe containing bonds' payment cash flows
    streak_data: Pandas Dataframe
        Dataframe containing bonds' payment calendar
    '''
    #saving old ytm
    df['ytm_kase'] = df['ytm'] / 100
    ##estimating new ytm
    df['ytm'] = np.array([loss_yield(df.iloc[i], coupons_cf, streak_data) 
                                          for i in range(df.shape[0])]) 
    return df

def filtering_ytm(df, min_yield=None, max_yield=None):
    '''
    Filter data based on ytm of deals
    
    Parameters
    -----------
    df: Pandas DataFrame
        DataFrame of bonds' data
    min_yield: float
    max_yield: float
    '''
    #throwing away transactions with too high or too low yield 
    if max_yield is not None:
        assert isinstance(max_yield, float)
        df = df.loc[df['ytm'] < max_yield] 
        print('filtering by max ytm', df.shape)
    if min_yield is not None:
        assert isinstance(min_yield, float)
        df = df.loc[df['ytm'] > min_yield]
        print('filtering by min ytm', df.shape)
    return df
    
### newton func fast and rough way to estimate ytm from price
### used for estimating ytm from proxy prices 
###--- used during Loss estimation
def newton_func(ytm, bond, ind, coupons_cf, streak_data, price_hat):
    delta = streak_data[ind].iloc[0]
    streak_ = np.arange(0, streak_data[ind].dropna().shape[0]) + delta * bond.annual_freq
    ytm_diff = ((coupons_cf[ind].dropna() / 
                 ((1 + ytm / bond.annual_freq) ** streak_)).sum() - price_hat)
    return ytm_diff

def newton_estimation(bond, price_hat, coupons_cf, streak_data, maxiter=50):
    '''
    Parameters
    ------------
    bond: Pandas Series
        
    price_hat: Pandas DataFrame
        
    coupons_cf: Pandas Dataframe
        Dataframe containing bonds' payment cash flows
    streak_data: Pandas Dataframe
        Dataframe containing bonds' payment calendar
        
    maxiter: int
        Maximum number of iteration allowed to optimizator to run
    
    Estimates ytm of deal by using secant method
    '''
    #approximating ytm for closer to reality x0 point
    if bond['span'] > 365:
        x0 = ytm_approximation.approx_ytm(bond, price_hat).henderson()
    else:
        x0 = ytm_approximation.approx_ytm(bond, price_hat).traditional()
    
    try:
        ytm = newton(func=newton_func, x0=x0, 
                     args=(bond, bond.name, coupons_cf, streak_data, price_hat), 
                     maxiter=maxiter, tol=10e-12)
    except Exception as e:
        print(e)
        print(f' index: {bond.name}\n price: {price_hat}\n x0: {x0}\n')
        ytm = np.random.rand(1)[0]
    return ytm