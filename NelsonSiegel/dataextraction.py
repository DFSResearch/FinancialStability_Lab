### Visualization and data extraction
import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    to_draw_graphs = True
    
except ImportError as e:
    to_draw_graphs = False
    
from weight_scheme import weight
from ns_func import Z, F, par_yield

spot_nelson, forward_nelson = Z, F

def draw_several(beta, theor_maturities, ax=None, shift=False, **kwargs):
    spot_rates = spot_nelson(theor_maturities, beta)
    if shift:
        spot_rates = (np.exp(spot_rates) - 1)
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(theor_maturities, spot_rates, **kwargs) 

def draw(beta, df, theor_maturities, title_date, 
         longest_maturity_year, weight_scheme=None,
         ax=None, draw_points=True, label=None, shift=False, **kwargs):
    '''
    Drawing spot curve based on beta
    
    Parameters
    ------------
    beta: list-like
        Should be length of 4. Nelson-Siegel's vector of parameters
    df: Pandas Dataframe
        Dataframe of bonds' data
    theor_maturities: list-like
        Vector of maturities
    title_date: str or datetime
        Settle date of the curve's construction
    longest_maturity_year: int
        Maximum tenor on which plot will be drawn
    weight_scheme: 'str', default: None
        Weights of deals
    ax: Matplotlib Axes, default: None
    draw_points: bool
        Whether to draw deals' ytm on plot or not
    label: str, default: None
    shift: bool, default: False
        If shift true transforms spot rate curve from discrete to continuous
    '''
    #defining zero curbe
    beta = beta.copy()
    spot_rates = spot_nelson(theor_maturities, beta)
    par = par_yield(theor_maturities, beta)
    if shift:
        spot_rates = (np.exp(spot_rates) - 1)
        par =  (np.exp(par) - 1)

    #setting axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 10))
    if label is None:
        beta[:3] *= 100
        label = f'{beta.round(2)}'
    ax.plot(theor_maturities, spot_rates*100, label=label, **kwargs)
    ax.plot(theor_maturities, par*100, label='Par curve', **kwargs)
    #draw scatterplot of deals or not?
    if draw_points:
        y_scatter = df['ytm'].values
        x_scatter = df['span'].values / 365 
        #size of points; depends on weight of transaction
        s = (weight(beta, df, weight_scheme=weight_scheme) / 
             weight(beta, df, weight_scheme=weight_scheme).sum())
        ax.scatter(x_scatter, y_scatter*100, s=s * 15 * 1e3,
                   facecolors='none', edgecolors='grey', alpha=0.9)
    #setting labels, ticks, legend and title
    ax.set_title(f'Кривые построенные на {pd.to_datetime(title_date):%d.%m.%Y}')
    ax.set_ylabel('%')
    ax.set_xlabel('Tenor in years')
    ax.set_xticks(np.arange(0, longest_maturity_year + 1, 1))
    ax.legend();

#create curves from beta parameter
def get_curves(beta, maturities=None, shift=False):
    if maturities is None:
        maturities = np.linspace(1e-6, 20, 1000)
    spot_rates     = spot_nelson(maturities, beta)
    forward_rates  = forward_nelson(maturities, beta)
    par_yields = par_yield(maturities, beta)
    
    if shift:
        spot_rates = (np.exp(spot_rates) - 1) 
        forward_rates = (np.exp(forward_rates) - 1) 
        par_yields = (np.exp(par_yields) - 1) 
        
    return spot_rates, forward_rates, par_yields

### Writting curves to excel
def curves_to_excel(path, beta, settle_date, maturities=None, shift=False):
    '''
    Writting beta based spot and forward curves in excel file
    '''
    if maturities is None:
        maturities = np.linspace(7/365, 30, 1000)
    spot_rates, forward_rates, par_yields = get_curves(beta, maturities=maturities, shift=shift)
    rates = pd.DataFrame([spot_rates, forward_rates]).T
    rates.index = pd.Series(maturities, name='years to maturity')
    rates.columns = ['spot', 'forward', 'par']
    rates.to_excel(path, sheet_name = f'{settle_date}')

    return rates

#Extracting calendar of payment to excel
def payment_calendar_to_excel(path, coupons_cf, streak_data):
    '''
    Writting calendar data to excel
    Parameters
    -----------
    path: str
        File path
    coupons_cf: Pandas Dataframe
        Dataframe containing bonds' payment cash flows
    streak_data: Pandas Dataframe
        Dataframe containing bonds' payment calendar
    '''
    writer = pd.ExcelWriter(path)
    coupons_cf.to_excel(writer, startrow=0, sheet_name='coupons_cf')
    streak_data.to_excel(writer, startrow=0, sheet_name='streak_data')
