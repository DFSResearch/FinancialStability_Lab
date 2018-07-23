###Bootstrapping
import pandas as pd
import numpy as np
import dask.multiprocessing, dask.threaded
from dask import compute, delayed
from dataextraction import draw_several
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from optimizator.slsqp_sim import iter_minimizer
import CONFIG


####bootstrap
def bootstrap(scatter_data, n_iter, settle_date, Loss, 
              coupons_cf, streak_data, num_workers=4, constr=None, 
              bounds=None, rho=None, weight_scheme=None, 
              bootstrap_betas=None, save_ntk=False):

    if bootstrap_betas is None:
        bootstrap_betas = pd.Series()
        def optimize_parallel(i):
            #fixing NTK bonds, so we make sure that they will be there forever
            if save_ntk is True:
                NTK_bonds = (scatter_data.index
                              .get_level_values(1)
                              .str.extract(r'([A-Z]+)') == 'NTK')
                fixed = scatter_data.loc[NTK_bonds]
                floating = scatter_data.loc[~NTK_bonds]
                floating = floating.sample(n=floating.shape[0], replace=True)
                sample_data = fixed.append(floating)
            else:
                sample_data = scatter_data.sample(n=scatter_data.shape[0], replace=True)
            loss_args = (scatter_data, coupons_cf, streak_data, rho, weight_scheme)
            x0 = [0.1, 0.01, 0.1, 1]
            res_ = iter_minimizer(Loss=Loss, beta_init=x0, 
                           loss_args=loss_args, constraints=constr, 
                           bounds=bounds, max_deal_span=30)

            beta_best = res_.x
            return beta_best
        #parallelized optimization    
        values = [delayed(optimize_parallel)(i) for i in range(n_iter)]
        bootstrap_betas = compute(*values, get=dask.multiprocessing.get, num_workers=num_workers)
    return bootstrap_betas

#drawing
def draw_bootstrap(df, bootstrap_betas, settle_date):
    max_deal_span = (df.span / 365).round().max()
    longest_maturity_year = max([max_deal_span, 20])
    X_maturities = np.linspace(0.001, longest_maturity_year, 1000)
    
    fig, ax = plt.subplots(figsize=(15, 10))
    try:
        for i in range(len(bootstrap_betas)):
            draw_several(bootstrap_betas[i], X_maturities, ax=ax)
    except Exception as e:
        print(e)
     #area between max and min
    area, av_area = area_boot(bootstrap_betas, maturities=X_maturities)
    ###########
    y_scatter = df['ytm'].as_matrix()
    x_scatter = df.span.as_matrix() / 365 
    ax.scatter(x_scatter, y_scatter, s=100)
    ax.axis('tight')
    ax.set_title(f'Area for envealope lines:' 
                 f' {round(area, 3)} and av.area is {av_area}' 
                 f' at {settle_date} for {df.shape[0]} points')
    ax.set_ylim(0, df.ytm.max() + 2)
        
def av_loss(data, bootstrap_betas, Loss, coupons_cf, streak_data, weight_scheme):
    av_loss_ser = pd.Series()
    data['bond_maturity_type'] = data.bond_maturity_type.astype('str')
    for mat_type in data.bond_maturity_type.unique():
        points = data[data.bond_maturity_type == mat_type]
        loss_array = []
        for beta_ in bootstrap_betas:
            res_ = Loss(beta_, df=points, coupons_cf=coupons_cf, 
                        streak_data=streak_data, rho=CONFIG.RHO, 
                        weight_scheme=weight_scheme)
            loss_array.append(res_)
        av_loss = np.sqrt(loss_array).mean()
        av_loss_ser[mat_type] = av_loss
    return av_loss_ser.sort_index()

### Bootstrapping over several dates
def time_bootstap(settle_dates, Loss, coupons_cf, streak_data, n_iter=10):
    bootstrap_df = pd.DataFrame()
    with PdfPages(f'bootstapping_sev_dates_{Loss.__name__}.pdf') as pdf:
        params = pd.Series()
        for settle_date in settle_dates:
            df = choosing_time_frame(settle_date, data, 
                          coupons_cf, streak_data, 
                          time_window=CONFIG.TIME_WINDOW, use_several_wind=True)
            if df.shape[0] < 10:
                print('too few points', settle_date)
                continue
            bootstrap_df[settle_date] = bootstrap(df, n_iter, settle_date, Loss, 
                                                  coupons_cf, streak_data, num_workers=8)
            pdf.savefig()
    return bootstrap_df