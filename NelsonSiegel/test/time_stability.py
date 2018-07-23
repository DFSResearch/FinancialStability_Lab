import os, sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath('..'))
try:
    import matplotlib.pyplot as plt
    from dataextraction import draw_several
    plt.style.use('ggplot')
    to_draw_graphs = True
except ImportError as e:
    to_draw_graphs = False
import CONFIG
from stability_assessment import stability_assession
from datapreparation.preparation_functions import processing_data, read_download_preprocessed_data
from estimation_ytm.estimation_ytm import new_ytm
from Loss import yield_Loss, price_Loss, naive_yield_Loss, filtering_ytm
from payments_calendar import download_calendar

plt.style.use('seaborn')
PATH = os.path.join('..', 'extracted_data')
datasets_path = os.path.join('..', 'datasets')
clean_data_path = os.path.join(datasets_path, 'clean_data.hdf')
calendar_data_path = os.path.join(datasets_path, 'coupons_data.hdf')
save_data = False


#####SET START AND END DATE FOR CURVES CONSTRUCTION:
START_DATE = '2018-01-01'
END_DATE = '2018-07-01'
####SET FREQUENCY OF DATES
FREQ = '1MS'


### Initialization
path = os.path.join(datasets_path, 'bonds.xls')
df = pd.read_excel(path, skiprows=2).rename(columns=CONFIG.NAME_MASK)

### data mungling
if save_data:
    clean_data = processing_data(df, 
                  mask_face_value=CONFIG.MASK_FACE_VALUE, mask_base_time=CONFIG.MASK_BASE_TIME,
                  needed_bonds=CONFIG.INSTRUMENTS, use_otc=CONFIG.USE_OTC, deal_market=CONFIG.DEAL_MARKET,
                  notes_in_otc=CONFIG.NOTES_IN_OTC, maturity_filter=CONFIG.MATURITY_FILTER, 
                  specific_deals=CONFIG.SPECIFIC_DEALS)
    
    #calendar payments data: saving and loading
    coupons_cf, streak_data = download_calendar(clean_data, hdf_coupons_path=calendar_data_path)
    #Estimating correct yield for data
    clean_data = (clean_data.pipe(new_ytm, coupons_cf, streak_data)
                            .pipe(filtering_ytm, max_yield=CONFIG.MAX_YIELD, 
                                  min_yield=CONFIG.MIN_YIELD))
    clean_data['bond_symb'] = clean_data.index.get_level_values(1).str.extract(r'([A-Z]+)', expand=False)
    clean_data = read_download_preprocessed_data(save_data, 
                                                 clean_data_path=clean_data_path, clean_data=clean_data)
else:
    clean_data = read_download_preprocessed_data(save_data, clean_data_path=clean_data_path,)
    #Coupon Data: saving and loading
    coupons_cf, streak_data = download_calendar(clean_data, hdf_coupons_path=calendar_data_path)
  
    
###Setting Loss arguments and optimization paramters 
#Initial guess vector(for optimization)
x0 = [0.85, -0.03, 0, 1.5]
#Parameters constraints
constr = ({'type':'ineq',
           'fun': lambda x: np.array(x[0] + x[1])})
#Longest matuiry year of deals in data
bounds =  ((0, 1), (None, None), (None, None), (CONFIG.TETA_MIN, CONFIG.TETA_MAX))
options = {'maxiter': 150, 'disp': True}

#defining loss -- Crucial --
#possible losses: ['yield_Loss', 'price_Loss', 'naive_yield_Loss']
loss = yield_Loss


print('start optimization\n')
###### OPTIMIZATION
stab_as = stability_assession(START_DATE, END_DATE, big_dataframe=clean_data,  
                    beta_init=x0, freq=FREQ)
stab_as.compute_curves(coupons_cf, streak_data, loss=loss, 
                       options=options, constraints=constr, solver_type='grid')    
print('end optimization\n')


### Showing results of work
total_area, av_area = stab_as.asses_area()
if to_draw_graphs:
    fig, ax = plt.subplots(figsize=(15, 10))
    linestyles = ['-', '--', '-.'] #you can easily extend this list
    #fixing random seed in order to make lines have same style every time
    np.random.seed(111)
    ls_array = np.random.choice(linestyles, size=stab_as.params.shape[0])
    range_dates = stab_as.params.index
    for i, date in enumerate(range_dates):
        ls = ls_array[i]
        X_mat = np.linspace(1 / 365, 30, 1000)
        draw_several(stab_as.params.loc[date], X_mat, ax=ax, 
                     label=date, linestyle=ls, shift=True)
        #getting data from object attritbute which was saved during optimization
        data = stab_as.data_different_dates[date]
        y_scatter = data['ytm'].as_matrix()
        x_scatter = data['span'].as_matrix() / 365 
        ax.scatter(x_scatter, y_scatter, s=30, alpha=0.5);
    ax.legend(loc='upper right', ncol=4)
    ax.set_title(('Av area between envelope lines is {av_area:.{n_digits}f} ' 
                  'bbp and total area is {total_area:.{n_digits}f}'
                  .format(av_area=av_area, total_area=total_area, n_digits=3)))
    plt.savefig(os.path.join(PATH, f'sev_curves_time_{START_DATE}-{END_DATE}_{loss.__name__}.png'), dpi=400)
else:
    print('Download matplotlib package to enable plotting graphs')