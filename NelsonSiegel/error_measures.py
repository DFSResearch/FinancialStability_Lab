import pandas as pd
import numpy as np
from ns_func import Z, D
from estimation_ytm.estimation_ytm import newton_estimation

#defining base mae
mean_absolute_error = lambda y_true, y_pred: np.average(np.abs(y_pred - y_true), axis=0)

#area between envealop lines
def area_boot(betas_array, tenors):
    spots_boot = pd.DataFrame()
    for i, beta_ in enumerate(betas_array):
        spots_boot[i] = Z(tenors, beta_)
    diff_curve = spots_boot.max(1) - spots_boot.min(1)
    area = np.trapz(diff_curve * 100, tenors)
    av_area = round(area / max(tenors), 3)
    return area, av_area

def MAE_YTM(beta, data, coupons_cf, streak_data):
    '''
    Mean absolute error between true ytm and predicted ytm
    '''
    ind = data.index
    Price = (coupons_cf[ind] * D(streak_data[ind], beta)).sum()
    ytm_hat = np.array([np.exp(newton_estimation(data.iloc[i], Price[i], 
                            coupons_cf, streak_data, maxiter=200)) - 1
                        for i in range(data.shape[0])])
    return mean_absolute_error(data['ytm'], ytm_hat)