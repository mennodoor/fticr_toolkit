import numpy as np
import pandas as pd


def calc_nu_c(nu_p, nu_z, nu_m):
    pd.options.mode.chained_assignment = None  # default='warn'

    return np.sqrt(nu_p**2 + nu_z**2 + nu_m**2)

def calc_nu_c_error(nu_p, nu_z, nu_m, dnu_p, dnu_z, dnu_m):
    pd.options.mode.chained_assignment = None  # default='warn'

    nu_c_set = calc_nu_c(nu_p, nu_z, nu_m)
    dnu_c_set = np.sqrt((nu_p/nu_c_set*dnu_p)**2 + (nu_z/nu_c_set*dnu_z)**2 + (nu_m/nu_c_set*dnu_m)**2)
    return nu_c_set, dnu_c_set

def calc_nu_c_sb(nu_p, nu_m):
    pd.options.mode.chained_assignment = None  # default='warn'

    return (nu_p + nu_m)

def calc_nu_c_sb_error(nu_p, nu_m, dnu_p, dnu_m):
    pd.options.mode.chained_assignment = None  # default='warn'

    nu_c_set = calc_nu_c_sb(nu_p, nu_m)
    dnu_c_set = np.sqrt( dnu_p**2 + dnu_m**2 )
    return nu_c_set, dnu_c_set

