#!\bin\python
import math
import uncertainties as uc
import numpy as np
from IPython.display import display
import pandas as pd
import re
import pathlib
this_files_path = pathlib.Path(__file__).parent.absolute()

from fticr_toolkit import ideal_trap_physics as itp
from fticr_toolkit import ame

df = pd.read_csv(str(this_files_path)+'/NIST_ionization_energy_table.txt', delimiter='\t', header=0)
subset = df.copy()
#print(df.columns)

def get_element(Z):
    global subset
    subset = subset[subset.Z == Z]
    #display(subset)
    return subset

def get_charge_state(q):
    global subset
    subset = subset[subset["charge_state"] == q]
    return subset

def get_charge_states(qmin, qmax):
    global subset
    subset = subset[(subset["charge_state"] >= qmin) & (subset["charge_state"] <= qmax)]
    return subset

def get_total_binding(Z, qmin, qmax, manual_binding_error=None, show=False, binding_errsum=False):
    """Calcs total binding energy of charge states qmin (e.g. 0) to qmax (e.g. 40 if your actual used charge state is 41!)

    Args:
        Z ([type]): [description]
        qmin ([type]): [description]
        qmax ([type]): [description]
    """
    get_element(Z)
    get_charge_states(qmin, qmax)
    if show: display(subset)
    ie = subset.ie.sum()
    errors = subset.ie_err.to_numpy()
    errorsum = subset.ie_err.sum()

    if manual_binding_error is None:
        die = np.sqrt(np.sum(errors**2))
        if binding_errsum:
            die = errorsum
    else:
        die = np.sqrt(np.sum(np.asarray([manual_binding_error]*len(subset))**2))
    reset()
    return ie, die

def get_total_binding_ionstr(ion='187Re32+', show=False):
    A, el, q = itp.re_ionstr(ion)
    Z = int(ame.get_isotope(A, el).Z)
    return get_total_binding(Z, 0, q-1, show=show)

def reset():
    global subset
    subset = df.copy()

################################

if __name__ == "__main__":

    print(get_total_binding(55, 0, 11, None, True))
    reset()

    print(get_total_binding(55, 0, 22, None, True))
    reset()

    print(get_total_binding(55, 0, 33, None, True))
    reset()

    print(get_total_binding(55, 0, 44, None, True))
    reset()
    print(get_total_binding(70, 0, 14, None, True))
    reset()
    print(get_total_binding(70, 0, 28, None, True))
    reset()
    print(get_total_binding(70, 0, 42, None, True))
    reset()
    print(get_total_binding(70, 0, 43, None, True))
    reset()
"""
    print(get_total_binding(10, 0, 11, None, True))
    reset()
    print(get_total_binding(67, 0, 40, None, True))
    reset()
    print(get_total_binding(14, 0, 14, None, True))
    reset()
    exit()
    print(get_total_binding(1, 0, 1, None, True))
    reset()
    print(get_total_binding(54, 0, 24, None, True))
    reset()
    print(get_total_binding(54, 0, 25, None, True))
    reset()
    print(get_total_binding(54, 0, 25, 5, True))
"""