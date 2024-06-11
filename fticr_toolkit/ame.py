#!\bin\python
import math
import numpy as np
from IPython.display import display
import pandas as pd
import re, pathlib
from uncertainties import ufloat

this_files_path = pathlib.Path(__file__).parent.absolute()

from fticr_toolkit import ideal_trap_physics as itp
from fticr_toolkit import NISTie as ie
from scipy import constants

amu = constants.physical_constants["atomic mass unit-kilogram relationship"][0]
elc = constants.e
m_e = constants.physical_constants["electron mass in u"] # error is 3e-11
m_e = ufloat(m_e[0], m_e[2])
m_e_eV = constants.physical_constants["electron mass energy equivalent in MeV"]
m_e_eV = ufloat(m_e_eV[0], m_e_eV[2])*1e6
ueV = constants.physical_constants["electron volt-atomic mass unit relationship"]
ueV = ufloat(ueV[0], ueV[2])

#df = pd.read_csv(str(this_files_path)+'\mass16.csv')
df = pd.read_csv(str(this_files_path)+'/mass20.csv')
subset = df.copy()

def get_element(el=None, Z=None):
    global subset
    if el is not None:
        subset = subset.loc[subset["el"].isin(el)]
    if Z is not None:
        subset = subset.loc[subset["Z"].isin(Z)]
    return subset

def get_ion_mass(ion='187Re29+', binding=None, full=False, show=False, debug = False):
    """return ions mass and error in atomic mass units

    Args:
        ion (str): ion description string, e.g. '187Re29+'.

    Returns:
        tuple: (mass->float, err->float)
    """
    A, el, q = itp.re_ionstr(ion)
    iso = get_isotope(A, el)
    Z = int(iso.Z)
    if q>Z:
        return np.NaN, np.NaN
    
    mneutral = float(iso.umass)
    dmneutral = float(iso.umass_err)
    
    if binding is None:
        binding, dbinding = ie.get_total_binding(Z, 0, q-1, show = show)
    else:
        try:
            dbinding = binding.s
            binding = binding.n
        except:
            dbinding = 0

    mion = mneutral - q*m_e.n + binding*ueV.n
    dmion = np.sqrt(dmneutral**2 + (dbinding*ueV.n)**2 + (dbinding*ueV.s)**2 + (q*m_e.s)**2)

    if debug: 
        print('A, Z, el, q, mneutral, -q*m_e, binding*ueV, mion, dbinding*ueV, dmneutral')
        print(A, Z, el, q, mneutral, -q*m_e, binding*ueV, mion, dbinding*ueV, dmneutral)
    
    if full:
        mass_excess = float(iso.mass)*1e3
        dmass_excess = float(iso.mass_err)*1e3

        mion_eV = mneutral/ueV - q*m_e_eV + binding
        mion_amu = ufloat(mneutral, dmneutral) - q*m_e + binding*ueV
        dmion_eV = np.sqrt((dmneutral/ueV.n)**2 + (dbinding)**2)
        total_binding = ufloat(binding, dbinding)

        extra_data ={
            "A" : A,
            "Z" : Z,
            "q" : q,
            "element" : el,
            "qe_amu" : q*m_e,
            "mneutral" : ufloat(mneutral, dmneutral),
            "mneutral_eV" : ufloat(mneutral/ueV.n, dmneutral/ueV.n),
            "mass_excess" : ufloat(mass_excess, dmass_excess),
            "mion_eV" : ufloat(mion_eV.n, dmion_eV),
            "mion_amu" : mion_amu,
            "total_binding" : total_binding,
            "total_binding_amu" : total_binding*ueV.n,
        }
    
        #return mion, dmion, mneutral, dmneutral, q*m_e/amu, binding*ueV
        return mion, dmion, extra_data

    return mion, dmion

def get_iso_mass(A, el):
    iso = get_isotope(A, el)
    mneutral = float(iso.umass)
    dmneutral = float(iso.umass_err)
    return mneutral, dmneutral

def get_iso_mass_excess(A, el):
    iso = get_isotope(A, el)
    mneutral = float(iso.mass)*1e3
    dmneutral = float(iso.mass_err)*1e3
    return mneutral, dmneutral

def get_isobar(A):
    global subset
    subset = subset.loc[subset["A"].isin(A)]
    return subset

def get_isotope(A=1, el='H', Z=None, iso=None):
    reset()
    if iso is not None:
        match = re.match(r"([0-9]+)([a-z]+)", iso, re.I)
        if match:
            items = match.groups()
        else:
            return None
        A = items[0]
        el = items[1]
    if Z is not None:
        get_element(Z=[Z])
    else:
        get_element([el])
    get_isobar([A])
    return subset

def min_column(col="A", value=1):
    global subset
    if value == 0:
        return subset
    subset = subset.loc[subset[col] >= value]
    return subset

def max_column(col="A", value=500):
    global subset
    subset = subset.loc[subset[col] <= value]
    return subset

def min_uncertainty(relative):
    return min_column("precision", relative)

def max_uncertainty(relative):
    return max_column("precision", relative)

def min_dmass(eV):
    return min_column("mass_err", eV)

def max_dmass(eV):
    return max_column("mass_err", eV)

def min_A(value):
    return min_column("A", value)

def max_A(value):
    return max_column("A", value)

def min_abundancy(abund):
    return min_column("abundancy", abund)

def min_half_life(seconds):
    return min_column("halflife", seconds)

def reset():
    global subset
    subset = df.copy()

################################
### qm stuff

def calc_qm_ratios(permanent=False, SIunits=True):
    global df
    if permanent:
        reset()
    q_max = subset.Z.max()
    e_over_u = constants.elementary_charge / constants.u
    print('max charge state', q_max)
    for q in range(1, q_max+1):
        col = 'q'+str(q)
        if SIunits:
            subset[col] = q/subset['umass'] * e_over_u
        else:
            subset[col] = q/subset['umass']
        subset.loc[(subset.Z - q) < 0, col] = np.nan # masking impossible charge states with nan values
    if permanent:
        df = subset.copy()
    return subset

def get_qm_matrix():
    qm_columns = [col for col in subset if col.startswith('q')]
    return subset[qm_columns]

def guess_ion_by_qm(qm, nearest=5, nu_z=740000):
    """[summary]

    Args:
        qm (float): charge to mass ratio in SI units!
        nearest (int, optional): [description]. Defaults to 5.
    """
    qms = get_qm_matrix()
    #display(qms)
    qms -= qm
    #display(qms)
    qms = qms.abs()
    qms.dropna(axis = 0, how = 'all', inplace = True)

    #display(qms)
    ions = pd.DataFrame(columns=["ion", "Z", "abundancy", "halflife", "dqm", "dnu_z"])
    for i in range(nearest):
        qms_numpy = qms.to_numpy()

        thisqm = np.nanmin(qms_numpy)
        #print(thisqm)
        iindex, icolumn = np.where(qms_numpy == thisqm)
        #print(iindex, icolumn)
        index = qms.index.to_numpy()[iindex][0]
        column = qms.columns.to_numpy()[icolumn][0]
        #print(index, column)
        q = column[1:]
        ame_row = subset.loc[index]
        el, A, Z, abund, half = str(ame_row["el"]), int(ame_row["A"]), int(ame_row["Z"]), float(ame_row["abundancy"]), float(ame_row["halflife"])
        qms = qms.drop(index)
        real_qm = float(q)*elc/float(ame_row["umass"])/amu
        dnu_z = itp.dnu_z(nu_z_ref=nu_z, qm_ref=qm, qm_ioi=real_qm)

        #print(A, el, q, Z, dnu_z)

        info = pd.Series(data=[str(A)+el+str(q)+"+", Z, abund, half, real_qm-qm, dnu_z], index=ions.columns)
        ions = ions.append(info, ignore_index=True)

    #print(ions)
    return ions

def guess_ion_by_freq(nu_z, U, c2=-1.496e4, nearest=5):
    qm = (2*np.pi*nu_z)**2 / (2*U*c2)
    return guess_ion_by_qm(qm, nearest=nearest, nu_z=nu_z)


def guess_ion(nu_z, U, c2=-1.496e4, nearest=25, min_abund=0.0, A=[1, 400], precision=[1,1e-16], min_halflife=1e-10, elements=[]):
    reset()
    if elements:
        get_element(elements)
    min_abundancy(min_abund)
    min_half_life(min_halflife)
    min_A(A[0])
    max_A(A[1])
    min_uncertainty(precision[1])
    max_uncertainty(precision[0])
    #display(subset)

    ions = guess_ion_by_freq(nu_z, U, c2, nearest=nearest)

    reset()
    return ions

################################
### PLOTTING

import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors
from matplotlib.colors import SymLogNorm, LogNorm
from matplotlib import ticker

def nuclide_chart(column_name, to_nan=[0]):

    Z = subset['Z']
    N = subset['N']

    zmin = Z.min()
    nmin = N.min()

    image = np.full( (Z.max()-Z.min()+1, N.max()-N.min()+1) , np.nan)

    # building up from bottom to top
    for z in Z.unique():
        lineset = subset.loc[Z == z]

        lineN = lineset["N"].to_numpy()
        for n in lineN:
            #print(z,n)
            try:
                image[z-zmin,n-nmin] = lineset.loc[lineset['N'] == n][column_name]
            except:
                pass

    for el in to_nan:
        image[image==el] = np.nan

    #print(image)
    return image

def plot_logheatmap(image, linthreash, maxvalue, minvalue, title="AME masses heatmap"):

    masked_array = np.ma.array(image, mask=np.isnan(image))
    #masked_array = np.ma.array(masked_array, mask=(masked_array>linthreash))
    cmap = mcm.inferno
    cmap.set_bad('white',1.)
    cmap.set_over('grey')

    plt.pcolormesh(masked_array, cmap=cmap, vmax=linthreash, norm=LogNorm(vmin=minvalue, vmax=maxvalue))
    #plt.pcolormesh(masked_array, cmap=cmap, norm=LogNorm(vmin=minvalue, vmax=maxvalue))

    p = plt.colorbar()
    p.locator=ticker.LogLocator(base=10)
    p.update_ticks()

    plt.xlabel("N")
    plt.ylabel("Z")
    plt.title(title)
    plt.grid()

def plot_logheatmap2(image, linthreash, maxvalue, minvalue, title="AME masses heatmap"):

    masked_array = np.ma.array(image, mask=np.isnan(image))
    cmap = mcm.hot
    cmap.set_bad('white',1.)

    plt.pcolormesh(image, cmap=cmap, norm=SymLogNorm(linthresh=linthreash, vmin=minvalue, vmax=maxvalue))

    p = plt.colorbar()
    p.locator=ticker.LogLocator(base=10)
    p.update_ticks()

    plt.xlabel("N")
    plt.ylabel("Z")
    plt.title(title)
    plt.grid()

def ame_precision_nuclide_chart(show=False, save=False):
    reset()

    image = nuclide_chart("precision")

    minv = ( math.floor(np.log10(np.nanmin(image))) )
    maxv = ( math.ceil( np.log10(np.nanmax(image))) ) - 1
    print(minv, maxv)

    plot_logheatmap(image, 1, 10**maxv, 10**minv, "AME 2020 $\Delta m /m$")
    if save:
        plt.savefig("precision_chart.png")
    if show:
        plt.show()

def halflife_nuclide_chart(show=False, save=False):
    reset()

    image = nuclide_chart("halflife")

    minv = -6
    maxv = 8
    print(minv, maxv)

    plot_logheatmap(image, 10**maxv, 10**maxv, 10**minv, "halflife in seconds")

    if save:
        plt.savefig("halflife_chart.png")
    if show:
        plt.show()

################################

if __name__ == "__main__":
    #pd.options.display.precision = 13

    #ame_precision_nuclide_chart(show=True, save=False)

    #halflife_nuclide_chart(show=True, save=False)
    reset() 
    sub = get_element(['Ca'])
    sub = max_uncertainty(1e-5)
    sub = sub[sub.A%2 == 0]
    display(sub)

    """
    reset()
    sub = max_dmass(0.005)
    display(sub)
    
    #reset()
    #sub = min_half_life(1000)
    #sub = get_element(["Ho"])
    #display(sub)

    reset()
    #calc_qm_ratios(permanent=True)
    calc_qm_ratios(permanent=True, SIunits=True)

    nuz = 740288.79 #-120
    U0 = -30.679
    c2 = -1.48861e4

    start = time.perf_counter()

    test_qm = itp.qm(nuz, U0, c2) #* elc / amu
    print('nuz', nuz, 'u0', U0, 'qm', test_qm)

    min_abundancy(0.0001)
    min_A(1)
    max_A(400)

    ions = guess_ion_by_qm(test_qm, nearest=5, nu_z=nuz)
    print(ions)
    ions = guess_ion_by_freq(nuz, U0, c2, nearest=5)
    print(ions)

    end = time.perf_counter()
    print('time for guessing', end-start)

    start = time.perf_counter()
    ions = guess_ion(nuz, U0, c2, nearest=5, min_abund=0.0001)
    print(ions)
    end = time.perf_counter()
    print('time for guessing', end-start)
 
    start = time.perf_counter()
    ions = guess_ion(nuz, U0, c2, nearest=15, min_abund=0, min_halflife=1e6, elements=["Dy", "Ho", "Bi"])
    print(ions)
    end = time.perf_counter()
    print('time for guessing', end-start)

    """

    reset()
    print(get_ion_mass('28Si13+', full=True, show=True, debug=True))
    sub = max_uncertainty(3.e-11)
    display(sub)
    """

    reset()
    sub = get_element(['Xe'])
    sub = sub[sub.A%2 == 0]
    display(sub[['A', 'el', 'Z', 'umass', 'umass_err', 'precision']] )
    print(sub.precision.min())
    """

    #sub = max_uncertainty(2e-9)
    #display(sub[['A', 'el', 'Z', 'umass', 'umass_err', 'precision', 'abundancy', 'halflife']] )
    reset() 
    sub = get_element(['Xe'])
    #sub = get_element(['Rn'])
    #sub = max_uncertainty(1e-9)
    #sub = sub[sub.A%2 == 0]
    #display(sub)
    display(sub[['A', 'el', 'Z', 'umass', 'umass_err', 'atomic_mass', 'atomic_mass_err', 'precision', 'abundancy', 'halflife']] )

    reset() 
    sub = get_element(['U'])
    #sub = get_element(['Rn'])
    #sub = max_uncertainty(1e-9)
    #sub = sub[sub.A%2 == 0]
    #display(sub)
    display(sub[['A', 'el', 'Z', 'umass', 'umass_err', 'atomic_mass', 'atomic_mass_err', 'precision', 'abundancy', 'halflife']] )


    reset()
    sub = get_element(['Yb'])
    #sub = get_element(['Rn'])
    #sub = max_uncertainty(1e-9)
    sub = sub[sub.A%2 == 0]
    #display(sub)
    display(sub[['A', 'el', 'Z', 'umass', 'umass_err', 'atomic_mass', 'atomic_mass_err', 'precision', 'abundancy', 'halflife']] )
    print(get_iso_mass(172, "Yb"))
    #print(sub.precision.min())

    reset()
    sub = get_element(['Ne'])
    sub = sub[sub.A%2 == 0]
    display(sub[['A', 'el', 'Z', 'umass', 'umass_err', 'precision']] )
    print(sub.precision.min())


    reset()
    sub = get_element(['Cf'])
    display(sub[['A', 'el', 'Z', 'umass', 'umass_err', 'atomic_mass', 'atomic_mass_err', 'precision', 'abundancy', 'halflife']] )
    print(sub.precision.min())


    
    reset()
    sub = get_element(['Rb'])
    sub = sub[sub.A == 87]
    display(sub[['A', 'el', 'Z', 'umass', 'umass_err', 'atomic_mass', 'atomic_mass_err', 'precision', 'abundancy', 'halflife']] )
    print(sub.precision.min())

    reset()
    sub = get_element(['Cs'])
    sub = sub[sub.A == 133]
    display(sub[['A', 'el', 'Z', 'umass', 'umass_err', 'atomic_mass', 'atomic_mass_err', 'precision', 'abundancy', 'halflife']] )
    print(sub.precision.min())

    reset()
    sub = max_uncertainty(3.e-11)
    display(sub)