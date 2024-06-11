#
#
# # This is a small library for systematic shifts in a Penning trap.
# 
# Author: Menno Door
# 
# All function give the "positive" shift as a result, meaning that the input data should correspond
# to the unperturbed (ideal trap) frequencies / ratio and the returned shift / corrected value corresponds
# to the measured frequencies:
#   omega_meas = omega_unperturbed + delta_omega
# So if you want to correct the measured data to unpertured frequencies you
# have to substract the shift results from your data:
#   omega_unperturbed = omega_meas - delta_omega
#  NOTE: You should check it yourself, but because the shifts
# are mostly small relative to the original frequencies, it often does not change the shift significantly 
# when you put in the measured frequencies instead of the ideal trap frequencies. Though you should not mix
# ideal/real frequencies, to keep the ratios/differences correct.
#
# All functions should also have docstrings explaining the input and output parameters and 
# have references where the actual formular was extracted from (a lot of these functions
# can be derived by others, anyway, when I was able to find a nice reference in some thesis
# I gave it, so people can also check for more detailed information about this or that effect.)
#
# All functions are build (hopefully) to allow for arrays in all reasonable arguments:
# e.g. give an array of radii or an array of voltages to calculated effects on multiple
# input parameters directly.
#
# I make intensive use of the uncertainties package, which does error propagation for you without
# having to give the explicit formulars for that. It can also give you correlations of calculated
# values and give you the individual impacts of input value uncertainties on the output value.
#
# NOTE: We use the unit-attached version of Ci parameters so they include the characteristic
# trap length d0, basically Ci_here = Ci_Gabrielse / (2*d0**i). Thats why some of the formulars
# might look wrong on first glance (also compared to the given references). Keep that in mind
# and adjust your values of Ci if neccessary.
#
# The following effects should be included:
# - relativistic shift
# - image charge shift (approximated function)
# - image current shift
# - Tilt and elipsisity
# - C4 / C6 shift
# - B2 shifts
# - B1 / axial displacement shift
# - C1 / C3
# - other combined shifts:
#    - B1/C1
#    - B2/C3
# 
#
# NOTE: Units for function parameters:
#
# every function should work with the following units for the input parameters:
# omega     : angular frequency (nu*2*pi)
# nu        : Hz
# radii     : m
# mass      : atomic mass units
# mass diff : eV
# charge    : integers (multiples of elementary charge e)
# temperatur: Kelvin
# energy    : eV
# B field   : T
# U         : V
# Ci coeff  : 1/m**i # e.g. design c2, c4, c6 at Pentatrap are -1.496(7)e4 [1/m**2], 0(4)e6 [1/m**4], 0(2)e11 [1/m**6]
# di coeff  : 1/m**i
# Bi coeff  : T/m**i # in the region of B1,B2 of 1e-3 [T/m], 50e-3 [T/m**2]
# angle     : radians
# inductivi.: Henry
# velocity  : m/s

import re
import numpy as np
from uncertainties import ufloat
from uncertainties.umath import *
from uncertainties import unumpy as unp
from uncertainties import correlation_matrix
from matplotlib import pyplot as plt
import matplotlib
import scipy.constants as cont

# needed constants
c = cont.c # exact
elq = cont.physical_constants["elementary charge"][0] # exact :)
m_e = cont.physical_constants["electron mass in u"] # error is 3e-11
m_e = ufloat(m_e[0], m_e[2])
epsilon0 = cont.physical_constants["vacuum electric permittivity"] # error is 1.5e-10
epsilon0 = ufloat(epsilon0[0], epsilon0[2])
eVboltzmann = cont.physical_constants["Boltzmann constant in eV/K"][0] # exact
bohr_magneton_eV = cont.physical_constants["Bohr magneton in eV/T"] # error is 3e-10
bohr_magneton_eV = ufloat(bohr_magneton_eV[0], bohr_magneton_eV[2])

# naming of conversion variables is based on unit1unit2 = unit2/unit1, e.g. ukg is kg/u (so basically ukg converts u to kg)
ukg = cont.physical_constants["atomic mass unit-kilogram relationship"][0] # the uncertainty of this conversions is bad (3e-10)
ueV = cont.physical_constants["atomic mass unit-electron volt relationship"][0] # the uncertainty of this conversions is bad (3e-10)
eVkg = cont.physical_constants["electron volt-kilogram relationship"][0] # exact
JeV = cont.physical_constants["joule-electron volt relationship"][0] # exact
pi2 = np.pi*2 # just handy to have

from fticr_toolkit import ame
from fticr_toolkit import NISTie
from fticr_toolkit import ideal_trap_physics as itp


################################################################################################################################################
### general helpers:
def uval(var, dvar=None, tag=None):
    """_summary_

    Args:
        var (_type_): some value or array or ufloat or uarray
        dvar (_type_, optional):
        tag (_type_, optional):

    Returns:
        _type_: original
    """
    try:
        _ = iter(var)
    except TypeError:
        # var is not iterable
        if dvar is None and isinstance(var, float) or isinstance(var, int) or isinstance(var, ufloat):
            return var
        elif dvar is not None and isinstance(var, float) or isinstance(var, int):
            return ufloat(var, dvar)
        else:
            raise TypeError(var)



def ratio_shift(Rmeas, omegacA, omegacB, domegacA=None, domegacB=None):
    """_summary_

    Args:
        Rmeas (ufloat): _description_
        omegacA (ufloat): _description_
        omegacB (ufloat): _description_

    Returns:
        _type_: _description_
    """


################################################################################################################################################
### relativistic shift

def rel_mass(mass, v):
    """calculate relativistic mass increase

    Args:
        mass (float): mass in arbitrary unit
        v (float): speed in m/s

    Returns:
        float: mass in same unit as input
    """
    return mass/np.sqrt(1-(v/c)**2)

'''
def domegaz_rel(rohp, omegap, omegaz):
    return 1/4/c**2 * omegaz*(omegap)**2 * rohp**2

def domegap_rel(rohp, omegap):
    return -1/2/c**2 * omegap**3 * rohp**2
'''

def domegap_rel(rhop, zamp, rhom, omegap, omegaz, omegam):
    """
    Formular from Ketter PhD (4.134)

    Args:
        rhop (float): amplitude in m
        zamp (float): amplitude in m
        rhom (float): amplitude in m
        omegap (float): angular trap cyclotron frequency 
        omegaz (float): angular trap magnetron frequency 
        omegam (float): angular trap axial frequency 

    Returns:
        float: abolute shift in angular trap cyclotron frequency due to relativistic mass increase
    """
    # meanv2 = (omegap*rhop)**2 + (omegam*rhom)**2 + (omegaz*zamp)**2 / 2 # "simple" 4.145 & 4.146
    meanv2 = (omegap*rhop)**2 + 2*(omegam*rhom)**2 + (omegaz*zamp)**2 / 2
    return - omegap**2 / (omegap - omegam) * meanv2 / 2 / c**2
    #return -1 * roh_p**2 * (omegap**2 - omegap*omegam) * omegac / 2 / c**2 

def domegaz_rel(rhop, zamp, rhom, omegap, omegaz, omegam):
    """
    Formular from Ketter PhD (4.114)

    Args:
        rhop (float): amplitude in m
        zamp (float): amplitude in m
        rhom (float): amplitude in m
        omegap (float): angular trap cyclotron frequency 
        omegaz (float): angular trap magnetron frequency 
        omegam (float): angular trap axial frequency 

    Returns:
        float: abolute shift in axial frequency due to relativistic mass increase
    """
    meanv2 = (omegap*rhop)**2 + (omegam*rhom)**2 + 3/4*(omegaz*zamp)**2 
    return - omegaz * meanv2 / 4 / c**2

def domegam_rel(rhop, zamp, rhom, omegap, omegaz, omegam):
    """
    Formular from Ketter PhD (4.145 and 4.146)

    Args:
        rhop (float): amplitude in m
        zamp (float): amplitude in m
        rhom (float): amplitude in m
        omegap (float): angular trap cyclotron frequency 
        omegaz (float): angular trap magnetron frequency 
        omegam (float): angular trap axial frequency 

    Returns:
        float: abolute shift in angular magnetron frequency due to relativistic mass increase
    """
    # meanv2 = (omegap*rhop)**2 + (omegam*rhom)**2 + (omegaz*zamp)**2 / 2 # "simple" 4.145 & 4.146
    meanv2 = 2*(omegap*rhop)**2 + (omegam*rhom)**2 + (omegaz*zamp)**2 / 2
    return omegam**2 / (omegap -  omegam) * meanv2 / 2 / c**2

def domegac_rel(roh_p, omegac, omegam, omegap):
    return -1 * roh_p**2 * (omegap**2 - omegap*omegam) * omegac / c**2 / 2

def R_rel(R, omegapA, omegapB, rohpA, rohpB):
    """Relativistic shift

    Args:
        R_measured (float): measured ratio
        omegapA (float): cyclotron frequency of ratio numerator, R = omegaB / omegaA
        omegapB (float): cyclotron frequency of ratio denominator, R = omegaB / omegaA
        rohpA (float): radius during measurement, ion numerator, R = omegaB / omegaA
        rohpB (float): radius during measurement, ion numerator, R = omegaB / omegaA

    Returns:
        float: corrected ratio
    """
    factor = unp.sqrt( (c**2 - (omegapB*rohpB)**2) / (c**2 - (omegapA*rohpA)**2) )
    print(factor)
    return R/factor 

def dR_rel(R, omegapA, omegapB, rohpA, rohpB):
    """Relativistic shift

    Args:
        R_measured (float): measured ratio
        omegapA (float): cyclotron frequency of ratio numerator, R = omegaB / omegaA
        omegapB (float): cyclotron frequency of ratio denominator, R = omegaB / omegaA
        rohpA (float): radius during measurement, ion numerator, R = omegaB / omegaA
        rohpB (float): radius during measurement, ion numerator, R = omegaB / omegaA

    Returns:
        float: corrected ratio
    """
    factor = unp.sqrt( (c**2 - (omegapB*rohpB)**2) / (c**2 - (omegapA*rohpA)**2) )
    return R/factor - R

def dR_rel2(R, omegapA, omegapB, rhopA, rhopB):
    """Relativistic shift

    Args:
        R_measured (float): measured ratio
        omegapA (float): cyclotron frequency of ratio numerator, R = omegaB / omegaA
        omegapB (float): cyclotron frequency of ratio denominator, R = omegaB / omegaA
        rohpA (float): radius during measurement, ion numerator, R = omegaB / omegaA
        rohpB (float): radius during measurement, ion numerator, R = omegaB / omegaA

    Returns:
        float: corrected ratio
    """
    dompompA = -(omegapA*rhopA)**2 /2 /c**2
    dompompB = -(omegapB*rhopB)**2 /2 /c**2

    return R*(dompompB - dompompA)


def dR_rel3(R, omegapA, omegapB, rhopA, rhopB):
    """Relativistic shift

    Args:
        R_measured (float): measured ratio
        omegapA (float): cyclotron frequency of ratio numerator, R = omegaB / omegaA
        omegapB (float): cyclotron frequency of ratio denominator, R = omegaB / omegaA
        rohpA (float): radius during measurement, ion numerator, R = omegaB / omegaA
        rohpB (float): radius during measurement, ion numerator, R = omegaB / omegaA

    Returns:
        float: corrected ratio
    """
    dompompA = -(omegapA*rhopA)**2 /2 /c**2
    dompompB = -(omegapB*rhopB)**2 /2 /c**2

    return -R*(1-(1-dompompB)/(1-dompompA))

def dR_rel4(R, omegapA, omegapB, rhopA, rhopB):
    """Relativistic shift

    Args:
        R_measured (float): measured ratio
        omegapA (float): cyclotron frequency of ratio denominator, R = omegaB / omegaA
        omegapB (float): cyclotron frequency of ratio numerator, R = omegaB / omegaA
        rohpA (float): radius during measurement, ion denominator, R = omegaB / omegaA
        rohpB (float): radius during measurement, ion numerator, R = omegaB / omegaA

    Returns:
        float: corrected ratio
    """
    dompompA = -(omegapA*rhopA)**2 /2 /c**2
    dompompB = -(omegapB*rhopB)**2 /2 /c**2

    return R*(dompompB-dompompA)


### E X T R A 

def domegac_rel_E(Ep, omegac, restmass, eVmass=False):
    """
    thesis Roux 3.30 / 3.31

    Ep: energy in eV
    omegac : (angular) cyclotron frequency
    restmass : rest mass in u

        omegap ~~ omegac = q*B/m_0 * np.sqrt( 1 - (v_+/cont.c)**2 )
                        ~~ omegac_0 ( 1 - 1/2 (v_+/cont.c)**2 )
    <=> domegac / omegac = - 1/2 * (v_+/cont.c)**2
                        = -1/2 * m_0 * v_+**2 / (m_0 * cont.c)**2
                         ~~ - E_kinp / m_0 / cont.c**2
                         ~~ - Ep / m0 / cont.c**2
    """
    if eVmass:
        return - Ep/JeV / restmass*eVkg / cont.c**2 * omegac # less error due to exact conversion of eV to kg
    else:
        return - Ep/JeV / restmass*ukg / cont.c**2 * omegac # bigger error possible due to ukg conversion

################################################################################################################################################


################################################################################################################################################
### image charge shift

def domegap_ICS(q, mass, trap_radius, omegac, relative_error = 0.05):
    """
    approximation in thesis Roux 5.14
    """
    dom = -(q*elq)**2 / ( 4*np.pi*epsilon0 * mass*ukg * trap_radius**3 * omegac )
    if relative_error is not None:
        nominals = unp.nominal_values(dom)
        dom = ufloat(nominals, np.absolute(nominals) * relative_error)
    return dom

def domegam_ICS(q, mass, trap_radius, omegac, relative_error = 0.05):
    """
    approximation in thesis Roux 5.14
    """
    dom = (q*elq)**2 / ( 4*np.pi*epsilon0 * (mass*ukg) * trap_radius**3 * omegac )
    if relative_error is not None:
        nominals = unp.nominal_values(dom)
        dom = ufloat(nominals, np.absolute(nominals) * relative_error)
    return dom

def domegac_ICS(q, mass, trap_radius, omegac, omegap, omegam, relative_error = 0.05):
    """
    Thesis Roux 5.15
    """
    dom = (omegam/omegac - omegap/omegac) * (q*elq)**2 / ( 4*np.pi*epsilon0 * mass*ukg * trap_radius**3 * omegac )
    if relative_error is not None:
        nominals = unp.nominal_values(dom)
        dom = ufloat(nominals, np.absolute(nominals) * relative_error)
    return dom

def dR_ICS(R, B0, trap_radius, delta_mass, relative_error = 0.05):
    """
    R is the measured ratio, in nuB over nuA, so qB/qA*mA/mB, then delta_mass is mB - mA (!)
    TODO: please could somebody cross check this? ;) ESPICIALLY THE SIGN!!!
    B0 in T,
    trap_radius in mm
    delta_mass in eV, mA - mB

    e.g. 208Pb41+ (A) / 132Xe21+ (B):
    R ~ 1.000125219425
    mA - mB ~ 76 (NOTE: sign!!!)
    -> so dR_ICS is also positive!
    """
    alpha = 4*np.pi*epsilon0*B0**2*trap_radius**3
    dR = R*delta_mass*eVkg/alpha
    print("*** ICS ***")
    print(delta_mass/ueV)
    print(dR)
    if relative_error is not None:
        nominals = unp.nominal_values(dR)
        print('ICS nominals', nominals, type(nominals))
        try:
            nominals = float(nominals)
            dR = ufloat(nominals, np.absolute(nominals) * relative_error)
        except:
            dR = unp.uarray([nominals, np.absolute(nominals) * relative_error])
        print('ICS nominals', nominals, type(nominals))

        #if not isinstance(nominals, float):
        #    dR = unp.uarray([nominals, np.absolute(nominals) * relative_error])
        #else:
        #    dR = ufloat(nominals, np.absolute(nominals) * relative_error)
    return dR

"""
trapR = np.arange(4.9, 5.1, 0.01)*1e-3
ics = dR_ICS(1.01, 7, trapR, 8*ueV, 0.05)
icsv = unp.nominal_values(ics)
icse = unp.std_devs(ics)
plt.errorbar(trapR, icsv, icse)
plt.axvline(4.95e-3)
plt.axvline(5.05e-3)
plt.axvline(5.005e-3)
plt.axvline(4.995e-3)
plt.show()
"""
### E X T R A 

def dR_precise_ICS(R, B0, trap_radius, eV_delta_mass, omegac_ioi, omegap_ioi, omegam_ioi, eV_mass_ref, kg_mass_ref, omegac_ref, omegap_ref, omegam_ref):
    """
    in the case of non perfect q/m doublets maybe the frequency prefactors in the iC ratio shift don't actually cancle out. To check this use this dR_ICS_precise
    most likely unnecessary except for large ICS difference (large mass difference) and imperfect q/m doublet. Example: 132Xe26+ vs 208Pb41+ difference on 10^-13 level
    """
    return (omegam_ioi/omegac_ioi-omegap_ioi/omegac_ioi)/( 4*np.pi*epsilon0*B0**2*trap_radius**3/kg_mass_ref +  (omegam_ref/omegac_ref-omegap_ref/omegac_ref)) * R * eV_delta_mass/eV_mass_ref


################################################################################################################################################
### Tilt and elipsodity

"""
master thesis Door (cited obviously from somewhere) 2.30

9/4 * theta**2 - 1/2 * epsilon**2 ~~ (omegam + omegap - omegac_invariance) / omegam
"""
#def domegaz_tilt(theta, omegaz):
#    """
#    Thesis Ulmer 6.8
#    """
#    return omegaz * (unp.sqrt( 1 - 3/4*theta**2 ))

def domegap_tilt_ellip(omegap, omegaz, omegam, epsilon=0, theta=0, phi=0):
    """
    Thesis S. Rau (2.20a)
    """
    return 3/4*omegaz * theta**2 * ( 1 + 1/3*epsilon*np.cos(2*phi) ) + omegam**2/2/omegap*epsilon**2

def domegaz_tilt_ellip(omegaz, epsilon=0, theta=0, phi=0):
    """
    Thesis S. Rau (2.20b)
    """
    return -3/4*omegaz * theta**2 * ( 1 + 1/3*epsilon*np.cos(2*phi) )

def domegam_tilt_ellip(omegap, omegaz, omegam, epsilon=0, theta=0, phi=0):
    """
    Thesis S. Rau (2.20c)
    """
    return 3/4*omegam * theta**2 * ( 1 + 1/3*epsilon*np.cos(2*phi) ) + omegam/2*epsilon**2

#def dR_tilt_ellip()

def angle(displacement_at_trap, L_below_trap):
    pass

def ddomegaz_tilt(theta, dtheta, omegaz):
    pass

################################################################################################################################################


################################################################################################################################################
### Trap potential - even Ci

### d2 shift
def domegaz_d2(omegaz, dTR, d2, c2):
    return d2/2/c2*dTR*omegaz

### C4 shift by amplitudes
def domegac_c4(rohp, rohm, omegap, omegam, c4, c2=-1.496e4):
    """
    WARNING! this is valid for the sideband relation, NOT for invariance
    J. Ketter PhD thesis / pertubation theory paper 
    """
    print('radii', rohp, rohm)
    rohp =0

    print('C4 / C2', (c4/c2).n, (c4/c2).s )
    print('C4 a', -3/2 * c4/c2 )
    print('C4 b',  omegam*omegap/(omegap-omegam) )
    print('C4 c',  (rohp**2 - rohm**2) )
    return -3/2 * c4/c2 * omegam*omegap/(omegap-omegam) * (rohp**2 - rohm**2)

def domegap_c4(zamp, rohp, rohm, omegap, omegam, c4, c2=-1.496e4):
    """
    Thesis Doerr 2.26
    default c2 from thesis Roux table 5.2, c4 is zero in his design, measured value definitly needed!

    all units are mm and angle frequency
    """
    return -3/2 * c4/c2 * omegam*omegap/(omegap-omegam) * (2*zamp**2 - rohp**2 - 2*rohm**2)

def domegam_c4(zamp, rohp, rohm, omegap, omegam, c4, c2=-1.496e4):
    """
    Thesis Doerr 2.27
    default c2 from thesis Roux table 5.2, c4 is needed
    """
    return 3/2 * c4/c2 * omegap*omegam/(omegap-omegam) * (2*zamp**2 - rohm**2 - 2*rohp**2)

def domegaz_c4(zamp, rohp, rohm, omegaz, c4, c2=-1.496e4):
    """
    Thesis Doerr 2.28
    default c2 from thesis Roux table 5.2, c4 is needed
    """
    #print('radii**2', zamp**2 - 2*rohp**2 - 2*rohm**2)
    #print('omegaz', omegaz)
    #print('c4/c2', c4/c2)
    return 3/4 * c4/c2 * (zamp**2 - 2*rohp**2 - 2*rohm**2) * omegaz

# C4 shift by energies
def domegapzm_c4(Ez, Ep, Em, omegaz, omegap, omegam, q, U, c4, c2=-1.496e4):
    """
    Thesis Roux 3.24
    default c2 from thesis Roux table 5.2, c4 is needed
    """
    omega_ratio = omegaz**2/omegap**2

    prefac = 3/(q*elq)/U*c4/c2**2 # TODO: is the c2**2 correct?

    delta_omegap = omegap * prefac * ( 1/4*omega_ratio**2*Ep - 1/2*omega_ratio*Ez - omega_ratio*Em)
    delta_omegaz = omegaz * prefac * (-1/2*omega_ratio*Ep - 1/4*Ez - Em)
    delta_omegam = omegam * prefac * ( -omega_ratio*Ep - Ez - Em)

    return delta_omegap, delta_omegaz, delta_omegam

def dR_c4(Rmeas, rhopA, rhomA, omegacA, omegapA, omegamA, 
            rhopB, rhomB, omegacB, omegapB, omegamB, c4, c2=-1.496e4):
    Cr = 3*c4/2/c2
    A = omegapA*omegamA/(omegapA-omegamA) * (rhopA**2 - rhomA**2) / omegacA
    B = omegapB*omegamB/(omegapB-omegamB) * (rhopB**2 - rhomB**2) / omegacB
    return Rmeas*((1+Cr*A)/(1+Cr*B) - 1)

### C6 shift by amplitudes
def domegac_c6(zamp, rohp, rohm, omegap, omegam, c6, c2=-1.496e4):
    """
    Thesis Doerr 2.29
    default c2, c6 from thesis Roux table 5.2
    """
    print('C6/C2', (c6/c2).n, (c6/c2).s )
    print('C6 a', 15/4 * c6/c2 )
    print('C6 b',  omegam*omegap/(omegap-omegam) )
    print('C6 c',  (rohp**2 - rohm**2)*(-3*zamp**2 + rohp**2 + rohm**2) )
    return 15/4 * c6/c2 * omegam*omegap/(omegap-omegam) * (rohp**2 - rohm**2)*(-3*zamp**2 + rohp**2 + rohm**2)

def domegap_c6(zamp, rohp, rohm, omegap, omegam, c6, c2=-1.496e4):
    """
    Thesis Doerr 2.29
    default c2, c6 from thesis Roux table 5.2
    """
    return -15/8 * c6/c2 * omegam/(omegap-omegam) * (3*zamp**4 + rohp**4 + 3*rohm**4 - 6*rohp**2*zamp**2 - 12*rohm**2*zamp**2 + 6*rohp**2*rohm**2) * omegap

def domegam_c6(zamp, rohp, rohm, omegap, omegam, c6, c2=-1.496e4):
    """
    Thesis Doerr 2.30
    default c2, c6 from thesis Roux table 5.2
    """
    return 15/8 * c6/c2 * omegam/(omegap-omegam) * (3*zamp**4 + rohm**4 + 3*rohp**4 - 6*rohm**2*zamp**2 - 12*rohp**2*zamp**2 + 6*rohp**2*rohm**2) * omegam

def domegaz_c6(zamp, rohp, rohm, omegaz, c6, c2=-1.496e4):
    """
    Thesis Doerr 2.31
    default c2, c6 from thesis Roux table 5.2
    """
    return 15/16 * c6/c2 * (zamp**4 + 3*rohm**4 + 3*rohp**4 - 6*rohp**2*zamp**2 - 6*rohm**2*zamp**2 + 12*rohp**2*rohm**2) * omegaz

def dR_c6(Rmeas, rhopA, zampA, rhomA, omegacA, omegapA, omegamA, 
            rhopB, zampB, rhomB, omegacB, omegapB, omegamB, c6, c2=-1.496e4):
    Cr = 15*c6/4/c2
    A = omegapA*omegamA/(omegapA-omegamA) * (rhopA**2 - rhomA**2) / omegacA * (-3*zampA**2 + rhopA**2 + rhomA**2)
    B = omegapB*omegamB/(omegapB-omegamB) * (rhopB**2 - rhomB**2) / omegacB * (-3*zampB**2 + rhopB**2 + rhomB**2)
    return Rmeas*((1+Cr*A)/(1+Cr*B) - 1)

# C4 shift by energies
def domegapzm_c6c4(rohp, zamp, rohm, omegap, omegaz, omegam, c4, c6, c2=-1.496e4):
    """
    """
    dp = domegap_c4(zamp, rohp, rohm, omegap, omegam, c4, c2=c2)
    dz = domegaz_c4(zamp, rohp, rohm, omegaz, c4, c2=c2)
    dm = domegam_c4(zamp, rohp, rohm, omegap, omegam, c4, c2=c2)
    
    dp += domegap_c6(zamp, rohp, rohm, omegap, omegam, c6, c2=c2)
    dz += domegaz_c6(zamp, rohp, rohm, omegaz, c6, c2=c2)
    dm += domegam_c6(zamp, rohp, rohm, omegap, omegam, c6, c2=c2)

    return dp, dz, dm

# C4 shift by energies
def domegaz_c4c6(Ez, omegaz, q, U, c4, c6, c2=-1.496e4):
    """
    Thesis Roux 3.24
    default c2 from thesis Roux table 5.2, c4, c6 is needed
    """
    prefac = Ez/q/elq/U
    delta_omegaz = omegaz * 3/4 * prefac * (c4/c2**2 + 5/4 * c6/c2**2 * prefac)

    return delta_omegaz


################################################################################################################################################


################################################################################################################################################
### Trap potential - odd Ci

'''
def dz_c1(c1, c2=-1.496e4):
    #J. Ketter PhD (4.37)
    return -c1/c2/2 # *d ? damn characteristic length...

def domegaz_c1c3(U0, UA, omegaz, c1, c3, c2=-1.496e4, d=1, z0=1):
    #Brown / Gabrielse 1986 (9.24)
    return -3/4*(d/z0)*c3*c1*(UA/U0)**2 * omegaz
'''

def dz_c1(c1, c2=-1.496e4):
    '''
    S. Dickopf PhD
    '''
    return -c1/c2/2

def dz_c3(rhop, rhom, zamp, c3, c2=-1.496e4):
    '''
    S. Dickopf PhD
    '''
    return 3/4*c3/c2*(rhom**2 + rhop**2 - zamp**2)

def domegaz_c3(rhop, rhom, zamp, omegaz, c3, c2=-1.496e4):
    '''
    S. Dickopf Phd
    '''
    return -1/16*c3**2/c2**2 * ( 18*(rhop**2 + rhom**2) - 15*zamp**2)* omegaz

################################################################################################################################################


################################################################################################################################################
### Magnetic field inhomogeneities

def dz_rhop_b1(rhop, omegac, omegap, omegaz, B1=1.41e-03, B0=7):
    '''
    Thesis Sailer (8.3)
    This shift is based on the ions magnetic moment, considering
    a high cyclotron radius rhop >> rhom
    '''
    return B1/B0/2 * omegac*omegap/omegaz**2 * rhop**2

def dz_rhom_b1(rhom, omegac, omegam, omegaz, B1=1.41e-03, B0=7):
    '''
    Thesis Sailer (8.3)
    This shift is based on the ions magnetic moment, considering
    a high magnetron radius rhom >> rhop
    '''
    return B1/B0/2 * omegac*omegam/omegaz**2 * rhom**2

def dB_b1_dz(dz, B1=1.41e-03):
    '''
    Thesis Roux (text below 3.28)
    '''
    return -2*B1*dz

def domegac_b2(rhop, rhom, zamp, omegap, omegam, B2=6.4e-2, B0=7):
    '''
    '''
    return B2/B0/2 * (omegap+omegam) * (zamp**2 + rhop**2*omegam/(omegap - omegam) - rhom**2*omegap/(omegap - omegam) )

def domegap_b2(rhop, rhom, zamp, omegap, omegam, B2=6.4e-2, B0=7):
    '''
    '''
    return B2/B0/2 * omegap*(omegap+omegam)/(omegap - omegam) * (zamp**2 - rhop**2 - rhom**2*(1 - omegam/omegap) )

def domegam_b2(rhop, rhom, zamp, omegap, omegam, B2=6.4e-2, B0=7):
    '''
    '''
    return -B2/B0/2 * omegam*(omegap+omegam)/(omegap - omegam) * (zamp**2 - rhom**2 - rhop**2*(omegap/omegam + 1) )

def domegaz_b2(rhop, rhom, omegaz, omegap, omegam, B2=6.4e-2, B0=7):
    '''
    '''
    return B2/B0/4 * omegaz*(omegap+omegam)/omegap/omegam * (rhom**2*omegam + rhop**2*omegap)

def dR_b2(Rmeas, rohpA, rohmA, zampA, omegapA, omegamA,
          rohpB, rohmB, zampB, omegapB, omegamB, B2=6.4e-2, B0=7):
    Br = B2/2/B0
    A = zampA**2 + rohpA**2*omegamA/(omegapA-omegamA) - rohmA**2*omegapA/(omegapA-omegamA)
    B = zampB**2 + rohpB**2*omegamB/(omegapB-omegamB) - rohmB**2*omegapB/(omegapB-omegamB)
    return Rmeas*((1+Br*A)/(1+Br*B) - 1)

'''
def dR_b2alt(Rmeas, rohpA, rohmA, zampA, omegapA, omegamA,
          rohpB, rohmB, zampB, omegapB, omegamB, B2=6.4e-2, B0=7):
    Br = B2/2/B0
    A = zampA**2 + rohpA**2*omegamA/(omegapA-omegamA) - rohmA**2*omegapA/(omegapA-omegamA)
    B = zampB**2 + rohpB**2*omegamB/(omegapB-omegamB) - rohmB**2*omegapB/(omegapB-omegamB)
    return Rmeas*((1+Br*A)/(1+Br*B) - 1)
'''

################################################################################################################################################


################################################################################################################################################
### Mixed Trap potential & Magnetic field inhomogeneities

# TODO: B1C1 / B1C3 


################################################################################################################################################


################################################################################################################################################
### Image current shift and related

# coil aka frequency pulling aka image current shift
def domegaz_freqpull(omegaz, omegares, q=42, m=172, Q=4000, L=1.5e-3, d_eff=11e-3):
    '''
    This is put together from multiple sources.... check thesises by Andreas Weigel, Alexander Egl, Thompson 
    '''

    R = omegares * L * Q
    domega = omegaz - omegares
    return (q*elq)**2/m/ukg/d_eff**2/np.pi * R*Q*domega/omegares / (1 + 4*Q**2*(domega/omegares)**2)

def domegaz_freqpull_ionstr(ionstr, U0, omegares, Q=4000, L=1.5e-3, d_eff=11e-3):
    '''
    Same, same, but different.
    '''
    omegaz = itp.omegaz_ionstr(ionstr, U0, nominal=True)
    _, _, q = itp.re_ionstr(ionstr)
    m, dm = ame.get_ion_mass(ionstr)
    return domegaz_freqpull(omegaz, omegares, q=q, m=m, Q=Q, L=L, d_eff=d_eff)

# dip fit corrections
def domegaz_dnures(dnuz_per_dnures, dnures):
    '''
    Not worth a function, just here for explaination:
    Since the dip fit is actually "correcting" for the coil pulling effect, the determined nu_z from a fit is the "unperturbed" axial frequency.
    This correction can be incomplete, meaning that there is a residual dependency of the dip fit axial frequency on the detuning, typically linear
    around the 10-40Hz range around the resonator. This slope is determined by doing hundreds of dip measurements while scanning (randomized) the 
    resonator frequency. 

    It doesn't matter if its nuz or omegaz, just be consistent and remember that the return is the same unit
    '''
    return dnuz_per_dnures*dnures

def domegac_dnures(dnuz_per_dnures, dnures, omegac, omegaz):
    '''
    Not worth a function, just here for explaination:
    Since the dip fit is actually "correcting" for the coil pulling effect, the determined nu_z from a fit is the "unperturbed" axial frequency.
    This correction can be incomplete, meaning that there is a residual dependency of the dip fit axial frequency on the detuning, typically linear
    around the 10-40Hz range around the resonator. This slope is determined by doing hundreds of dip measurements while scanning (randomized) the 
    resonator frequency. 
    '''
    return omegaz/omegac*domegaz_dnures(dnuz_per_dnures, dnures)

def dR_offresonator(R, dnuz_per_dnuresA, dnuz_per_dnuresB, domegaresA, domegaresB, omegazA, omegazB, omegacA, omegacB):
    return R * ( omegazB/omegacB**2 * dnuz_per_dnuresB*domegaresB - omegazA/omegacA**2 * dnuz_per_dnuresA*domegaresA )


################################################################################################################################################
### Other stuff

### Spin state
def domegaz_spin(g, m, omegaz, B2=6e-2):
    return g/m/ueV/omegaz  /2 * bohr_magneton_eV * B2

################################################################################################################################################

################################################################################################################################################
### Summarized effects

### radii dependent shifts
def domegac(omegac, omegap, omegaz, omegam, rhop, zamp, rhom, mass_u, REL=True, C2=-1.496e4, C4=0, C6=0, B0=7, B1=0, B2=0, C1=0, C3=0):
    '''
    Sum of all shifts which are dependent on the radius, with correlation in mind!
    '''
    # rel
    print('radii**2', rhop**2, zamp**2, rhom**2)
    shifts = []
    if REL:
        drel_sideband = domegac_rel(rhop, omegac, omegam, omegap)
        drelp = domegap_rel(rhop, zamp, rhom, omegap, omegaz, omegam)
        drelz = domegaz_rel(rhop, zamp, rhom, omegap, omegaz, omegam)
        drelm = domegam_rel(rhop, zamp, rhom, omegap, omegaz, omegam)
        drel_inv = itp.omegac_invariance(omegap+drelp, omegaz+drelz, omegam+drelm) - itp.omegac_invariance(omegap, omegaz, omegam)
        print('drel sideband', drel_sideband)
        print('drel invariance', drel_inv) 
        shifts.append( drel_inv )
    
    dc4_sideband = domegac_c4(rhop, rhom, omegap, omegam, C4, C2)
    dc4p = domegap_c4(zamp, rhop, rhom, omegap, omegam, C4, C2)
    dc4z = domegaz_c4(zamp, rhop, rhom, omegaz, C4, C2)
    dc4m = domegam_c4(zamp, rhop, rhom, omegap, omegam, C4, C2)
    dc4_inv = itp.omegac_invariance(omegap+dc4p, omegaz+dc4z, omegam+dc4m) - itp.omegac_invariance(omegap, omegaz, omegam)
    print('dc4 sideband', dc4_sideband)
    print('dc4 invariance', dc4_inv)
    shifts.append( dc4_inv )
    dc6_sideband = domegac_c6(zamp, rhop, rhom, omegap, omegam, C6, C2)
    dc6p = domegap_c6(zamp, rhop, rhom, omegap, omegam, C6, C2)
    dc6z = domegaz_c6(zamp, rhop, rhom, omegaz, C6, C2)
    dc6m = domegam_c6(zamp, rhop, rhom, omegap, omegam, C6, C2)
    dc6_inv = itp.omegac_invariance(omegap+dc6p, omegaz+dc6z, omegam+dc6m) - itp.omegac_invariance(omegap, omegaz, omegam)
    print('dc6 sideband', dc6_sideband)
    print('dc6 invariance', dc6_inv)
    shifts.append( dc6_inv )
    
    dc3z = domegaz_c3(zamp, omegaz, C3, C2)
    print('dnuz c3', dc3z/2/np.pi)
    dc3_inv = itp.omegac_invariance(omegap, omegaz+dc3z, omegam) - itp.omegac_invariance(omegap, omegaz, omegam)
    print('dc3 invariance', dc3_inv, dc3_inv.n, dc3_inv.s)
    shifts.append( dc3_inv )

    db2_sideband = domegac_b2(rhop, rhom, zamp, omegap, omegam, B2, B0)
    db2p = domegap_b2(rhop, rhom, zamp, omegap, omegam, B2, B0)
    db2z = domegaz_b2(rhop, rhom, omegaz, omegap, omegam, B2, B0)
    db2m = domegam_b2(rhop, rhom, zamp, omegap, omegam, B2, B0)
    db2_inv = itp.omegac_invariance(omegap+db2p, omegaz+db2z, omegam+db2m) - itp.omegac_invariance(omegap, omegaz, omegam)
    print('db2 sideband', db2_sideband)
    print('db2 invariance', db2_inv)
    shifts.append( db2_inv )

    print(shifts)
    print(correlation_matrix(shifts))
    
    sumomegac = 0
    for val in shifts: # real loop for easy ufloat
        print(val/omegac)
        sumomegac += val

    return sumomegac 

def domegac_Tion(Tz, ion_str, nures=700e3, C2=-1.496e4, C4=0, C6=0, B0=7, B1=0, B2=0, C1=0, C3=0):
    mass_u, dmass_u = ame.get_ion_mass(ion_str)
    omc, omp, omz, omm, p, z, m = itp.ion_stats(ion_str, nures, None, Tz, B0)
    print("nup, nuz, num", omp/2/np.pi, omz/2/np.pi, omm/2/np.pi)
    return domegac(omc, omp, omz, omm, p, z, m, mass_u, True, C2, C4, C6, B0, B1, B2, C1, C3), omc
    #return domegac(omc, omp, omz, omm, p, z, m, mass_u, False, C2, 0, 0, B0, B1, 0, 0, C3), omc

def domegac_iontrap(ion_str, trap_config):
    Tz = trap_config['Temp_res']
    nures = trap_config['nu_res'].n
    B0 = trap_config['B0'].n
    B1 = trap_config['B1']
    B2 = trap_config['B2']
    C2 = trap_config['C2']
    C4 = (trap_config['TR_opt4'] - trap_config['TR_set']) * trap_config['d4']
    #print(correlation_matrix([C4, trap_config['TR_opt4'], trap_config['d4']]))
    C6 = (trap_config['TR_opt6'] - trap_config['TR_set']) * trap_config['d6']
    C1 = trap_config['C1']
    C3 = trap_config['C3']
    return domegac_Tion(Tz, ion_str, nures, C2, C4, C6, B0, B1, B2, C1, C3)



################################################################################################################################################

###
###  Reformat ratio results (neutral ratio, mass, Q-value)
###

def neutral_mass(ratio_measured, dratio_measured, ion_ref, ion_ioi, Rcompare = None, fig_size=(6,4),
                 Eb_ref=None, Eb_ioi=None, sys_on=True, show=True, exc_radius = None, dexc_radii_ratio=None, ICS_precision=0.05,
                 omegac_ref=None, omegac_ioi=None, omegap_ref=None, omegap_ioi=None, omegam_ref=None, omegam_ioi=None):

    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    plt.rcParams["figure.figsize"] = fig_size
    font = {'size'   : 10}
    matplotlib.rc('font', **font)

    """
    ratio measured is always given in R = ion_ioi / ion_ref = nuc_ioi / nuc_ref = (q/m)_ioi / (q/m)_ref = m_ref/m_ioi * q_ioi/q_ref, but it is tested for this as well.
    """
    # ufloat ratio
    uR = ufloat(ratio_measured, dratio_measured)
    if show: print("ratio", uR)

    # reference stuff
    isotope, charge = re.findall(r'\d+', ion_ref)
    element_key_ref = ion_ref[len(isotope):-(len(charge)+1)]
    A_ref = int(isotope)
    q_ref = int(charge)

    # neutral reference mass
    ame_ref = ame.get_isotope(A=A_ref, el=element_key_ref)
    Z_ref = int(ame_ref.Z)
    u_ref = ufloat(float(ame_ref.umass), float(ame_ref.umass_err))
    eV_ref = ufloat(float(ame_ref.mass), float(ame_ref.mass_err))*1e3 + A_ref*ueV
    unc_ref = float(ame_ref.precision)
    if show: print('neutral reference mass (name, u, eV, rel unc)', ion_ref, u_ref, eV_ref, unc_ref)

    # ion of interest values
    isotope_ioi, charge_ioi = re.findall(r'\d+', ion_ioi)
    element_key_ioi = ion_ioi[len(isotope_ioi):-(len(charge_ioi)+1)]
    A_ioi = int(isotope_ioi)
    q_ioi = int(charge_ioi)

    # ame isotope of interest neutral mass for comparison
    ame_ioi = ame.get_isotope(A=A_ioi, el=element_key_ioi)
    Z_ioi = int(ame_ioi.Z)
    uame_ioi = ufloat(float(ame_ioi.umass), float(ame_ioi.umass_err))
    eVame_ioi = ufloat(float(ame_ioi.mass), float(ame_ioi.mass_err))*1e3 + A_ioi*ueV
    unc_ioi = float(ame_ioi.precision)
    if show: print('neutral ioi mass (name, u, eV, rel unc)', ion_ioi, uame_ioi, eVame_ioi, unc_ioi)
    #print(ame_ioi)

    # flip ratio if needed
    #  R = m_ref/m_ioi * q_ioi/q_ref
    testR = (q_ioi/q_ref * eV_ref / eVame_ioi) - 1
    testR2 = ratio_measured - 1
    flipped = False
    if testR*testR2 < 0:
        flipped = True
        uR = 1/uR
        if show: print("flipped ratio", uR)

    # total binding of missing electrons reference
    if Eb_ref is None:
        total_bin_ref = NISTie.get_total_binding(Z_ref, 0, q_ref-1, None, show)
        total_bin_ref = ufloat(total_bin_ref[0],total_bin_ref[1])
    else:
        total_bin_ref = Eb_ref

    # electron masses reference
    total_m_e_ref = q_ref*m_e # TODO: this is error-vise controversial?

    # reference ion mass
    eV_ref_ion = eV_ref - total_m_e_ref*ueV + total_bin_ref
    u_ref_ion = u_ref - total_m_e_ref + total_bin_ref/ueV
    if show: 
        print("total binding energy ref ion", total_bin_ref)
        print("ion reference mass (u,eV,rel unc)", u_ref_ion, eV_ref_ion, u_ref_ion.s/u_ref_ion.n)
        try:
            print("errors ref mass, emass, binding (eV)", eV_ref.s, total_m_e_ref.s, total_bin_ref.s, np.sqrt(eV_ref.s**2 + total_bin_ref.s**2 + total_m_e_ref.s**2))
        except:
            print("errors ref mass, binding (eV)", eV_ref.s, total_bin_ref.s, np.sqrt(eV_ref.s**2 + total_bin_ref.s**2))

    ### SYSTEMATICS: ###########################################################################################################################
    
    if sys_on:
        # image charge
        trap_radius = ufloat(5e-3, 50e-6)
        if exc_radius is None:
            exc_radius = ufloat(15.1, 8)*1e-6
        eVame_diff = eVame_ioi - eV_ref

        #precise ICS only necessary for very large m and q/m differences
        ICS_ratio_shift_precise = dR_precise_ICS(uR, ufloat(7.00216735, 1e-3), trap_radius, eVame_diff, omegac_ioi, omegap_ioi, omegam_ioi, eV_ref, u_ref*ukg,  omegac_ref, omegap_ref, omegam_ref)
        ICS_ratio_shift_precise = ufloat(ICS_ratio_shift_precise.n, abs(ICS_ratio_shift_precise.n)*ICS_precision)
        uR_ICS = uR - ICS_ratio_shift_precise
        #ICS_ratio_shift = systematics.dR_ICS(uR, ufloat(7.00216735, 1e-3), trap_radius, eVame_diff)
        #uR_ICS = uR - ICS_ratio_shift
        if show: 
            print('ICS ratio shift, uR, uR_new', ICS_ratio_shift_precise, uR, uR_ICS)
            print('old ratio error, new error', uR.s, uR_ICS.s)

        # relativistc
        # dR = R_meas - R_real, R = B/A
        if dexc_radii_ratio is None:    
            dexc_radii_ratio = 0.01
        print( 'exc radii for RS:',  exc_radius,  exc_radius*ufloat(1.0, dexc_radii_ratio) )
        dR_RS = R_rel(uR, omegap_ioi.n, omegap_ref.n, exc_radius, exc_radius*ufloat(1.0, dexc_radii_ratio) ) - uR
        uR_RS = uR_ICS - dR_RS
        if show: 
            print('RS ratio shift, uR, uR_new', dR_RS, uR_ICS, uR_RS)
            print('old ratio error, new error', uR_ICS.s, uR_RS.s)

        uR = uR_RS
    ################ ###########################################################################################################################

    if show: print("corrected ratio, 1/ratio", uR, 1/uR )

    # calc mass of ION of interest
    eV_ioi_ion = q_ioi/q_ref / uR * eV_ref_ion
    u_ioi_ion = q_ioi/q_ref / uR * u_ref_ion
    if show: 
        print('ion of interest mass via ref/ratio*qratio (u, eV)', u_ioi_ion, eV_ioi_ion)
        print('qioi/qref, ref_mass.n/R*q/q ()', q_ioi/q_ref, q_ioi/q_ref / uR * eV_ref_ion)

    if Eb_ioi is None:
        total_bin_ioi = NISTie.get_total_binding(Z_ioi, 0, q_ioi-1, None, show)
        total_bin_ioi = ufloat(total_bin_ioi[0], total_bin_ioi[1])
    else:
        total_bin_ioi = Eb_ioi
        #total_bin = ufloat(28651,200)

    # electron masses
    total_m_e_ioi = q_ioi*m_e # TODO: this is error-vise controversial?

    # neutral mass (from this data)
    eV_ioi = eV_ioi_ion + total_m_e_ioi*ueV - total_bin_ioi
    u_ioi = u_ioi_ion + total_m_e_ioi - total_bin_ioi/ueV

    if show:
        print("ioi eV emass, binding", total_m_e_ioi*ueV, total_bin_ioi)
        print("ioi u emass, binding", total_m_e_ioi, total_bin_ioi/ueV)
        print("total binding energy interest ion", total_bin_ioi.n, total_bin_ioi.s)
        print("neutral IoI mass in eV", eV_ioi, eVame_ioi, u_ioi-eVame_ioi)
        print("neutral IoI mass in u", u_ioi, uame_ioi, u_ioi-uame_ioi)
        try:
            print("errors mass, emass, binding (eV), ref_mass*R*q/q", eV_ioi.s, total_m_e_ioi.s, total_bin_ioi.s,  q_ioi/q_ref / uR * eV_ref_ion.s)
        except:
            print("errors mass, binding (eV), ref_mass*R*q/q", eV_ioi.s, total_bin_ioi.s,  q_ioi/q_ref / uR * eV_ref_ion.s)

    if show: 

        plt.errorbar(["AME"], uame_ioi.n, uame_ioi.s, fmt='^', c='k')
        plt.errorbar(["this work"], u_ioi.n, u_ioi.s, fmt='^', c='y')
        plt.show()

    # neutral ratio
    nRatio = eV_ioi / eV_ref
    # neutral ratio (better)
    uR = 1/uR
    nRatio = uR*q_ioi/q_ref + total_m_e_ioi*(1-uR)/u_ref + ((uR*q_ioi/q_ref - 1)*total_bin_ref/ueV + (total_bin_ioi-total_bin_ref)/ueV)/u_ref
    uR = 1/uR
    #nRatio = 1/nRatio

    nRatio_ame = uame_ioi / u_ref
    if Rcompare is not None:
        rRatio_ame = Rcompare
    if show: 
        print('neutral ame', nRatio_ame, nRatio_ame.s)
        print('neutral ame relative', nRatio, nRatio_ame.s/nRatio_ame.n)
        print('neutral 1/ame', 1/nRatio_ame, (1/nRatio_ame).s)
        print('neutral ratio', nRatio, nRatio.s)
        print('neutral ratio relative', nRatio, nRatio.s/nRatio.n)
        print('neutral 1/ratio', 1/nRatio, (1/nRatio).s)
        print("Rneutral / diff to ame", nRatio, nRatio_ame, nRatio - nRatio_ame)

        plt.errorbar(["AME"], nRatio_ame.n, nRatio_ame.s, fmt='^', c='k')
        plt.errorbar(["this work"], nRatio.n, nRatio.s, fmt='^', c='y')

        plt.xlim(-0.7,1.7)
        plt.grid(which='both')
        plt.tick_params(direction="in", bottom=True, top=True, left=True, right=True)
        plt.ylabel('neutral mass ratio (1)')
        plt.tight_layout()
        plt.savefig('RATIO_compare.svg')
        plt.savefig('RATIO_compare.png')
        plt.show()

        #plt.errorbar(["ame"], uame_ioi.n, uame_ioi.s, fmt='^', c='k')
        #plt.errorbar(["this work"], u_ioi.n, u_ioi.s, fmt='^', c='y')
        #plt.show()

        print("u mass measured, u mass ame, difference ", u_ioi, uame_ioi, u_ioi-uame_ioi)
        print("u mass measured relative error ", u_ioi.s/u_ioi.n)
        print("u mass ame relative error ", uame_ioi.s/uame_ioi.n)

    # mass excesses (u_real - A)
    eV_mass_excess = eV_ioi - A_ioi*ueV
    ame_mass_excess = ufloat(float(ame_ioi.mass*1000), float(ame_ioi.mass_err*1000))

    # better eV mass excess
    uR = 1/uR
    eV_mass_excess = ueV*(uR*q_ioi/q_ref*u_ref - A_ioi) + q_ioi*m_e*ueV*(1-uR) + uR*q_ioi/q_ref*total_bin_ref - total_bin_ioi
    #eV_mass_excess = 1/ueV*(uR_sys*charge_ioi/charge*u_ref - isotope_ioi)+uR_sys*charge_ioi/charge*(total_bin_ref-m_e*charge)+m_e*charge_ioi - total_bin_ioi

    if show: 
        print('mass excess (this, ame, this-ame)', eV_mass_excess, ame_mass_excess, eV_mass_excess-ame_mass_excess)
        print('our value, our error', eV_mass_excess.n, eV_mass_excess.s )
        print('ame value, ame error', ame_mass_excess.n, ame_mass_excess.s )
        print('ame - meas (eV, sqr sum error)', eV_mass_excess.n - ame_mass_excess.n, np.sqrt(eV_mass_excess.s**2 + ame_mass_excess.s**2) )
        print('ame - meas (in combined sigma)', (eV_mass_excess.n - ame_mass_excess.n)/np.sqrt(eV_mass_excess.s**2 + ame_mass_excess.s**2) )

        plt.errorbar(["AME"], ame_mass_excess.n, ame_mass_excess.s, fmt='^', c='k')
        plt.errorbar(["this work"], eV_mass_excess.n, eV_mass_excess.s, fmt='^', c='y')
        
        plt.xlim(-0.7,1.7)
        plt.grid(which='both')
        plt.tick_params(direction="in", bottom=True, top=True, left=True, right=True)
        plt.ylabel('$m_{meas} - A_{isotope}$ (eV)')
        plt.tight_layout()
        plt.savefig('MASS_compare.svg')
        plt.savefig('MASS_compare.png')
        plt.show()

        plt.errorbar(["AME"], 0, ame_mass_excess.s, fmt='^', c='k')
        plt.errorbar(["this work"], eV_mass_excess.n- ame_mass_excess.n, eV_mass_excess.s, fmt='^', c='y')
        
        plt.xlim(-0.7,1.7)
        plt.grid(which='both')
        plt.tick_params(direction="in", bottom=True, top=True, left=True, right=True)
        plt.ylabel('$m_{meas} - m_{AME}$ (eV)')
        plt.tight_layout()
        plt.savefig('MASS_compare_diff.svg')
        plt.savefig('MASS_compare_diff.png')
        plt.show()

    Qvalue = eV_ioi - eV_ref
    if Qvalue < 0: Qvalue*(-1)

    if flipped:
        nRatio = eV_ref / eV_ioi
    else:
        nRatio = eV_ioi / eV_ref

    return eV_mass_excess, Qvalue, nRatio



if __name__ == "__main__":
        
    trap2_config = {
        'idx': 2,
        'trap_radius': ufloat(5e-3, 5e-6), # trap radius, 1 um mechanical tolerance (probably worse?)
        #'nu_res': ufloat(477000.7, 10.0),
        'nu_res': ufloat(736086.4, 10.0),
        'Q_res': ufloat(3500, 300.0),
        'L_res': ufloat(1.5e-3, 0.1e-3),
        'd_eff_res': ufloat(11e-3, 1e-3),
        'Temp_res': ufloat(7.0, 2.0),
        'B0': ufloat(7.00212326, 1e-08),
        'B1': ufloat(1.41e-03, 0.27e-03), # T/m
        #'B2': ufloat(0.064, 0.005), # T/m**2
        'B2': ufloat(0.064, 0.005), # T/m**2
        #'B2': ufloat(0.024, 0.005), # T/m**2
        'B3': ufloat(0, 0),
        'B4': ufloat(0, 0),
        'C1': ufloat(0, 0),
        'C2': ufloat(-1.496e4, 70), ## NOTE: all  # 1/m**2
        'd2': ufloat(-2.5, 124),
        'C3': ufloat(2.6e-06, 7.6e-06)*1e9,
        #'d4': ufloat(0.8878e-3, 2.6e-06), # 1/mm**4
        'd4': ufloat(0.8878e-3, 1e-05)*1e12, # 1/mm**4 *1e12 to convert to 1/m**4
        #'d6': ufloat(-6.1e-05, 4e-06), # 1/mm**6
        'd6': ufloat(-6.1e-05, 1e-05)*1e18, #  1/mm**6 *1e18 to convert to 1/m**6
        #'TR_opt4': ufloat(0.880143, 4e-06),
        'TR_opt4': ufloat(0.880143, 1e-04),
        #'TR_opt6': ufloat(0.8786, 0.00013),
        'TR_opt6': ufloat(0.8786, 0.00013),
        'TR_set': 0.880143, # perfect C4 TR
        'epsilon': 0, #0.015, # 0.05
        'theta': 0, #0.5/180*np.pi, # 1
        'phi': 0, #1e-3/180*np.pi,
        'trap_offsets': [0, 0, 0, 0, 0],
        'Bdrift_per_hour': ufloat(-1.0, 0.1)*1e-9, # relative drift
        'dBB_per_second': ufloat(3, 0.5)*1e-11, # relative jitter
        'dUU_per_second': ufloat(1, 0.2)*1e-7, # relative jitter (allan dev at 1s)
    }

    trap3_config = {
        'idx': 3,
        'trap_radius': ufloat(5e-3, 5e-6), # trap radius, 1 um mechanical tolerance (probably worse?)
        'nu_res': ufloat(477000.7, 2.0),
        #'nu_res': ufloat(707000.7, 2.0),
        'Q_res': ufloat(10000, 300.0),
        'L_res': ufloat(1.5e-3, 0.1e-3),
        'd_eff_res': ufloat(11e-3, 1e-3),
        'Temp_res': ufloat(7.0, 2.0),
        'B0': ufloat(7.00276641, 1e-08),
        'B1': ufloat(-1.41e-03, 0.27e-03), # T/m
        #'B2': ufloat(0.064, 0.005), # T/m**2
        #'B2': ufloat(0.064, 0.005), # T/m**2
        'B2': ufloat(0.024, 0.005), # T/m**2
        'B3': ufloat(0, 0),
        'B4': ufloat(0, 0),
        'C1': ufloat(0, 0),
        'C2': ufloat(-1.496e4, 70), ## NOTE: all  # 1/m**2
        'd2': ufloat(-2.5, 124),
        'C3': ufloat(-6.1e-08, 6.1e-06)*1e9,
        #'d4': ufloat(0.8878e-3, 2.6e-06), # 1/mm**4
        'd4': ufloat(0.8878e-3, 1e-05)*1e12, # 1/mm**4 *1e12 to convert to 1/m**4
        #'d6': ufloat(-6.1e-05, 4e-06), # 1/mm**6
        'd6': ufloat(-6.1e-05, 1e-05)*1e18, #  1/mm**6 *1e18 to convert to 1/m**6
        #'TR_opt4': ufloat(0.880143, 4e-06),
        'TR_opt4': ufloat(0.880143, 1e-04),
        #'TR_opt6': ufloat(0.8786, 0.00013),
        'TR_opt6': ufloat(0.8786, 0.00013),
        'TR_set': 0.880143, # NOTE: BAD TR!
        'epsilon': 0.015, # 0.05
        'theta': 0.5/180*np.pi, # 1
        'phi': 1e-3/180*np.pi,
        'trap_offsets': [0, 0, 0, 0, 0],
        'Bdrift_per_hour': ufloat(-1.0, 0.1)*1e-9, # relative drift
        'dBB_per_second': ufloat(3, 0.5)*1e-11, # relative jitter
        'dUU_per_second': ufloat(1, 0.2)*1e-7, # relative jitter (allan dev at 1s)
    }


    trapHe = {
        'idx': 2,
        'trap_radius': ufloat(3.5e-3, 5e-6), # trap radius, 1 um mechanical tolerance (probably worse?)
        #'nu_res': ufloat(477000.7, 10.0),
        'nu_res': ufloat(500000.4, 10.0),
        'Q_res': ufloat(3500, 300.0),
        'L_res': ufloat(1.5e-3, 0.1e-3),
        'd_eff_res': ufloat(11e-3, 1e-3),
        'Temp_res': ufloat(10.0, 2.0),
        'B0': ufloat(5.7, 1e-08),
        'B1': ufloat(1.41e-03, 0.27e-03), # T/m
        #'B2': ufloat(0.064, 0.005), # T/m**2
        'B2': ufloat(0.064, 0.005), # T/m**2
        #'B2': ufloat(0.024, 0.005), # T/m**2
        'B3': ufloat(0, 0),
        'B4': ufloat(0, 0),
        'C1': ufloat(0, 0),
        'C2': ufloat(-1.496e4, 70), ## NOTE: all  # 1/m**2
        'd2': ufloat(-2.5, 124),
        'C3': ufloat(0, 0),
        #'d4': ufloat(0.8878e-3, 2.6e-06), # 1/mm**4
        'd4': ufloat(0.8878e-3, 1e-05)*1e12, # 1/mm**4 *1e12 to convert to 1/m**4
        #'d6': ufloat(-6.1e-05, 4e-06), # 1/mm**6
        'd6': ufloat(-6.1e-05, 1e-05)*1e18, #  1/mm**6 *1e18 to convert to 1/m**6
        #'TR_opt4': ufloat(0.880143, 4e-06),
        'TR_opt4': ufloat(0.880143, 1e-04),
        #'TR_opt6': ufloat(0.8786, 0.00013),
        'TR_opt6': ufloat(0.8786, 0.00013),
        'TR_set': 0.880143, # perfect C4 TR
        'epsilon': 0, #0.015, # 0.05
        'theta': 0, #0.5/180*np.pi, # 1
        'phi': 0, #1e-3/180*np.pi,
        'trap_offsets': [0, 0, 0, 0, 0],
        'Bdrift_per_hour': ufloat(-1.0, 0.1)*1e-9, # relative drift
        'dBB_per_second': ufloat(3, 0.5)*1e-11, # relative jitter
        'dUU_per_second': ufloat(1, 0.2)*1e-7, # relative jitter (allan dev at 1s)
    }


    TableTrap = {
        'idx': 42,
        'trap_radius': ufloat(5e-3, 5e-6), # trap radius, 1 um mechanical tolerance (probably worse?)
        #'nu_res': ufloat(477000.7, 10.0),
        'nu_res': ufloat(50000.4, 10.0),
        'Q_res': ufloat(3500, 300.0),
        'L_res': ufloat(1.5e-3, 0.1e-3),
        'd_eff_res': ufloat(11e-3, 1e-3),
        'Temp_res': ufloat(300.0, 2.0),
        'B0': ufloat(0.2, 1e-08),
        'B1': ufloat(1.41e-03, 0.27e-03), # T/m
        #'B2': ufloat(0.064, 0.005), # T/m**2
        'B2': ufloat(0.064, 0.005), # T/m**2
        #'B2': ufloat(0.024, 0.005), # T/m**2
        'B3': ufloat(0, 0),
        'B4': ufloat(0, 0),
        'C1': ufloat(0, 0),
        'C2': ufloat(-1.496e4, 70), ## NOTE: all  # 1/m**2
        'd2': ufloat(-2.5, 124),
        'C3': ufloat(0, 0),
        #'d4': ufloat(0.8878e-3, 2.6e-06), # 1/mm**4
        'd4': ufloat(0.8878e-3, 1e-05)*1e12, # 1/mm**4 *1e12 to convert to 1/m**4
        #'d6': ufloat(-6.1e-05, 4e-06), # 1/mm**6
        'd6': ufloat(-6.1e-05, 1e-05)*1e18, #  1/mm**6 *1e18 to convert to 1/m**6
        #'TR_opt4': ufloat(0.880143, 4e-06),
        'TR_opt4': ufloat(0.880143, 1e-04),
        #'TR_opt6': ufloat(0.8786, 0.00013),
        'TR_opt6': ufloat(0.8786, 0.00013),
        'TR_set': 0.880143, # perfect C4 TR
        'epsilon': 0, #0.015, # 0.05
        'theta': 0, #0.5/180*np.pi, # 1
        'phi': 0, #1e-3/180*np.pi,
        'trap_offsets': [0, 0, 0, 0, 0],
        'Bdrift_per_hour': ufloat(-1.0, 0.1)*1e-9, # relative drift
        'dBB_per_second': ufloat(3, 0.5)*1e-11, # relative jitter
        'dUU_per_second': ufloat(1, 0.2)*1e-7, # relative jitter (allan dev at 1s)
    }
    print("***\n\n")

    #dhel, chel = domegac_iontrap("4He1+", trapHe)
    dhel, chel = domegac_iontrap("14N4+", TableTrap)
    print("***\n\n", dhel/2/np.pi, chel/2/np.pi, "\n\n***\n\n")

    """
    dneon, cneon = domegac_iontrap("20Ne10+", trap2_config)
    dcarbon, ccarbon = domegac_iontrap("12C6+", trap2_config)
    print(correlation_matrix([dneon, dcarbon]))
    print(dneon)
    print(dcarbon)
    print(dcarbon-dneon)
    print(cneon/ccarbon)
    print((cneon+dneon)/(ccarbon+dcarbon))
    print((cneon+dneon)/(ccarbon+dcarbon) - (cneon/ccarbon))

    """

"""
###
### Testing
###

from fticr_toolkit import jitters as jitter
from fticr_toolkit import ideal_trap_physics as itp

c2 = -1.496e4

d2 = -8.5e-6
d4 = -0.0006110 # (0.0000049)
e4 = 0.0005379 # (0.0000043)
d6 = 0.0004537 # (0.0000065)
e6 = -0.000399 # (0.0000057)
TR = 0.8795
c4 = itp.c4(TR+0.01, d4, e4)
c6 = itp.c6(TR+0.01, d6, e6)

# Axial shift due to TR shift considering resonator temperature:

Treso = 12

nu_z = 27020.87 + 437000
print(nu_z)
nu_p = 16674079.42
nu_m = 6457.1
omegaz = nu_z*2*np.pi
omegam = nu_m*2*np.pi
omegap = nu_p*2*np.pi
Uring = 19.109

Tz = Treso
Tm = itp.T_sideband_thermalz(Tz, nu_z, nu_m)
Tp = itp.T_sideband_thermalz(Tz, nu_z, nu_p)
print("temperatures", Tp, Tz, Tm)

Ez = itp.E_thermal_1dim_osci(Tz)
Em = itp.E_thermal_1dim_osci(Tm)
Ep = itp.E_thermal_1dim_osci(Tp)
print("energies", Ep*eVperjoule, Ez*eVperjoule, Em*eVperjoule)
print("energies", Tp*eVboltzmann, Tz*eVboltzmann, Tm*eVboltzmann)

zamp = itp.zamp(Ez, omegaz, 187) * 1000
rohp = itp.rohp(Ep, omegap, omegam, 187) * 1000
rohm = itp.rohm(Em, omegap, omegam, 187) * 1000
print("radii", rohp, zamp, rohm)

domegap, domegaz, domegam = domegaAll_c4(Ez, Ep, Em, 2*np.pi*nu_z, 2*np.pi*nu_p, 2*np.pi*nu_m, 29, Uring, c4, c2)
print("domega C4", domegap, domegaz, domegam)
print("dnu C4", domegap/2/np.pi, domegaz/2/np.pi, domegam/2/np.pi)
domegaz = domegaz_c4c6(Ez, omegaz, 29, Uring, c4, c6, c2)
print("dnuz C4/6", domegaz/2/np.pi)
domegaz = domegaz_d2(omegaz, 0.01, d2, c2)
print("dnuz c2d2", domegaz/2/np.pi)

"""

""" # Axial jitter TODO: proper axial thermal amplitude jitter estimation (after excitation)


TR_span = np.arange(0.878, 0.882, 1e-6)

c4 = itp.c4(TR_span, d4, e4)
c6 = itp.c6(TR_span, d6, e6)

phase_jitterc4 = jitter.del_phic4(c2, c4, 0.14, 504000, 3.5)
phase_jitterc6 = jitter.del_phic6(c2, c6, 0.14, 504000, 3.5)

import matplotlib.pyplot as plt

plt.plot(TR_span, phase_jitterc4, label="c4")
plt.plot(TR_span, phase_jitterc6, label="c6")
#plt.plot(TR_span, np.sqrt(phase_jitterc4**2+phase_jitterc6**2), label="squared")
plt.plot(TR_span, phase_jitterc4+phase_jitterc6, label="both")
plt.legend()
plt.show()

"""
