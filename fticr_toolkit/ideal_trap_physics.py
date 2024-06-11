import numpy as np
from numpy.random import rand, exponential, normal
from scipy import constants
import re

from uncertainties import ufloat
from uncertainties.umath import *
from uncertainties import correlation_matrix
import uncertainties.unumpy as unp
from fticr_toolkit import ame

ukg = constants.physical_constants["atomic mass unit-kilogram relationship"][0]
eVJ = constants.physical_constants["electron volt-joule relationship"][0]
eVboltzmann = constants.physical_constants["Boltzmann constant in eV/K"][0]
hbar = constants.hbar

#print(ukg) # ukg = 1.661e-27 kg/u
#print(eVJ) # eVJ = 1.602e-19 J/eV

### basics

def re_ionstr(ion='187Re29+'):
    """splits up a ion description string and returns A, el, q as int, str, int

    Args:
        ion (str): Ion identification string, e.g. '187Re29+'.

    Raises:
        ValueError: when regular expressions fail the given ion string is not correct

    Returns:
        [tuple]: (A->int, el->str, q->int)
    """
    match = re.match(r"([0-9]+)([a-z]+)([0-9]+)", ion, re.I)
    if match:
        items = match.groups()
        A = int(items[0])
        el = items[1]
        q = int(items[2])
    else:
        raise ValueError("please provide proper ion string, e.g. '187Re29+'")
    return A, el, q

def ion_stats(ion_str, nures=None, U0=None, Tz=ufloat(7, 2), B0=7, nominal=False):
    A, el, q = re_ionstr(ion_str)
    mass_u, dmass_u = ame.get_ion_mass(ion_str)
    if not nominal:
        mass_u = ufloat(mass_u, dmass_u)
    omz = nures*2*np.pi if nures is not None else omegaz(q, mass_u, U0)
    omc = omegac(q, mass_u, B0)
    omp = omegap(omc, omz)
    omm = omegam(omc, omz)
    p, z, m = radii2(Tz, omp, omz, omm, mass_u, show_corr=False)
    #print("radii", p, z, m)
    return omc, omp, omz, omm, p, z, m

def omega(nu):
    return nu*2*np.pi

def omegac(q, mass, B=7):
    """
    q in units of e and mass in units of amu
    """
    return q/mass*B * constants.e/ukg # 1/u*T * C/kg*u = kg/A/s^2 * As/kg = 1/s

def B(q, mass, omegac):
    """
    q in units of e and mass in amu
    """
    return mass/q*omegac * ukg/ constants.e 

def B_ionstr(ionstr, omegac):
    """
    q in units of e and mass in amu
    """
    A, el, q = re_ionstr(ionstr)
    mass, dm = ame.get_ion_mass(ionstr)
    return mass/q*omegac * ukg/ constants.e 

def omegaz(q, mass, U0, c2=-1.496e4, nominal=False):
    """
    q in units of e and mass in units of amu, c2 in units of m (e.g. -1.4e4)
    """
    arr = unp.sqrt( 2*c2*q*U0/mass * constants.e/ukg)
    if nominal:
        return unp.nominal_values(arr)
    else:
        return arr

def omegaz_ionstr(ionstr, U0, c2=-1.496e4, nominal=False, nominal_mass=True):
    """
    q in units of e and mass in units of amu, c2 in units of m (e.g. -1.4e4)
    """
    _, _, q = re_ionstr(ionstr)
    m, dm = ame.get_ion_mass(ionstr)
    if nominal_mass:
        mass = m
    else:
        mass = ufloat(m, dm)
    return omegaz(q, mass, U0, c2, nominal)

def omegam_ionstr(ionstr, U0, c2=-1.496e4, B=7, nominal=False, nominal_mass=True):
    """
    q in units of e and mass in units of amu, c2 in units of m (e.g. -1.4e4)
    """
    _, _, q = re_ionstr(ionstr)
    m, dm = ame.get_ion_mass(ionstr)

    if nominal_mass:
        mass = m
    else:
        mass = ufloat(m, dm)

    omz = omegaz(q, mass, U0, c2, nominal)
    omc = omegac(q, mass, B=B)
    return omegam(omc, omz, nominal=nominal)

def omegap_ionstr(ionstr, U0, c2=-1.496e4, B=7, nominal=False, nominal_mass=True):
    """
    q in units of e and mass in units of amu, c2 in units of m (e.g. -1.4e4)
    """
    _, _, q = re_ionstr(ionstr)
    m, dm = ame.get_ion_mass(ionstr)

    if nominal_mass:
        mass = m
    else:
        mass = ufloat(m, dm, "ion mass ame")

    omz = omegaz(q, mass, U0, c2, nominal)
    omc = omegac(q, mass, B=B)
    return omegap(omc, omz, nominal=nominal)

def dnu_z(nu_z_ref, qm_ref, qm_ioi):
    # thats noice
    return nu_z_ref * (1 - np.sqrt( qm_ioi / qm_ref ) )

def domega_c(qm_ref, qm_ioi, B=7):
    """ omega c frequency shift between a reference and another ion
    d omega c = omega c A - omega c B     # (ref - ioi)

    Args:
        qm_ref (float): qmA
        qm_ioi (float): qmB
        B (int, optional): magnetic field in tesla Defaults to 7.

    Returns:
        float: delta omegac in 1/s radial frequency
    """
    return B * ( qm_ref - qm_ioi )

def dnu_p(nu_p_ref, q_ref, m_ref, q, m):
    return None

def U0(q, mass, nu_res, c2=-1.496e4):
    """
    q in units of e and mass in units of amu, nu_res in Hz (NOT ANGULAR FREQUENCY), c2 in units of m (e.g. -1.4e4)
    """
    return (nu_res*2*np.pi)**2 * mass / 2 / c2 / q * ukg/constants.e

def Umax(q, m, B0, c2=-1.496e-2):
    return q*constants.e / m / ukg * B0**2 /(4*c2*1e6)

def qm(nu, U0, c2=-1.496e4):
    """
    q in units of e and mass in units of amu, nu_res in Hz (NOT ANGULAR FREQUENCY), c2 in units of m (e.g. -1.4e4)
    """
    return (nu*2*np.pi)**2 / 2 / c2 / U0

def omegap(omegac, omegaz, nominal=False):
    arr = omegac/2 + unp.sqrt(omegac**2/4 - omegaz**2/2)
    if nominal:
        return unp.nominal_values(arr)
    else:
        return arr

def omegam(omegac, omegaz, nominal=False):
    arr = omegac/2 - unp.sqrt(omegac**2/4 - omegaz**2/2)
    if nominal:
        return unp.nominal_values(arr)
    else:
        return arr

def omegac_invariance(omegap, omegaz, omegam):
    return unp.sqrt(omegap**2 + omegaz**2 + omegam**2)

def omegac_sideband(omegap, omegam):
    return omegap + omegam

# honorable mentions:
# omegap*omegam = omegaz**2 / 2
#

### potential and magnetic field



### Energy and amplitudes
def Ep(mass, rohp, omegap, omegam):
    """
    mass in u
    roh in m
    omega in radians
    output in eV
    """
    return mass*ukg/2 * rohp**2 * (omegap**2 - omegap*omegam) / eVJ

def Em(mass, rohm, omegap, omegam):
    """
    mass in u
    roh in m
    omega in radians
    output in eV
    """
    return mass*ukg/2 * rohm**2 * (omegam**2 - omegap*omegam) / eVJ

def Ez(mass, zamp, omegaz):
    """
    mass in u
    roh in m
    omega in radians
    output in eV
    """
    return mass*ukg/2 * zamp**2 * omegaz**2 / eVJ

def n_p(Ep, omegap):
    return Ep*eVJ/hbar/omegap - 0.5

def Ep_qua(np, omegap):
    return (np + 0.5)*hbar*omegap / eVJ

def n_m(Em, omegam):
    return -Em*eVJ/hbar/omegam - 0.5

def Em_qua(nm, omegam):
    return -(nm + 0.5)*hbar*omegam / eVJ

def n_z(Ez, omegaz):
    return Ez*eVJ/hbar/omegaz - 0.5

def Ez_qua(nz, omegaz):
    return (nz + 0.5)*hbar*omegaz / eVJ

def Etot(mass, rohp, zamp, rohm, omegap, omegaz, omegam):
    """
    mass in u
    roh in m
    omega in radians
    output in eV
    """
    return Ep(mass, rohp, omegap, omegam) + Em(mass, rohm, omegap, omegam) + Ez(mass, zamp, omegaz)

def E_thermal_1dim_osci(temp):
    """
    1 dimensional oszillator thermal energy
    temp in K
    output in eV
    """
    return eVboltzmann * temp

def zamp(Ez, omegaz, mass):
    """
    mass in u
    omega in radians
    energy in eV
    output in m
    """
    return unp.sqrt(2 * Ez*eVJ / mass / ukg) / omegaz

def zamp_sideband_m(rhom, omegam, omegaz):
    """See Egl, A PhD thesis 2020, eq. (2.53b)

    Args:
        rhom (float): radius in m
        omegam (float): eigenfrequency
        omegaz (float): eigenfrequency
    """
    return unp.sqrt(omegaz/2/omegam)*rhom


def zamp_sideband_p(rhop, omegap, omegaz):
    """See Egl, A PhD thesis 2020, eq. (2.53b)

    Args:
        rhom (float): radius in m
        omegam (float): eigenfrequency
        omegaz (float): eigenfrequency
    """
    return unp.sqrt(omegap/omegaz)*rhop

def zamp_Tz(Tz, omegaz, mass):
    return unp.sqrt(2*constants.k*Tz/mass/ukg/omegaz**2)


def rhom(Em, omegap, omegam, mass):
    """
    mass in u
    omega in radians
    energy in eV
    output in m
    """
    return unp.sqrt(Em*eVJ / mass / ukg * 2 / (omegam**2 - omegap*omegam) )

def rhom_Tz(Tz, omegam, omegaz, mass):
    zamp = zamp_Tz(Tz, omegaz, mass)
    return unp.sqrt(2*omegam/omegaz)*zamp

def rhop(Ep, omegap, omegam, mass):
    """
    mass in u
    omega in radians
    energy in eV
    output in m
    """
    return unp.sqrt(Ep*eVJ / mass / ukg * 2 / (omegap**2 - omegap*omegam) )

def rhop_Tz(Tz, omegap, omegaz, mass):
    zamp = zamp_Tz(Tz, omegaz, mass)
    Tp = omegap/omegaz*Tz
    #print(Tp)
    rhop = unp.sqrt(2*constants.k*Tp/mass/ukg/omegap**2)
    return rhop
    #return unp.sqrt(omegaz/omegap)*zamp

def T_sideband(Ta, omegaa, omegab):
    """
    Thesis Doerr 3.32
    Tb = omegab/omegaa * Ta
    """
    return omegab/omegaa * Ta

def radii2(Tz, omegap, omegaz, omegam, mass_u, show_corr=True):
    z = zamp_Tz(Tz, omegaz, mass_u)
    p = rhop_Tz(Tz, omegap, omegaz, mass_u)
    m = rhom_Tz(Tz, omegam, omegaz, mass_u)
    try:
        if show_corr:
            print(correlation_matrix([p, z, m]))
    except Exception as e:
        print("correlation matrix failed: ", e)
    return p, z, m

def radii(omegap, omegaz, omegam, ion_mass, rhop_exc = 0, z_exc = 0, rhom_exc = 0, Tz=4, simcount = 5e3):
    phis = [rand(simcount)*2*np.pi, rand(simcount)*2*np.pi, rand(simcount)*2*np.pi]

    Tp = Tz * unp.mean(omegap)/unp.mean(omegaz)
    Tm = Tz * unp.mean(omegam)/unp.mean(omegaz)

    E0p = exponential(constants.k*Tp, simcount)
    r0p = unp.sqrt(2*E0p/ukg/ion_mass/omegap**2) 

    E0m = exponential(constants.k*Tm, simcount)
    r0m = np.sqrt(4*E0m/ukg/ion_mass/omegaz**2) 

    E0z = exponential(constants.k*Tm*Tz, simcount)
    z0r = np.sqrt(2*E0z/ukg/ion_mass/omegaz**2)

    # switch coordinate system and add excitation radii
    x0p = r0p*np.cos(phis[0]) + rhop_exc
    y0p = r0p*np.sin(phis[0])
    z0x = z0r*np.cos(phis[1]) + z_exc
    z0y = z0r*np.sin(phis[1])
    x0m = r0m*np.cos(phis[2]) + rhom_exc
    y0m = r0m*np.sin(phis[2])

    # and switch back 
    r0p_exc = np.sqrt(x0p**2 + y0p**2)
    phip_exc = np.arctan2( y0p, x0p )
    z0r_exc = np.sqrt(z0x**2 + z0y**2)
    phiz_exc = np.arctan2( z0y, z0x )
    r0m_exc = np.sqrt(x0m**2 + y0m**2)
    phim_exc = np.arctan2( y0m, x0m )

    data = [r0p_exc, z0r_exc, r0m_exc, phip_exc, phiz_exc, phim_exc]

    return unp.mean(r0p_exc), unp.mean(z0r_exc), unp.mean(r0m_exc), data

## trap parameter

# TODO: calc e4 from opti TR & d4: e4 = -d4*TR_opti

def c4(TR, d4, e4):
    return d4*TR + e4

def dc4(dTR, d4):
    return d4*dTR

def c6(TR, d6, e6):
    return d6*TR + e6

def dc6(dTR, d6):
    return d6*dTR

if __name__ == "__main__":


    """ room temp Trap """
    ion = '14N1+'
    m, dm, data = ame.get_ion_mass(ion, full=True)
    q = data['q']
    print('ion', ion, m, dm, q)
    omc = omegac(q, m, 0.5)
    nuz = 10e3
    omz = 2*np.pi*nuz
    unull = U0(q, m, omz)
    print('U0', unull)

    omp = omegap(omc, omz)
    omm = omegam(omc, omz)
    print('nuc, p, z, m', omc/2/np.pi, omp/2/np.pi, nuz, omm/2/np.pi )

    z = zamp_Tz(300, omz, m)
    p = rhop_Tz(300, omp, omz, m)
    m = rhom_Tz(300, omm, omz, m)
    print('amps [um] p,z,m', p*1e6, z*1e6, m*1e6)

    sys.exit()

    """ Test wit Egl values 40Ar13+ """
    m_ar, _ = ame.get_ion_mass('40Ar13+')
    z = zamp_Tz(4.2, 2*np.pi*650e3, m_ar)
    print("should be 10.2um:", np.around(unp.nominal_values(z)*1e6, 3), "um", z)

    """ Test wit Neon and Carbon """
    m_ne, _ = ame.get_ion_mass('20Ne10+')
    omc = omegac(10, m_ne, 7.002)
    omz = 477e3*2*np.pi
    omp = omegap(omc, omz)
    T = ufloat(3.5, 2)
    z = zamp_Tz(T, omz, m_ne)
    p = rhop_Tz(T, omp, omz, m_ne)
    print("z:", np.around(unp.nominal_values(z)*1e6, 3), "um", z)
    print("p um:", np.around(unp.nominal_values(p)*1e6, 3), "um", p)
    m_c, _ = ame.get_ion_mass('12C6+')
    z = zamp_Tz(ufloat(7, 2), omz, m_c)
    print("should be 10.2um:", np.around(unp.nominal_values(z)*1e6, 3), "um", z)

    """ Test with Rb values """
    om_c = omegac(1, 85, B=6)
    om_z = omegaz(1, 85, -25, c2=-1400)
    om_p = omegap( om_c, om_z )
    om_m = omegam( om_c, om_z )
    print('trap frequencies')
    print('c,p,z,m')
    print(om_c/2/np.pi, om_p/2/np.pi, om_z/2/np.pi, om_m/2/np.pi)

    om_c = omegac(1, 40, B=6)
    om_z = omegaz(1, 40, -25, c2=-1400)
    om_p = omegap( om_c, om_z )
    om_m = omegam( om_c, om_z )
    print('trap frequencies')
    print('c,p,z,m')
    print(om_c/2/np.pi, om_p/2/np.pi, om_z/2/np.pi, om_m/2/np.pi)


    """ Test with Rhenium values """
    om_c = omegac(29, 187)
    om_z = omegaz(29, 187, -22.5105)
    om_p = omegap( om_c, om_z )
    om_m = omegam( om_c, om_z )

    print('trap frequencies')
    print('c,p,z,m')
    print(om_c/2/np.pi, om_p/2/np.pi, om_z/2/np.pi, om_m/2/np.pi)

    print('measured radii p,z,m')
    print( 2e-6, 11e-6, 2e-6)
    En_p = Ep(187, 2e-6, om_p, om_m)
    En_m = Em(187, 2e-6, om_p, om_m)
    En_z = Ez(187, 11e-6, om_z)

    print('energies by radius (and frequencies and mass)')
    print('p,z,m')
    print( En_p, En_z, En_m)

    En_z_thermal = E_thermal_1dim_osci(13.75)
    T_p = T_sideband(13.75, om_z, om_p)
    T_m = T_sideband(13.75, om_z, om_m)
    #print('T_p, T_m', T_p, T_m)
    En_p_thermal = E_thermal_1dim_osci( T_p )
    En_m_thermal = -E_thermal_1dim_osci( T_m ) # negative!!!

    print('energies by temperature / sideband cooling temperature')
    print('p,z,m')
    print( En_p_thermal, En_z_thermal, En_m_thermal)

    amp_z_thermal = zamp(En_z_thermal, om_z, 187)
    roh_p_thermal = rhop(En_p_thermal, om_p, om_m, 187)
    roh_m_thermal = rhom(En_m_thermal, om_p, om_m, 187)

    print('radii by thermal energy')
    print('p,z,m')
    print( roh_p_thermal, amp_z_thermal, roh_m_thermal)

    amp_z = zamp(En_z, om_z, 187)
    roh_m = rhom(En_m, om_p, om_m, 187)
    roh_p = rhop(En_p, om_p, om_m, 187)

    print('radii by radii (over energy), this is just a circle basically, should result in initial radii')
    print('p,z,m')
    print( roh_p, amp_z, roh_m)

