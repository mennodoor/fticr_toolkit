from doctest import Example
import numpy as np
from numpy.random import rand, exponential, normal
import uncertainties
from uncertainties import ufloat
from uncertainties import unumpy as unp
from uncertainties import umath
from copy import deepcopy
from inspect import getmembers, isfunction
from pprint import pprint

e = 1.6021766208E-19
u = 1.660538921E-27
em = 5.4857990907E-4 #9.10938358E-31
pi2 = np.pi * 2

from scipy import constants as cont

''' THIS MODULE

This module provides the means to easily apply shifts to 'simulated' ions/ratios and estimate the systematic effects 

TODO TODO TODO:
there are some major issues here regarding statistics due to the N times phases and radii, there are means and uncertainties of means or single values
which dont make sense AAAHHHHHHHH I hate this code

TODO: make it so i can put in measurement data and correct it directly. Actually thats kind of directly doable, maybe for
      some efffects (offresonant nuz effect) it could be made easier (?)

TODO:
What you also can do with this class is 'simulate' (more like generate in a statistical manner):
1) measurement data to test your analysis
2) phase jitters for different parameter scans and stabilities and so on
3) show dependenties of systematic ratio uncertainties to specific parameters

These functions are not immediately build in here (because that would be messy), you can find these
in a lot of jupyter notebooks of this package with additional informations :)

TODO:
fix relativistic shift for dR,
fix/find solution for linked radii error

'''


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

amu = cont.physical_constants["atomic mass unit-kilogram relationship"][0]
e = cont.e
eeV = cont.physical_constants["electron mass energy equivalent"][0]
m_e = cont.m_e
ueV = cont.physical_constants["electron volt-atomic mass unit relationship"][0]
dueV =   cont.physical_constants["electron volt-atomic mass unit relationship"][2]
ueV = ufloat(ueV, dueV)
kb = cont.k

from fticr_toolkit import ame
import fticr_toolkit.ideal_trap_physics as itp
import fticr_toolkit.systematics as sys


example_trap2 = {
    'idx': 2,
    'trap_radius': ufloat(5e-3, 5e-6), # trap radius, 1 um mechanical tolerance (probably worse?)
    #'nu_res': ufloat(477000.7, 10.0),
    #'nu_res': ufloat(697383.4, 1.0),
    'nu_res': ufloat(736074, 2.0),
    #'nu_res': ufloat(1e6, 1.0),
    'Q_res': ufloat(3500, 300.0),
    'L_res': ufloat(1.5e-3, 0.1e-3),
    'd_eff_res': ufloat(11e-3, 1e-3),
    'Temp_res': ufloat(4.5, 2.0),
    'B0': ufloat(7.002154, 1e-06),
    'B1': ufloat(1.41e-03, 0.27e-03), # T/m
    #'B2': ufloat(0.064, 0.005), # T/m**2
    #'B2': ufloat(0.064, 0.005), # T/m**2
    'B2': ufloat(0.028, 0.002), # T/m**2
    #'B2': ufloat(0.024, 0.005), # T/m**2
    'B3': ufloat(0, 0),
    'B4': ufloat(0, 0),
    'C1': ufloat(0, 0),
    'C2': ufloat(-1.488576e4, 0.1), ## NOTE: all  # 1/m**2
    'd2': ufloat(-2.5, 124),
    'C3': ufloat(0, 0),
    #'d4': ufloat(0.8878e-3, 2.6e-06), # 1/mm**4
    'd4': ufloat(0.8878e-3, 1e-05)*1e12, # 1/mm**4 *1e12 to convert to 1/m**4
    #'d6': ufloat(-6.1e-05, 4e-06), # 1/mm**6
    'd6': ufloat(-6.1e-05, 1e-05)*1e18, #  1/mm**6 *1e18 to convert to 1/m**6
    #'TR_opt4': ufloat(0.880143, 4e-06),
    'TR_opt4': ufloat(0.880143, 1e-03),
    #'TR_opt6': ufloat(0.8786, 0.00013),
    'TR_opt6': ufloat(0.8786, 1e-03),
    'TR_set': 0.87966, # NOTE: BAD TR!!!!
    'epsilon': 0, #0.015, # 0.05
    'theta': 0, #0.5/180*np.pi, # 1
    'phi': 0, #1e-3/180*np.pi,
    'non_lin1': 0.25, # bana first order in degree, amplitude of sinus (phase readout depends on phase of ion),
    'non_lin2': 0.0, # bana second order (phase readout has accumulation time depended offset (so ref phase substraction does not apply properly),
    'dnuz_per_dnures': ufloat(20, 170)*1e-6, #  Hz/Hz , determined from scanning the resonator below a dip. For dip lineshape sys
    'trap_offsets': [0, 0, 0, 0, 0],
    'Bdrift_per_hour': ufloat(-1.0, 0.1)*1e-9, # relative drift
    'dBB_per_second': ufloat(3, 0.5)*1e-11, # relative jitter
    'dUU_per_second': ufloat(1, 0.2)*1e-7, # relative jitter (allan dev at 1s)
}


example_trap3 = {
    'idx': 3,
    'trap_radius': ufloat(5e-3, 5e-6), # trap radius, 1 um mechanical tolerance (probably worse?)
    #'nu_res': ufloat(477104.3, 1.0),
    'nu_res': ufloat(501493, 2.0),
    #'nu_res': ufloat(707000.7, 2.0),
    'Q_res': ufloat(10000, 300.0),
    'L_res': ufloat(1.5e-3, 0.1e-3),
    'd_eff_res': ufloat(11e-3, 1e-3),
    'Temp_res': ufloat(10.0, 2.0),
    'B0': ufloat(7.002163, 5e-6),
    'B1': ufloat(-1.41e-03, 0.27e-03), # T/m
    #'B2': ufloat(0.064, 0.005), # T/m**2
    #'B2': ufloat(0.064, 0.005), # T/m**2
    #'B2': ufloat(0.024, 0.005), # T/m**2
    'B2': ufloat(-0.005, 0.002), # T/m**2
    'B3': ufloat(0, 0),
    'B4': ufloat(0, 0),
    'C1': ufloat(0, 0),
    'C2': ufloat(-1.489708e4, 1), ## NOTE: all  # 1/m**2
    'd2': ufloat(-2.5, 124),
    'C3': ufloat(0, 0),
    #'d4': ufloat(0.8878e-3, 2.6e-06), # 1/mm**4
    'd4': ufloat(0.8878e-3, 1e-05)*1e12, # 1/mm**4 *1e12 to convert to 1/m**4
    #'d6': ufloat(-6.1e-05, 4e-06), # 1/mm**6
    'd6': ufloat(-6.1e-05, 1e-05)*1e18, #  1/mm**6 *1e18 to convert to 1/m**6
    #'TR_opt4': ufloat(0.880143, 4e-06),
    'TR_opt4': ufloat(0.880143, 1e-03),
    #'TR_opt6': ufloat(0.8786, 0.00013),
    'TR_opt6': ufloat(0.8786, 0.001),
    'TR_set': 0.879002+0e-3, # NOTE: BAD TR!
    'epsilon': 0.015, # 0.05
    'theta': 0.5/180*np.pi, # 1
    'phi': 1e-3/180*np.pi,
    'non_lin1': 0.25, # bana first order in degree, amplitude of sinus (phase readout depends on phase of ion),
    'non_lin2': 0.0, # bana second order (phase readout has accumulation time depended offset (so ref phase substraction does not apply properly),
    'dnuz_per_dnures': ufloat(1650, 130)*1e-6, # Hz/Hz , determined from scanning the resonator below a dip. For dip lineshape sys
    'trap_offsets': [0, 0, 0, 0, 0],
    'Bdrift_per_hour': ufloat(-1.0, 0.1)*1e-9, # relative drift
    'dBB_per_second': ufloat(3, 0.5)*1e-11, # relative jitter
    'dUU_per_second': ufloat(1, 0.2)*1e-7, # relative jitter (allan dev at 1s)
}



class Ion():
    """_summary_
    """
    def __init__(self, trap, ion, nuc=None):
        self.ion = ion
        self.trap = trap

        self.A, self.el, self.q = itp.re_ionstr(ion)
        self.m = self.set_mass()
        self.qm_si = self.q*e / (self.m*ukg)

        self.enable_m_shifts = True
        self.enable_z_shifts = True
        self.enable_p_shifts = True
        self.enable_c_shifts = True

        self.fixed_radii = False
        self.rpf, self.azf, self.rmf = 0, 0, 0

        if nuc is not None:
            trap["B0"] = self.itp.B(self.q, self.m, nuc*pi2)

        self.ICS_error = None
        self.off_res = ufloat(0, 3) # Hz
        self.dnuz_per_dnures = ufloat(1.65, 0.13)*1e-3 # Hz/Hz

        self.init()


    def init(self, set_U0_by_nures=True, zero_shifts=True, init_coordinates=True, dU=0):
        '''
        In case you change the B0 or nu_res value afterwards.
        '''
        if set_U0_by_nures: self.set_U0(nuz=self.trap['nu_res'], dU=dU)
        self.calc_TR_dependent()
        self.calc_eigenfrequencies()
        if zero_shifts: self.zero_domega()
        if init_coordinates: self.init_coordinates(Tz=self.trap['Temp_res'].n)

    def __str__(self):
        return self.ion + ' in trap ' + self.trap['idx']

    def _uval(self, key='m'):
        uval = ufloat(getattr(self, key), getattr(self, 'd'+key))
        setattr(self, key, uval)

    def _nval(self, key='m'):
        nval = getattr(self, key).n
        sval = getattr(self, key).s
        setattr(self, key, nval)
        setattr(self, 'd'+key, sval)

    def set_mass(self, binding=None, dm=0, dmeV=False):
        """set the ions mass, using ame module, add dm in amu

        Args:
            binding (float, optional): binding energy of missing electrons in eV. Defaults to None.
            dm (float, optional): mass offset in amu. Defaults to None.
        """
        if dmeV:
            dmu = dm*ueV.n
            dme = dm
        else:
            dmu = dm
            dme = dm/ueV.n
        print(dm, dmu, dme, binding, dmeV)
    
        self.m, self.dm = ame.get_ion_mass(self.ion, binding)
        self.m_neutral, self.dm_neutral = ame.get_iso_mass(self.A, self.el)
        self.m_neutral_eV, self.dm_neutral_eV = ame.get_iso_mass_excess(self.A, self.el)
        self.m += dmu
        print(self.m)
        self.m_neutral += dmu
        self.m_neutral_eV += dme
        return self.m

    def set_U0(self, nuz=None, U0=None, dU=0, delU=None):
        print(self.m)
        if nuz is not None:
            self.U0 = itp.U0(self.q, self.m, nuz, self.trap['C2'])
        elif U0 is not None:
            self.U0 = U0
        else:
            raise ValueError('Needs either nuz or U0 as input')
        try:
            self.U0 = unp.nominal_values(self.U0)
        except:
            pass
        self.U0 += dU
        if delU is not None:
            self.U0 = ufloat(self.U0, delU)

        return self.U0

    def calc_TR_dependent(self):
        TRset = self.trap['TR_set']
        self.trap['C4'] = (self.trap['TR_opt4'] - TRset) * self.trap['d4']
        self.trap['C6'] = (self.trap['TR_opt6'] - TRset) * self.trap['d6']
        #print('C2/4/6', self.trap['C2'], self.trap['C4'], self.trap['C6'])

    def no_Ci_errors(self):
        for i in range(10):
            try:
                self.trap['C'+str(i)] = self.trap['C'+str(i)].n
            except:
                pass

    def no_Bi_errors(self):
        for i in range(10):
            try:
                self.trap['B'+str(i)] = self.trap['B'+str(i)].n
            except:
                pass

    def calc_eigenfrequencies(self):
        B0 = self._nominal(self.trap['B0'])

        self.omegac = itp.omegac(self.q, self.m, B0)
        self.omegaz = itp.omegaz(self.q, self.m, self.U0, self.trap['C2'].n)
        self.omegap = itp.omegap(self.omegac, self.omegaz)
        self.omegam = itp.omegam(self.omegac, self.omegaz)
        #print( self.omegap, self.omegaz, self.omegam )
        return self.omegap, self.omegaz, self.omegam

    def zero_domega(self):
        self.domegac, self.domegap, self.domegaz, self.domegam = 0, 0, 0, 0

        self.omegac_c = self.omegac
        self.omegap_c = self.omegap
        self.omegaz_c = self.omegaz
        self.omegam_c = self.omegam
        return None

    def _nominal(self, values):
        if isinstance(values, uncertainties.UFloat):
            values = values.n
        elif isinstance(values, np.ndarray):
            try:
                values = unp.nominal_values(values)
            except:
                try:
                    values = values.n
                except:
                    pass
        return values

    def _nominal_mean(self, values):
        return np.mean(self._nominal(values))

    def get_frequencies(self, mean=False):
        if mean:
            return self._nominal_mean(self.omegac), self._nominal_mean(self.omegap), self._nominal_mean(self.omegaz), self._nominal_mean(self.omegam)
        return self.omegac, self.omegap, self.omegaz, self.omegam

    def get_nu_frequencies(self, mean=False):
        if mean:
            return self._nominal_mean(self.omegac/pi2), self._nominal_mean(self.omegap/pi2), self._nominal_mean(self.omegaz/pi2), self._nominal_mean(self.omegam/pi2)
        return self.omegac/pi2, self.omegap/pi2, self.omegaz/pi2, self.omegam/pi2

    def get_corrected_frequencies(self):
        return self.omegac_c, self.omegap_c, self.omegaz_c, self.omegam_c

    def get_corrected_nu_frequencies(self):
        return self.omegac_c/pi2, self.omegap_c/pi2, self.omegaz_c/pi2, self.omegam_c/pi2

    def get_invariance_omega_c(self, corrected=True):
        if corrected:
            omegac_inv = itp.omegac_invariance(self.omegap_c, self.omegaz_c, self.omegam_c)
        else:
            omegac_inv = itp.omegac_invariance(self.omegap, self.omegaz, self.omegam)
        return omegac_inv

    def show_frequencies(self, nu=True, corrected=True):
        if corrected and nu:
            print(self.get_corrected_nu_frequencies())
        elif not corrected and nu:
            print(self.get_nu_frequencies())
        elif corrected and not nu:
            print(self.get_corrected_frequencies())
        else:
            print(self.get_frequencies())

    def show_radii(self):
        print("radii +, z, -", self.get_radii())

    def compare_omegac_and_invariance(self):
        omegac_inv = itp.omegac_invariance(self.omegap_c, self.omegaz_c, self.omegam_c)
        print('invariance', omegac_inv)
        print('direct    ', self.omegac_c)
        print('inv - direct', omegac_inv - self.omegac_c)

    def get_radii(self, mean=True):
        if self.fixed_radii:
            return self.rpf, self.azf, self.rmf

        if mean:
            if isinstance(mean, str) and mean == 'std':
                rhop = ufloat(np.mean(unp.nominal_values(self.rcp)), np.std(unp.nominal_values(self.rcp)))
                zamp = ufloat(np.mean(unp.nominal_values(self.acz)), np.std(unp.nominal_values(self.acz)))
                rhom = ufloat(np.mean(unp.nominal_values(self.rcm)), np.std(unp.nominal_values(self.rcm)))
            else: # error of the mean
                rhop = ufloat(np.mean(unp.nominal_values(self.rcp)), np.std(unp.nominal_values(self.rcp), ddof=1) / np.sqrt(np.size(self.rcp)))
                zamp = ufloat(np.mean(unp.nominal_values(self.acz)), np.std(unp.nominal_values(self.acz), ddof=1) / np.sqrt(np.size(self.acz)))
                rhom = ufloat(np.mean(unp.nominal_values(self.rcm)), np.std(unp.nominal_values(self.rcm), ddof=1) / np.sqrt(np.size(self.rcm)))
            
        else:
            rhop = self.rcp
            zamp = self.acz
            rhom = self.rcm

        #print(rhop, zamp, rhom)
        return rhop, zamp, rhom

    def fix_radii(self, rp=None, az=None, rm=None, mean="std"):
        mrp, maz, mrm = self.get_radii(mean=mean)
        self.fixed_radii = True
        if rp is None:
            self.rpf = mrp
        else:
            self.rpf = rp
        if az is None:
            self.azf = maz
        else:
            self.azf = az
        if rm is None:
            self.rmf = mrm
        else:
            self.rmf = rm

    def unfix_radii(self):
        self.fix_radii = False

    def init_coordinates(self, N=100e3, Tz=None, Tp=None, Tm=None):
        self.N = int(N)
        N = int(N)
        _, omegap, omegaz, omegam = self.get_frequencies(mean=True)

        if Tz is None:
            Tz = self.trap['Temp_res'].n
        if Tp is None:
            Tp = itp.T_sideband(Tz, omegaz, omegap)
        if Tm is None:
            Tm = itp.T_sideband(Tz, omegaz, omegam)

        E0z = exponential(kb*Tz, self.N)

        E0p = exponential(kb*Tp, self.N)
        #E0p = E0z*omegap/omegaz
        #print(np.mean(E0z)*omegap/omegaz, np.mean(E0p))
        self.r0p = np.sqrt(2*E0p/amu/self.m/(omegap)**2)

        self.rcp = deepcopy(self.r0p)
        print(kb, Tz)
        rp_cross = np.sqrt(kb*Tz/(self.m*amu*2*np.pi**2*self.omegap/2/np.pi*self.omegaz/2/np.pi))
        print("cross_check radii nup, mean/median(sampled), kbT-calc", self.ion, np.mean(self.r0p), np.median(self.r0p), rp_cross)
        #from matplotlib import pyplot as plt
        #plt.hist(self.r0p, bins=100)
        #plt.vlines([np.mean(self.r0p)], [0], [1000])
        #plt.show()
        print('mean p vs p itp', np.mean(self.r0p), itp.rhop_Tz(Tz, omegap, omegaz, self.m))
        #print('mean E0p vs kbT', np.mean(E0p), kb*Tp)

        E0m = exponential(kb*Tm, self.N)
        self.r0m = np.sqrt(4*E0m/amu/self.m/(omegaz)**2)
        self.rcm = deepcopy(self.r0m)

        E0z = exponential(kb*Tz, self.N)
        self.a0z = np.sqrt(2*E0z/amu/self.m/(omegaz)**2)
        self.acz = deepcopy(self.a0z)
        #print('mean z vs z itp', np.mean(self.a0z), itp.zamp_Tz(Tz, omegaz, self.m))

        self.phi0p=rand(N)
        self.phicp=deepcopy(self.phi0p)
        self.phi0m=rand(N)
        self.phicm=deepcopy(self.phi0m)
        self.phi0z=rand(N)
        self.phicz=deepcopy(self.phi0z)

        return self.get_radii(mean=True)
    
    def _excite(self, r, phi, exc_radius):
        
        if isinstance(exc_radius, uncertainties.UFloat):
            exc_radius = np.random.randn(self.N) * exc_radius.s + exc_radius.n
        x0 = r*unp.cos(phi*2*np.pi) + exc_radius
        y0 = r*unp.sin(phi*2*np.pi)
        r_exc = unp.sqrt(x0**2 + y0**2)
        phi_exc = unp.arctan2( y0, x0 )
        return r_exc, phi_exc

    def excite_p(self, exc_radius=20e-6):
        self.rcp, self.phicp = self._excite(self.rcp, self.phicp, exc_radius)
    
    def excite_m(self, exc_radius=20e-6):
        self.rcm, self.phicm = self._excite(self.rcm, self.phicm, exc_radius)

    def excite_z(self, exc_amplitude=100e-6):
        self.acz, self.phicz = self._excite(self.acz, self.phicz, exc_amplitude)

    def adjust_domega_sum(self, domegac, domegap, domegaz, domegam):
        if self.enable_c_shifts: self.domegac += domegac
        if self.enable_p_shifts: self.domegap += domegap
        if self.enable_z_shifts: self.domegaz += domegaz
        if self.enable_m_shifts: self.domegam += domegam

    def adjust_omega_current_by_domega(self):
        if self.enable_c_shifts: self.omegac_c = self.omegac + self.domegac
        if self.enable_p_shifts: self.omegap_c = self.omegap + self.domegap
        if self.enable_z_shifts: self.omegaz_c = self.omegaz + self.domegaz
        if self.enable_m_shifts: self.omegam_c = self.omegam + self.domegam

    def show_current_shifts(self):
        print('shifts:\t dnu_c \tdnu_p \tdnu_z \tdnu_m')
        print('abs.  :\t', self.domegac/pi2, self.domegap/pi2, self.domegaz/pi2, self.domegam/pi2)
        print('rel.  :\t', self.domegac/self.omegac, self.domegap/self.omegap, self.domegaz/self.omegaz, self.domegam/self.omegam)

    def apply_relativistic_shift(self, mean=True):
        rhop, zamp, rhom = self.get_radii(mean=mean)

        domegac = sys.domegac_rel(rhop, self.omegac, self.omegam, self.omegap)
        domegap = sys.domegap_rel(rhop, zamp, rhom, self.omegap, self.omegaz, self.omegam)
        domegaz = sys.domegaz_rel(rhop, zamp, rhom, self.omegap, self.omegaz, self.omegam)
        domegam = sys.domegam_rel(rhop, zamp, rhom, self.omegap, self.omegaz, self.omegam)
        print("rel omegam", self.ion, domegam)
        self.adjust_domega_sum(domegac, domegap, domegaz, domegam)
        self.adjust_omega_current_by_domega()
        return domegac, domegap, domegaz, domegam

    def apply_image_charge_shift(self, mean=True):
        
        domegam = sys.domegam_ICS(self.q, self.m, self.trap['trap_radius'], self.omegac, relative_error=self.ICS_error)
        domegap = sys.domegap_ICS(self.q, self.m, self.trap['trap_radius'], self.omegac, relative_error=self.ICS_error)
        domegac = sys.domegac_ICS(self.q, self.m, self.trap['trap_radius'], self.omegac, self.omegap, self.omegam, relative_error=self.ICS_error)
        self.ics_domegam = domegam
        print(self.ion, domegac, domegap, domegam)
        print("ICS omegam", self.ion, domegam)
        self.adjust_domega_sum(domegac, domegap, 0, domegam)
        self.adjust_omega_current_by_domega()
        return domegac, domegap, 0, domegam

    def apply_image_current_shift(self, mean=True, off_res=None):
        Q = self.trap['Q_res']
        L = self.trap['L_res']
        d_eff = self.trap['d_eff_res']
        omega_res = self.trap['nu_res']*pi2
        if off_res is not None:
            omega_res = (self.omegaz/pi2 - off_res)*pi2
        #print(self.omegaz/pi2, omega_res/pi2)

        domegaz = sys.domegaz_freqpull(self.omegaz, omega_res, q=self.q, m=self.m, Q=Q, L=L, d_eff=d_eff)
        self.adjust_domega_sum(0, 0, domegaz, 0)
        self.adjust_omega_current_by_domega()

        #print(domegaz/pi2)
        return 0, 0, domegaz, 0

    def apply_offresonant_dip_fit_shift(self, mean=True, off_res=0, off_res_shift_per_q=0.030):
        """This is a systematic cause by the fit model. You can test this by scanning fixed
        resonator frequencies and leave the other parameters of the fit free. You will get,
        depended on Q and dip width, a linear dependency of the fit result for the axial frequency
        over the used resonator frequency. This is a systematic offset of your "measured" axial
        frequency. You dont know what the right resonator frequency is, but you can tune this 
        out if you use (for same dip width/Q) the same nu_res-nu_z value, resulting in the same
        offset of the axial frequency. How fine you have to tune this depends on your ratio and
        the given impacts of the axial frequencies in the invariance theorem.
        For vastly different Q and/or dip width, you should scan for all cases and estimate
        the errors individually.
        """
        #domegaz = off_res*(off_res_shift_per_q/self.q)
        domegaz = sys.domegaz_dnures(self.dnuz_per_dnures, self.off_res) * 2*np.pi
        self.adjust_domega_sum(0, 0, domegaz, 0)
        self.adjust_omega_current_by_domega()
        return 0, 0, domegaz, 0

    def apply_B2_shift(self, mean=True, B2=None, B0=None):
        if B2 is None:
            B2 = self.trap['B2']
        if B0 is None:
            B0 = self.trap['B0']
        
        rhop, zamp, rhom = self.get_radii(mean=mean)

        domegac = sys.domegac_b2(rhop, rhom, zamp, self.omegap, self.omegam, B2=B2, B0=B0)
        domegap = sys.domegap_b2(rhop, rhom, zamp, self.omegap, self.omegam, B2=B2, B0=B0)
        domegam = sys.domegam_b2(rhop, rhom, zamp, self.omegap, self.omegam, B2=B2, B0=B0)
        domegaz = sys.domegaz_b2(rhop, rhom, self.omegaz, self.omegap, self.omegam, B2=B2, B0=B0)
        print(self.ion, "B2", domegac, domegap, domegaz, domegam)
        print("B2 omegam", self.ion, domegam)
        self.adjust_domega_sum(domegac, domegap, domegaz, domegam)
        self.adjust_omega_current_by_domega()
        return domegac, domegap, domegaz, domegam

    def apply_C3_shift(self, mean=True, c3=None, c2=None):
        if c3 is None:
            c3 = self.trap['C3']
        if c2 is None:
            c2 = self.trap['C2']

        _, zamp, _ = self.get_radii(mean=mean)

        domegaz = sys.domegaz_c3(zamp, self.omegaz, c3, c2)
        self.adjust_domega_sum(0, 0, domegaz, 0)
        self.adjust_omega_current_by_domega()
        return 0, 0, domegaz, 0

    def apply_C4_shift(self, mean=True, c4=None, c2=None):
        if c4 is None:
            c4 = self.trap['C4']
        if c2 is None:
            c2 = self.trap['C2']
        print("C42", c4, c2)

        rhop, zamp, rhom = self.get_radii(mean=mean)
        #print("radii", rhop, zamp, rhom)

        domegac = sys.domegac_c4(rhop, rhom, self.omegap, self.omegam, c4, c2)
        domegap = sys.domegap_c4(zamp, rhop, rhom, self.omegap, self.omegam, c4, c2)
        domegaz = sys.domegaz_c4(zamp, rhop, rhom, self.omegaz, c4, c2)
        domegam = sys.domegam_c4(zamp, rhop, rhom, self.omegap, self.omegam, c4, c2)
        print("C4 omegam", self.ion, domegam)
        self.adjust_domega_sum(domegac, domegap, domegaz, domegam)
        self.adjust_omega_current_by_domega()
        return domegac, domegap, domegaz, domegam

    def apply_C6_shift(self, mean=True, c6=None, c2=None):
        if c6 is None:
            c6 = self.trap['C6']
        if c2 is None:
            c2 = self.trap['C2']
        print("C62", c6, c2)

        rhop, zamp, rhom = self.get_radii(mean=mean)
        #print("radii", rhop, zamp, rhom)

        domegac = sys.domegac_c6(zamp, rhop, rhom, self.omegap, self.omegam, c6, c2)
        domegap = sys.domegap_c6(zamp, rhop, rhom, self.omegap, self.omegam, c6, c2)
        domegaz = sys.domegaz_c6(zamp, rhop, rhom, self.omegaz, c6, c2)
        domegam = sys.domegam_c6(zamp, rhop, rhom, self.omegap, self.omegam, c6, c2)
        print("C6 omegam", self.ion, domegam)
        self.adjust_domega_sum(domegac, domegap, domegaz, domegam)
        self.adjust_omega_current_by_domega()
        return domegac, domegap, domegaz, domegam

    def apply_B1C3_shift(self, mean=True):
        pass

    def apply_num_shift(self, mean=True, value=0):

        domegac = 0
        domegap = 0
        domegaz = 0
        domegam = value
        self.adjust_domega_sum(domegac, domegap, domegaz, domegam)
        self.adjust_omega_current_by_domega()
        return domegac, domegap, domegaz, domegam

    def apply_nuz_shift(self, mean=True, value=0):

        domegac = 0
        domegap = 0
        domegaz = value
        domegam = 0
        self.adjust_domega_sum(domegac, domegap, domegaz, domegam)
        self.adjust_omega_current_by_domega()
        return domegac, domegap, domegaz, domegam

    def apply_num_shift(self, mean=True, value=0):

        domegac = 0
        domegap = 0
        domegaz = 0
        domegam = value
        self.adjust_domega_sum(domegac, domegap, domegaz, domegam)
        self.adjust_omega_current_by_domega()
        return domegac, domegap, domegaz, domegam

    def apply_tiltellip_shift(self, mean=True, epsilon=None, theta=None, phi=None):
        if epsilon is None:
            epsilon = self.trap['epsilon']
        if theta is None:
            theta = self.trap['theta']
        if phi is None:
            phi = self.trap['phi']

        domegap = sys.domegap_tilt_ellip(self.omegap, self.omegaz, self.omegam, epsilon=epsilon, theta=theta, phi=phi)
        domegaz = sys.domegaz_tilt_ellip(self.omegaz, epsilon=epsilon, theta=theta, phi=phi)
        domegam = sys.domegam_tilt_ellip(self.omegap, self.omegaz, self.omegam, epsilon=epsilon, theta=theta, phi=phi)
        print("tilt omegam", self.ion, domegam)
        domegac = self.get_invariance_omega_c() - self.omegac_c
        self.adjust_domega_sum(domegac, domegap, domegaz, domegam)
        self.adjust_omega_current_by_domega()
        return domegac, domegap, domegaz, domegam

    def all_frequency_shifts(self, mean=True, callback=print):
        self.zero_domega()

        functions = getmembers(Ion, isfunction)
        #print(functions)
        for name, fun in functions:
            #print(name)
            if name.startswith('apply_') and name.endswith("_shift"):
                shifts = getattr(self, name)(mean=mean)
                shifts = np.asarray(shifts)
                if callback is not None:
                    try:
                        shifts /= pi2
                        callback(name, *shifts)
                    except:
                        callback(name, shifts)
                #self.show_current_shifts()

class Ratio():
    def __init__(self, ionA, ionB, trap, sameU=False, Tfactor=1.0, dU=0, dmBeV=0, trapconf_ddres=True):
        '''
        ionA and ionB can either be string, e.g. '172Yb42+' or directly of Ion-class type
        trap has to be a dictionary with all the needed parameters

        There are (kind of) three different ways to calculate the shift due to some effect on the ratio:
        1) You find a formular for calculating the shift on the ratio directly. (dR)
        2) You find a formular for calculating the shift on the free cyclotron directly, do the ratio (dR2)
        3) You apply the shifts to all eigenfrequencies individually, do the invariance, do the ratio (dR3)
        All these shifts should be equal in the end (except see below). This can be a nice way of checking
        your equations if you get them from different sources or just for typos. 

        dR3 is the most flexible in this class because sometimes you just want to see the effect of a shift
        of a single eigenfrequency because for example you measure the eigenfrequencies at different 'ion
        situations', e.g. you measure nu+ with small excitation in rho+ but cool in nuz and nu- and then
        you measure nuz also phase sensitive with higher excitation and cool rho+ and rho-. In that case
        C4 shifts for example would turn out much different for dR3 than for dR or dR2 in the nuz situation
        (since C4 shifts in nuc and R are only rho+/- depended).

        dR is much easier to work with if you want to link parameter sizes and errors together using the 
        ufloat package, e.g. link radii together so that the excitation radius for ion A (in the mean) 
        is the same as for ionB on an error level of 0.5% -> exc_radB = exc_radA * ufloat(1, 0.005). 

        '''
        if isinstance(ionA, Ion):
            self.ionA = ionA
        else:
            self.ionA = Ion(trap, ionA)
        if isinstance(ionB, Ion):
            self.ionB = ionB
        else:
            self.ionB = Ion(trap, ionB)

        print("default radii")
        self.show_radii()

        if sameU:
            self.ionB.set_U0(U0=self.ionA.U0, dU=dU)
            self.ionB.calc_eigenfrequencies()
            self.ionB.zero_domega()
            self.ionB.init_coordinates()
            print("sameU", sameU, "so resonators are tuned. Difference in nu_z is", (self.ionA.omegaz-self.ionB.omegaz)/2/pi2)
        
        if Tfactor or dmBeV:
            self.ionB.trap = deepcopy(self.ionA.trap)
            self.ionB.trap["Temp_res"] = self.ionA.trap["Temp_res"] * Tfactor
            self.ionB.set_mass(0, dmBeV, dmeV=True)
            self.ionB.calc_eigenfrequencies()
            self.ionB.zero_domega()
            self.ionB.init_coordinates()
            print("Tfactor", Tfactor, "radii:")
            self.show_radii()

        if trapconf_ddres:
            self.ionA.dnuz_per_dnures = trap["dnuz_per_dnures"]
            self.ionB.dnuz_per_dnures = trap["dnuz_per_dnures"]

        self.trap = trap

        self.radii_link_quality = ufloat(1, 0.005)
        
        self.ICS_error = None

        self.init()
        
        self.disabled = []
        self.reset_freq_shifts = True

    def init(self):
        #.we need the initial ion cyclotron ratio (ion ratio), neutral mass ratio, mass differences, 
        self.R = self.ionB.omegac/self.ionA.omegac # mA_ion / mB_ion
        self.R_neutral = self.ionA.m_neutral/self.ionB.m_neutral
        self.mdiff = self.ionA.m - self.ionB.m # A-B
        self.mdiff_eV = self.mdiff/ueV # TODO: there is probably a better way with smaller error, no time for that now.
        self.mdiff_neutral = self.ionA.m_neutral - self.ionB.m_neutral
        self.mdiff_neutral_eV = self.mdiff_neutral*ueV

        self.zero_dR()

    def zero_dR(self):
        self.dR = 0
        self.dR2 = 0 # this is just as a cross check
        self.R_c = self.R
        self.remove_ion_shifts()
        return None

    def manual_domegac(self, domegac_A=None, domegac_B=None):
        if domegac_A is not None and domegac_B is None:
            dR = self.ionB.omegac/(self.ionA.omegac + domegac_A) - self.R
        if domegac_B is not None and domegac_A is None:
            dR = domegac_B/self.ionA.omegac

        dR2 = 0
        self.adjust_dR_sum(dR, dR2)
        self.adjust_R_current_by_dR()
        return dR, dR2

    def show_radii(self):
        rpA, azA, rmA = self.ionA.get_radii(mean=True)
        print('A', rpA, azA, rmA)
        rpB, azB, rmB = self.ionB.get_radii(mean=True)
        print('B', rpB, azB, rmB)

    def remove_ion_shifts(self):
        self.ionA.zero_domega()
        self.ionB.zero_domega()

    def init_ion_coordinates(self):
        self.ionA.init_coordinates()
        self.ionB.init_coordinates()

    def adjust_dR_sum(self, dR, dR2):
        self.dR += dR
        self.dR2 += dR2

    def adjust_R_current_by_dR(self):
        self.R_c = self.R + self.dR

    def show_current_shifts(self):
        print("\ncurrent ratio shifts:")
        print('shifts:  dR \t\tdR2 (on ratio, on freqs)')
        print('dR     ', self.dR, self.dR2)
        print('dR/R   ', self.dR/self.R, self.dR2/self.R)

    def show_ion_frequencies(self):
        print("\n ion frequencies")
        ocA, opA, ozA, omA = self.ionA.get_corrected_nu_frequencies()
        ocB, opB, ozB, omB = self.ionB.get_corrected_nu_frequencies()
        print("ionA c/p/z/m", ocA, opA, ozA, omA)
        print("ionB c/p/z/m", ocB, opB, ozB, omB)
        print("ionA - ionB ratio p/c", opA/ocA-opB/ocB, opA/ocA, opB/ocB)
        print("diffs A-B", ocA-ocB, opA-opB, ozA-ozB, omA-omB)
        print("ratio A/B", ocA/ocB, opA/opB, ozA/ozB, omA/omB)
        print("ratio A/B-1", ocA/ocB-1, opA/opB-1, ozA/ozB-1, omA/omB-1)
        print("ratio A/B^2-1", (ocA/ocB)**2-1, (opA/opB)**2-1, (ozA/ozB)**2-1, (omA/omB)**2-1)
        print("ratio A/B^4-1", (ocA/ocB)**4-1, (opA/opB)**4-1, (ozA/ozB)**4-1, (omA/omB)**4-1)
        print("ratio A/B-1^2", (ocA/ocB-1)**2, (opA/opB-1)**2, (ozA/ozB-1)**2, (omA/omB-1)**2)
        print("ratio diff A/B nuc/nuc-nup/nup", ocA/ocB - opA/opB)
        ocAo, opAo, ozAo, omAo = self.ionA.get_nu_frequencies()
        ocBo, opBo, ozBo, omBo = self.ionB.get_nu_frequencies()
        print("diffs original-corrected A", ocAo-ocA, opAo-opA, ozAo-ozA, omAo-omA)
        print("diffs original-corrected B", ocBo-ocB, opBo-opB, ozBo-ozB, omBo-omB)
        print("diffs num A-B, original/corrected", omAo-omBo, omA-omB,  (omAo-omBo) - (omA-omB) )
        try:
            domegam_diff_all = omA-omB
            domegam_ics = self.ics_diff_A_minus_B_omegam
            print("diffs num A-B, all/ics", domegam_diff_all, domegam_ics/2/np.pi )
        except:
            pass

    def apply_relativistic_shift(self, mean=True, correlate_radii=1e-2):
        if "relativistic_shift" in self.disabled:
            return 0, 0
        _, ompA, _, _ = self.ionA.get_frequencies()
        _, ompB, _, _ = self.ionB.get_frequencies()
        rpA, _, _ = self.ionA.get_radii(mean=mean)
        rpB, _, _ = self.ionB.get_radii(mean=mean)

        #if correlate_radii > 0 and mean:
        #    rpB = rpB.n/rpA.n * rpA * ufloat(1, correlate_radii)
        #    #print(rpA, rpB)

        dR = sys.R_rel(self.R, ompA, ompB, rpA, rpB) - self.R
        dR = sys.dR_rel3(self.R, ompA, ompB, rpA, rpB)
        dR = sys.dR_rel4(self.R, ompA, ompB, rpA, rpB)
        print('relativistic', sys.R_rel(self.R, ompA, ompB, rpA, rpB))
        print('relativistic', self.R)
        self.ionA.apply_relativistic_shift()
        self.ionB.apply_relativistic_shift()
        dR2 = self.ionB.get_invariance_omega_c() / self.ionA.get_invariance_omega_c() - self.R
        self.adjust_dR_sum(dR, dR2)
        self.adjust_R_current_by_dR()
        return dR, dR2

    def apply_image_charge_shift(self, mean=True):
        # this shift happens on the ideal ratio, so correction with opposite sign!
        dR = -sys.dR_ICS(self.R, self.trap['B0'], self.trap['trap_radius'], -self.mdiff_eV, relative_error = self.ICS_error)

        self.ionA.apply_image_charge_shift(mean=mean)
        self.ionB.apply_image_charge_shift(mean=mean)
        dR2 = self.ionB.get_invariance_omega_c() / self.ionA.get_invariance_omega_c() - self.R
        self.adjust_dR_sum(dR, dR2)
        self.adjust_R_current_by_dR()
        return dR, dR2 

    def apply_image_current_shift(self, mean=True, off_resA = ufloat(-0, 2), off_resB = ufloat(-0, 2)):
        # there is no direct version of this, no simple dR function
        self.ionA.apply_image_current_shift(mean=mean, off_res=off_resA)
        self.ionB.apply_image_current_shift(mean=mean, off_res=off_resB)
        self.ics_diff_A_minus_B_omegam = self.ionA.ics_domegam - self.ionB.ics_domegam
        dR2 = self.ionB.get_invariance_omega_c() / self.ionA.get_invariance_omega_c() - self.R
        self.adjust_dR_sum(dR2, dR2)
        self.adjust_R_current_by_dR()
        return dR2, dR2 

    def apply_offresonant_dip_fit_shift(self, mean=True) :
        # there is no direct version of this, no simple dR function

        ocA, opA, ozA, omA = self.ionA.get_nu_frequencies()
        ocB, opB, ozB, omB = self.ionB.get_nu_frequencies()
        dnuresA = self.ionA.off_res
        dnuresB = self.ionB.off_res
        dnuzperdnuresA = self.ionA.dnuz_per_dnures
        dnuzperdnuresB = self.ionB.dnuz_per_dnures
        dR = sys.dR_offresonator(self.R, dnuzperdnuresA, dnuzperdnuresB, dnuresA, dnuresB, ozA, ozB, ocA, ocB)
        self.ionA.apply_offresonant_dip_fit_shift(mean=mean)
        self.ionB.apply_offresonant_dip_fit_shift(mean=mean)
        dR2 = self.ionB.get_invariance_omega_c() / self.ionA.get_invariance_omega_c() - self.R
        self.adjust_dR_sum(dR, dR2)
        self.adjust_R_current_by_dR()
        return dR, dR2 

    def apply_B2_shift(self, mean=True, B2=None, B0=None, correlate_radii=1e-2):
        if B2 is None:
            B2 = self.trap['B2']
        if B0 is None:
            B0 = self.trap['B0']

        _, ompA, _, ommA = self.ionA.get_frequencies()
        _, ompB, _, ommB = self.ionB.get_frequencies()
        rpA, azA, rmA = self.ionA.get_radii(mean=mean)
        rpB, azB, rmB = self.ionB.get_radii(mean=mean)

        #if correlate_radii > 0 and mean:
        #    rpB = rpB.n/rpA.n * rpA * ufloat(1, correlate_radii)
        #    #print(rpA, rpB)
    
        dR = sys.dR_b2(self.R, rpA, rmA, azA, ompA, ommA, rpB, rmB, azB, ompB, ommB, B2, B0)
        self.ionA.apply_B2_shift(mean=mean)
        self.ionB.apply_B2_shift(mean=mean)
        dR2 = self.ionB.get_invariance_omega_c() / self.ionA.get_invariance_omega_c() - self.R
        self.adjust_dR_sum(dR, dR2)
        self.adjust_R_current_by_dR()
        return dR, dR2

    def apply_C3_shift(self, mean=True, c3=None, c2=None):
        pass

    def apply_C4_shift(self, mean=True, c4=None, c2=None, correlate_radii=1e-2):
        if c2 is None:
            c2 = self.trap['C2']
        if c4 is None:
            c4 = self.trap['C4']

        omcA, ompA, _, ommA = self.ionA.get_frequencies()
        omcB, ompB, _, ommB = self.ionB.get_frequencies()
        rpA, azA, rmA = self.ionA.get_radii(mean=mean)
        rpB, azB, rmB = self.ionB.get_radii(mean=mean)

        #if correlate_radii > 0 and mean:
        #    rpB = rpB.n/rpA.n * rpA * ufloat(1, correlate_radii)
        #    #print(rpA, rpB)

        dR = sys.dR_c4(self.R, rpA, rmA, omcA, ompA, ommA, rpB, rmB, omcB, ompB, ommB, c4, c2=c2)

        _, _, domzA, _ = self.ionA.apply_C4_shift(mean=mean)
        _, _, domzB, _ = self.ionB.apply_C4_shift(mean=mean)
        #print("dnuz A/B/diff", domzA/pi2, domzB/pi2, domzA/pi2 - domzB/pi2)
        dR2 = self.ionB.get_invariance_omega_c() / self.ionA.get_invariance_omega_c() - self.R
        self.adjust_dR_sum(dR, dR2)
        self.adjust_R_current_by_dR()
        return dR, dR2 

    def apply_C6_shift(self, mean=True, c6=None, c2=None, correlate_radii=1e-2):
        if c2 is None:
            c2 = self.trap['C2']
        if c6 is None:
            c6 = self.trap['C6']

        omcA, ompA, _, ommA = self.ionA.get_frequencies()
        omcB, ompB, _, ommB = self.ionB.get_frequencies()
        rpA, azA, rmA = self.ionA.get_radii(mean=mean)
        rpB, azB, rmB = self.ionB.get_radii(mean=mean)

        #if correlate_radii > 0 and mean:
        #    rpB = rpB.n/rpA.n * rpA * ufloat(1, correlate_radii)
        #    #print(rpA, rpB)

        dR = sys.dR_c6(self.R, rpA, azA, rmA, omcA, ompA, ommA, rpB, azB, rmB, omcB, ompB, ommB, c6, c2=c2)

        self.ionA.apply_C6_shift(mean=mean)
        self.ionB.apply_C6_shift(mean=mean)
        dR2 = self.ionB.get_invariance_omega_c() / self.ionA.get_invariance_omega_c() - self.R
        self.adjust_dR_sum(dR, dR2)
        self.adjust_R_current_by_dR()
        return dR, dR2 

    def apply_B1_shift(self, dz=0, mean=True):
        pass
    
    def apply_B1C1_shift(self, mean=True):
        pass

    def apply_num_shift(self, mean=True, value=10, valueB=None):
        if valueB is None:
            valueB = value
        self.ionA.apply_num_shift(value=value)
        self.ionB.apply_num_shift(value=valueB)
        dR2 = self.ionB.get_invariance_omega_c() / self.ionA.get_invariance_omega_c() - self.R
        self.adjust_dR_sum(0, dR2)
        self.adjust_R_current_by_dR()
        return 0, dR2 

    def apply_nuz_shift(self, mean=True, value=0, valueB=None):
        if valueB is None:
            valueB = value

        before_R = self.ionB.get_invariance_omega_c() / self.ionA.get_invariance_omega_c()
        self.ionA.apply_nuz_shift(value=value)
        self.ionB.apply_nuz_shift(value=valueB)
        dR2 = self.ionB.get_invariance_omega_c() / self.ionA.get_invariance_omega_c() - before_R
        self.adjust_dR_sum(0, dR2)
        self.adjust_R_current_by_dR()
        return 0, dR2 

    def apply_tiltellip_shift(self, mean=True, epsilon=None, theta=None, phi=None):

        self.ionA.apply_tiltellip_shift(mean=mean, epsilon=epsilon, theta=theta, phi=phi)
        self.ionB.apply_tiltellip_shift(mean=mean, epsilon=epsilon, theta=theta, phi=phi)
        dR2 = self.ionB.get_invariance_omega_c() / self.ionA.get_invariance_omega_c() - self.R
        self.adjust_dR_sum(0, dR2)
        self.adjust_R_current_by_dR()
        return 0, dR2 
        

    def all_frequency_shifts(self, mean=True, callback=print, show_individual_freq_shifts=False):
        self.zero_dR()

        table = {} # name: [dr1, dr2]

        functions = getmembers(Ratio, isfunction)
        #print(functions)
        for name, fun in functions:
            #print(name)
            if name.startswith('apply_') and name.endswith("_shift"):
                if name[6:] in self.disabled:
                    continue
                shifts = getattr(self, name)(mean=mean)
                table[name[6:]] = shifts
                if callback is not None:
                    try:
                        callback(name, *shifts)
                    except:
                        callback(name, shifts)
                if show_individual_freq_shifts:
                    print('\n', name, '\n')
                    self.show_ion_frequencies()
                if self.reset_freq_shifts:
                    self.remove_ion_shifts()

        return table



if __name__ == '__main__':

    # excitations
    print("\n\n --> GO GO GO \n")
    exc_radius = ufloat(12, 10)*1e-6 # in m
    exc_radius_m = ufloat(40, 5)*1e-6 # in m
    exc_radius_z = ufloat(300, 2)*1e-6 # in m

    # meta stable:
    if False: 
        ionA = '208Pb41+'
        test_ionA = Ion(example_trap2, ionA)
        test_ionB = Ion(example_trap2, ionA)
        test_ionA.show_radii()
        test_ionB.set_mass(None, 3, True)
        test_ionB.init_coordinates()

        print("\n\n --> now ratio, mean radii, cool\n")
        test_R = Ratio(test_ionA, test_ionB, example_trap2, sameU=True, dU=0)
        test_R.reset_freq_shifts = False
        test_R.ionA.excite_p(exc_radius)
        test_R.ionB.excite_p(exc_radius)
        #test_R.ionA.excite_z(exc_radius_z)
        #test_R.ionB.excite_z(exc_radius_z)
        #test_R.disabled.append("image_charge_shift")
        test_R.disabled.append("num_shift")
        test_R.disabled.append("nuz_shift")
        test_R.disabled.append("image_current_shift")
        test_R.disabled.append("offresonant_dip_fit_shift")
        #test_R.apply_nuz_shift(True, 0.1)
        test_R.all_frequency_shifts()
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()

    ionA = '133Cs11+'
    ionA = '172Yb43+'
    ionA = '238U54+'
    ionA = '176Yb42+'
    ionA = '40Ca10+'
    #ionA = '12C6+'
    #ionB = '168Yb42+'
    ionB = '172Yb42+'
    #ionB = '132Xe30+'
    #ionB = '12C3+'
    #ionB = '20Ne10+'
    #ionB = '168Yb34+'
    #ionB = '28Si7+'
    #ionB = '28Si7+'
    ionB = '48Ca12+'


    """
    print("\n\n --> simple stuff, mean radii, cool single ion effects\n")
    test_ion = Ion(example_trap, ionA)
    test_ion.all_frequency_shifts()
    test_ion.show_current_shifts()

    print("\n\n --> simple stuff, mean radii, excited single ion effects\n")
    test_ion.init()
    test_ion.excite_p(exc_radius)
    rp, az, rm = test_ion.get_radii()
    oc, op, oz, om = test_ion.get_frequencies()
    dz =  sys.dz_rhop_b1(rp, oc, op, oz, B1=1.41e-03, B0=7)
    print(dz, dz*test_ion.trap["B1"]/test_ion.trap["B0"])

    test_ion.all_frequency_shifts()
    test_ion.show_current_shifts()

    print("\n\n --> simple stuff, mean radii, excited magnetron (TR scan) ion effects\n")
    test_ion.init()
    test_ion.excite_m(ufloat(500, 5)*1e-6)
    test_ion.all_frequency_shifts()
    test_ion.show_current_shifts()


    print("\n\n --> simple stuff, mean radii, axially excited\n")
    test_ion.init()

    az = itp.zamp_sideband_p(exc_radius, test_ion.omegap, test_ion.omegaz)
    rp = ufloat(np.sqrt(az.n**2/2), 1e-6)

    test_ion.excite_z(az)
    test_ion.excite_p(rp)
    
    rp, az, rm = test_ion.get_radii()
    rp = rp.n/az.n * az * ufloat(1, 0.02)
    test_ion.fix_radii(rp, az, rm)

    test_ion.all_frequency_shifts()
    test_ion.show_current_shifts()
    

    # TODO more complicated, numerous statistical radii and random phase plus excitation
    """


    # Ca chain
    if False:
            
        ionA = '40Ca10+'
        ionB = '48Ca12+'

        print("\n\n --> now ratio, mean radii, cool\n", ionA, ionB)
        test_R = Ratio(ionA, ionB, example_trap3, sameU=True, Tfactor=1)
        #test_R.disabled.append("image_charge_shift")
        test_R.disabled.append("num_shift")
        test_R.disabled.append("image_current_shift")
        test_R.disabled.append("tiltellip_shift")
        test_R.disabled.append("offresonant_dip_fit_shift")
        test_R.show_radii()

        print(test_R.ionA.U0)
        test_R.all_frequency_shifts()
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()
        test_R.show_radii()

        print("\n\n --> now ratio, mean radii, excited \n", ionA, ionB)
        test_R.zero_dR()
        test_R.show_radii() 
        print('ion ratio:', test_R.R)
        test_R.ionA.excite_p(exc_radius)
        test_R.ionB.excite_p(exc_radius*1.00)
        #test_R.ionA.excite_m(exc_radius_m)
        #test_R.ionB.excite_m(exc_radius_m)

        test_R.show_radii()
        test_R.show_ion_frequencies()
        test_R.reset_freq_shifts = False
        test_R.all_frequency_shifts()
        #test_R.ionA.all_frequency_shifts()
        #test_R.ionB.all_frequency_shifts()
        #test_R.ionA.show_current_shifts()
        #test_R.ionB.show_current_shifts()
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()
        print("temp A B", test_R.ionA.trap["Temp_res"], test_R.ionB.trap["Temp_res"])


    # Neon/Helium absolute
    if True:
            
        ionA = '12C6+'
        ionB = '4He2+'
        ionB = '20Ne10+'
        #ionB = '9Be7+'
        
        example_trap = example_trap3
        example_trap["Temp_res"] = ufloat(9, 2)

        test_ionA = Ion(example_trap, ionA)
        test_ionB = Ion(example_trap, ionB)
        test_ionB.init()

        print("\n\n --> now ratio, mean radii, cool\n", ionA, ionB)
        test_R = Ratio(ionA, ionB, example_trap, sameU=False, Tfactor=1, dmBeV=0)
        test_R.show_radii()

        test_R.ionA.excite_p(exc_radius)
        test_R.ionB.excite_p(exc_radius)
        #test_R.ionA.excite_z(exc_radius*10)
        #test_R.ionB.excite_z(exc_radius*10)
        #test_R.ionA.excite_m(exc_radius_m)
        #test_R.ionB.excite_m(exc_radius_m)

        #test_R.disabled.append("image_charge_shift")
        #test_R.disabled.append("num_shift")
        test_R.disabled.append("image_current_shift")
        test_R.disabled.append("image_charge_shift")
        test_R.disabled.append("offresonant_dip_fit_shift")
        #test_R.disabled.append("tiltellip_shift")
        test_R.reset_freq_shifts = False
        test_R.show_radii()

        print(test_R.ionA.U0)
        test_R.all_frequency_shifts(show_individual_freq_shifts=True)
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()
        test_R.show_radii()
        """
        print("\n\n --> now ratio, mean radii, excited \n", ionA, ionB)
        test_R.zero_dR()
        test_R.show_radii()
        print('ion ratio:', test_R.R)
        test_R.ionA.excite_p(exc_radius)
        test_R.ionB.excite_p(exc_radius*ufloat(1.0, 0.01))
        #test_R.ionA.excite_m(exc_radius_m)
        #test_R.ionB.excite_m(exc_radius_m)

        test_R.show_radii()
        test_R.show_ion_frequencies()
        test_R.reset_freq_shifts = True
        test_R.all_frequency_shifts(show_individual_freq_shifts=True)
        #test_R.ionA.all_frequency_shifts()
        #test_R.ionB.all_frequency_shifts()
        #test_R.ionA.show_current_shifts()
        #test_R.ionB.show_current_shifts()
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()
        """

  # Beryllium absolute
    if False:
            
        ionA = '12C4+'
        ionB = '9Be3+'
        ionB = '4He2+'
        #ionB = '19Ne10+'
        example_trap3["Temp_res"] = ufloat(2.5, 1)
        exc_ratio = 1+0.01

        test_ionA = Ion(example_trap3, ionA)
        test_ionB = Ion(example_trap3, ionB)
        test_ionA.show_radii()
        test_ionB.init()

        print("\n\n --> now ratio, mean radii, cool\n", ionA, ionB)
        test_R = Ratio(ionA, ionB, example_trap3, sameU=True, Tfactor=1, dmBeV=0)

        test_R.ionA.excite_p(exc_radius)
        test_R.ionB.excite_p(exc_radius*exc_ratio)

        #test_R.disabled.append("num_shift")
        test_R.disabled.append("image_current_shift")
        test_R.disabled.append("image_charge_shift")
        test_R.disabled.append("offresonant_dip_fit_shift")
        test_R.disabled.append("tiltellip_shift")
        test_R.reset_freq_shifts = False
        test_R.show_radii()

        print(test_R.ionA.U0)
        test_R.all_frequency_shifts(show_individual_freq_shifts=True)
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()
        test_R.show_radii()
    

    # Neon absolute
    if False:
            
        ionA = '12C6+'
        ionB = '20Ne10+'
        #ionB = '19Ne10+'

        test_ionA = Ion(example_trap3, ionA)
        test_ionB = Ion(example_trap3, ionB)
        test_ionA.show_radii()
        test_ionB.init()

        print("\n\n --> now ratio, mean radii, cool\n", ionA, ionB)
        test_R = Ratio(ionA, ionB, example_trap2, sameU=False, Tfactor=1, dmBeV=0)


        """
        Ns = np.logspace(1, 5, 100)
        ps = []
        zs = []
        ms = []
        for N in Ns:
            N =int(N)
            print(N)
            test_R.ionA.init_coordinates(N)
            p, z, m = test_R.ionA.get_radii()
            ps.append(p.n)
            zs.append(z.n)
            ms.append(m.n)

        from matplotlib import pyplot as plt
        plt.loglog(Ns, ps, label="p")
        plt.loglog(Ns, ms, label="m")
        #plt.loglog(Ns, zs, label="z")
        plt.legend()
        plt.xlabel("N")
        plt.ylabel("radius mean (samples)")
        plt.grid(which="both")
        plt.show()

        plt.loglog(Ns, zs, label="z")
        plt.legend()
        plt.xlabel("N")
        plt.ylabel("radius mean (samples)")
        plt.grid(which="both")
        plt.show()
        """

        #test_R.disabled.append("image_charge_shift")
        #test_R.disabled.append("num_shift")
        test_R.disabled.append("image_current_shift")
        test_R.disabled.append("image_charge_shift")
        test_R.disabled.append("offresonant_dip_fit_shift")
        #test_R.disabled.append("tiltellip_shift")
        test_R.reset_freq_shifts = False
        test_R.show_radii()

        print(test_R.ionA.U0)
        test_R.all_frequency_shifts(show_individual_freq_shifts=True)
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()
        test_R.show_radii()
        """
        print("\n\n --> now ratio, mean radii, excited \n", ionA, ionB)
        test_R.zero_dR()
        test_R.show_radii()
        print('ion ratio:', test_R.R)
        test_R.ionA.excite_p(exc_radius)
        test_R.ionB.excite_p(exc_radius*ufloat(1.0, 0.01))
        #test_R.ionA.excite_m(exc_radius_m)
        #test_R.ionB.excite_m(exc_radius_m)

        test_R.show_radii()
        test_R.show_ion_frequencies()
        test_R.reset_freq_shifts = True
        test_R.all_frequency_shifts(show_individual_freq_shifts=True)
        #test_R.ionA.all_frequency_shifts()
        #test_R.ionB.all_frequency_shifts()
        #test_R.ionA.show_current_shifts()
        #test_R.ionB.show_current_shifts()
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()
        """


    # HoDy
    if False:
            
        ionA = '163Ho38+'
        ionB = '163Dy38+'

        test_R = Ratio(ionA, ionB, example_trap3, sameU=True, Tfactor=False)
        #test_R.disabled.append("image_charge_shift")
        test_R.disabled.append("num_shift")
        test_R.disabled.append("nuz_shift")
        test_R.disabled.append("image_current_shift")
        test_R.disabled.append("tiltellip_shift")
        test_R.disabled.append("offresonant_dip_fit_shift")
        test_R.show_radii()

        print("\n\n --> now ratio, mean radii, excited \n", ionA, ionB)
        test_R.zero_dR()
        test_R.show_radii()
        print('ion ratio:', test_R.R)
        test_R.ionA.excite_p(exc_radius)
        test_R.ionB.excite_p(exc_radius*ufloat(1.0 - 0.00, 0.01))
        #test_R.ionA.excite_m(exc_radius_m)
        #test_R.ionB.excite_m(exc_radius_m)

        test_R.show_radii()
        test_R.show_ion_frequencies()
        test_R.reset_freq_shifts = False
        test_R.all_frequency_shifts()
        #test_R.ionA.all_frequency_shifts()
        #test_R.ionB.all_frequency_shifts()
        #test_R.ionA.show_current_shifts()
        #test_R.ionB.show_current_shifts()
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()


    # U binding chain
    if False:
            
        ionA = '238U90+'
        ionB = '238U91+'
        exc_radius = ufloat(15, 1)*1e-6 # in m

        print("\n\n --> now ratio, mean radii, cool\n", ionA, ionB)
        test_R = Ratio(ionA, ionB, example_trap2, sameU=True, Tfactor=False)
        print(">>> trap depth: U0=", test_R.ionA.U0)
        #test_R.disabled.append("image_charge_shift")
        test_R.disabled.append("num_shift")
        test_R.disabled.append("nuz_shift")
        #test_R.disabled.append("image_current_shift")
        #test_R.disabled.append("tiltellip_shift")
        test_R.disabled.append("offresonant_dip_fit_shift")
        test_R.show_radii()

        print(test_R.ionA.U0)
        test_R.all_frequency_shifts()
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()
        test_R.show_radii()

        print("\n\n --> now ratio, mean radii, excited \n", ionA, ionB)
        test_R.zero_dR()
        test_R.show_radii()
        print('ion ratio:', test_R.R)
        test_R.ionA.excite_p(exc_radius)
        #test_R.ionB.excite_p(exc_radius*ufloat(1.00, 0.05))
        #test_R.ionB.excite_p(exc_radius*ufloat(1.05, 0.05))
        test_R.ionB.excite_p(exc_radius*ufloat(0.95, 0.05))
        #test_R.ionA.excite_m(exc_radius_m)
        #test_R.ionB.excite_m(exc_radius_m)

        test_R.show_radii()
        test_R.show_ion_frequencies()
        test_R.reset_freq_shifts = False
        test_R.all_frequency_shifts()
        #test_R.ionA.all_frequency_shifts()
        #test_R.ionB.all_frequency_shifts()
        #test_R.ionA.show_current_shifts()
        #test_R.ionB.show_current_shifts()
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()


    # Yb chain
    if False:
            
        ionA = '168Yb42+'
        ionB = '172Yb42+'
        exc_radius = ufloat(12, 2)*1e-6 # in m
        exc_ratio = 1+0.02
        exc_ratio = 1-0.02

        temp_plus = 0
        example_trap2["Temp_res"] = ufloat(5+temp_plus, 2) # temperature dependence is low here
        example_trap3["Temp_res"] = ufloat(8+temp_plus, 2)
        

        print("\n\n --> now ratio, mean radii, cool\n", ionA, ionB)
        test_R = Ratio(ionA, ionB, example_trap2, sameU=True, Tfactor=False, trapconf_ddres=True)
        print(">>> trap depth: U0=", test_R.ionA.U0)
        #test_R.disabled.append("image_charge_shift")
        test_R.disabled.append("num_shift")
        test_R.disabled.append("nuz_shift")
        test_R.disabled.append("image_current_shift")
        test_R.disabled.append("tiltellip_shift")
        test_R.disabled.append("offresonant_dip_fit_shift")
        test_R.show_radii()

        print(test_R.ionA.U0)
        test_R.all_frequency_shifts()
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()
        test_R.show_radii()

        print("\n\n --> now ratio, mean radii, excited \n", ionA, ionB)
        test_R.zero_dR()
        test_R.show_radii()
        print('ion ratio:', test_R.R)
        test_R.ionA.excite_p(exc_radius*ufloat(exc_ratio, 0.02))
        #test_R.ionB.excite_p(exc_radius*ufloat(1.00, 0.05))
        #test_R.ionB.excite_p(exc_radius*ufloat(1.01, 0.01))
        test_R.ionB.excite_p(exc_radius)#*ufloat(exc_ratio, 0.02))
        #test_R.ionA.excite_m(exc_radius_m)
        #test_R.ionB.excite_m(exc_radius_m)

        test_R.show_radii()
        test_R.show_ion_frequencies()
        test_R.reset_freq_shifts = True
        table2 = test_R.all_frequency_shifts()
        table2["mdiff_eV"]= test_R.mdiff_eV
        table2["mdiff"]= test_R.mdiff
        table2["C2"]= test_R.trap['C2']*(1e-3)**2
        table2["C4"]= test_R.trap['C4']*(1e-3)**4/( test_R.trap['C2']*(1e-3)**2 )
        table2["C6"]= test_R.trap['C6']*(1e-3)**6/( test_R.trap['C2']*(1e-3)**2 )
        pprint(table2)
        #test_R.ionA.all_frequency_shifts()
        #test_R.ionB.all_frequency_shifts()
        #test_R.ionA.show_current_shifts()
        #test_R.ionB.show_current_shifts()
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()

        
        print("\n\n --> now ratio, mean radii, cool\n", ionA, ionB)
        test_R = Ratio(ionA, ionB, example_trap3, sameU=True, Tfactor=False, trapconf_ddres=True)
        print(">>> trap depth: U0=", test_R.ionA.U0)
        #test_R.disabled.append("image_charge_shift")
        test_R.disabled.append("num_shift")
        test_R.disabled.append("nuz_shift")
        test_R.disabled.append("image_current_shift")
        test_R.disabled.append("tiltellip_shift")
        test_R.disabled.append("offresonant_dip_fit_shift")
        test_R.show_radii()

        print(test_R.ionA.U0)
        test_R.all_frequency_shifts()
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()
        test_R.show_radii()

        print("\n\n --> now ratio, mean radii, excited \n", ionA, ionB)
        test_R.zero_dR()
        test_R.show_radii()
        print('ion ratio:', test_R.R)
        test_R.ionA.excite_p(exc_radius*ufloat(exc_ratio, 0.02))
        #test_R.ionB.excite_p(exc_radius*ufloat(1.00, 0.05))
        #test_R.ionB.excite_p(exc_radius*ufloat(1.01, 0.01))
        test_R.ionB.excite_p(exc_radius)
        #test_R.ionB.excite_p(exc_radius*ufloat(exc_ratio, 0.02))
        #test_R.ionA.excite_m(exc_radius_m)
        #test_R.ionB.excite_m(exc_radius_m)

        test_R.show_radii()
        test_R.show_ion_frequencies()
        test_R.reset_freq_shifts = True
        table3 = test_R.all_frequency_shifts()
        table3["mdiff_eV"]= test_R.mdiff_eV
        table3["mdiff"]= test_R.mdiff
        table3["C2"]= test_R.trap['C2']*(1e-3)**2
        table3["C4"]= test_R.trap['C4']*(1e-3)**4/( test_R.trap['C2']*(1e-3)**2 )
        table3["C6"]= test_R.trap['C6']*(1e-3)**6/( test_R.trap['C2']*(1e-3)**2 )
        pprint(table3)
        #test_R.ionA.all_frequency_shifts()
        #test_R.ionB.all_frequency_shifts()
        #test_R.ionA.show_current_shifts()
        #test_R.ionB.show_current_shifts()
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()

        for key, item in table2.items():
            print(key, item, table3[key])
        print("omegac_B/omega_cA", test_R.R)

    # Yb chain, dependency on radii
    if False:
            
        ionA = '176Yb42+'
        ionB = '172Yb42+'
        print("\n\n --> now ratio, mean radii\n", ionA, ionB)

        exc_rhos = np.arange(10, 15, 2)*1e-6
        dRs = []
        ddRs = []
        for exc_rho in exc_rhos:
            test_R = Ratio(ionA, ionB, example_trap3, sameU=False, Tfactor=False)
            test_R.disabled.append("image_charge_shift")
            test_R.disabled.append("num_shift")
            test_R.disabled.append("nuz_shift")
            test_R.disabled.append("image_current_shift")
            test_R.disabled.append("tiltellip_shift")
            test_R.disabled.append("offresonant_dip_fit_shift")
            test_R.reset_freq_shifts = False

            test_R.ionA.excite_p(exc_rho)
            test_R.ionB.excite_p(exc_rho*1.01)

            test_R.all_frequency_shifts()
            #test_R.show_current_shifts()
            #test_R.show_ion_frequencies()
            curr_dR = test_R.dR
            dRs.append(curr_dR.n)
            ddRs.append(curr_dR.s)

        import matplotlib.pyplot as plt
        plt.plot(exc_rhos*1e6, dRs)
        plt.show()
        plt.plot(exc_rhos*1e6, ddRs)
        plt.show()

    # 172Yb absolute
    if False:
            
        ionA = '12C4+'
        ionB = '172Yb43+'
        ionB = '87Rb29+'

        print("\n\n --> now ratio, mean radii, cool\n", ionA, ionB)
        test_R = Ratio(ionA, ionB, example_trap3, sameU=True, Tfactor=False)
        #test_R.disabled.append("image_charge_shift")
        test_R.disabled.append("num_shift")
        test_R.disabled.append("image_current_shift")
        test_R.disabled.append("offresonant_dip_fit_shift")
        test_R.show_radii()

        print(test_R.ionA.U0)
        test_R.all_frequency_shifts()
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()
        test_R.show_radii()

        print("\n\n --> now ratio, mean radii, excited \n", ionA, ionB)
        test_R.zero_dR()
        test_R.show_radii()
        print('ion ratio:', test_R.R)
        test_R.ionA.excite_p(exc_radius)
        test_R.ionB.excite_p(exc_radius)
        #test_R.ionA.excite_m(exc_radius_m)
        #test_R.ionB.excite_m(exc_radius_m)

        test_R.show_radii()
        test_R.show_ion_frequencies()
        test_R.reset_freq_shifts = False
        test_R.all_frequency_shifts()
        #test_R.ionA.all_frequency_shifts()
        #test_R.ionB.all_frequency_shifts()
        #test_R.ionA.show_current_shifts()
        #test_R.ionB.show_current_shifts()
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()

    if False:
            
        ionB = '12C6+'
        ionA = '2H1+'

        exc_radius = ufloat(19.5, 1)*1e-6 # in m # trap 3
        #exc_radius = ufloat(13, 1)*1e-6 # in m # trap 2

        print("\n\n --> now ratio, mean radii, cool\n", ionA, ionB)
        test_R = Ratio(ionA, ionB, example_trap3, sameU=True, Tfactor=False)

        test_R.ICS_error = 0.01
        test_R.ionA.ICS_error = 0.01
        test_R.ionB.ICS_error = 0.01
        #test_R.disabled.append("image_charge_shift")
        test_R.disabled.append("num_shift")
        test_R.disabled.append("image_current_shift")
        test_R.disabled.append("offresonant_dip_fit_shift")
        test_R.disabled.append("tiltellip_shift")
        test_R.show_radii()

        print(test_R.ionA.U0)
        test_R.all_frequency_shifts()
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()
        test_R.show_radii()

        print("\n\n --> now ratio, mean radii, excited \n", ionA, ionB)
        test_R.zero_dR()
        test_R.show_radii()
        print('ion ratio:', test_R.R)
        test_R.ionA.excite_p(exc_radius)
        test_R.ionB.excite_p(exc_radius*1.02)
        #test_R.ionA.excite_m(exc_radius_m*1.02)
        #test_R.ionB.excite_m(exc_radius_m)
        test_R.show_radii()
        test_R.show_ion_frequencies()
        test_R.reset_freq_shifts = True
        test_R.all_frequency_shifts(show_individual_freq_shifts=True)
        #test_R.ionA.all_frequency_shifts()
        #test_R.ionB.all_frequency_shifts()
        #test_R.ionA.show_current_shifts()
        #test_R.ionB.show_current_shifts()
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()

        print('theo ICS dnum A-B', (test_R.ionA.q - test_R.ionB.q)*e/(4*np.pi*epsilon0*test_R.trap['trap_radius']**3*test_R.trap['B0'])/2/np.pi)

    """
    # 238U image charge
    # 238U mass vs Xenon
    if True:
            
        ionA = '12C3+'
        ionB = '238U58+'
            
        ionA = '132Xe26+'
        ionB = '238U47+'

        exc_radius = ufloat(19.5, 1)*1e-6 # in m # trap 3
        #exc_radius = ufloat(13, 1)*1e-6 # in m # trap 2

        print("\n\n --> now ratio, mean radii, cool\n", ionA, ionB)
        test_R = Ratio(ionA, ionB, example_trap3, sameU=True, Tfactor=False)

        test_R.ICS_error = 0.01
        test_R.ionA.ICS_error = 0.01
        test_R.ionB.ICS_error = 0.01
        #test_R.disabled.append("image_charge_shift")
        test_R.disabled.append("num_shift")
        test_R.disabled.append("image_current_shift")
        test_R.disabled.append("offresonant_dip_fit_shift")
        test_R.disabled.append("tiltellip_shift")
        test_R.show_radii()

        print(test_R.ionA.U0)
        test_R.all_frequency_shifts()
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()
        test_R.show_radii()

        print("\n\n --> now ratio, mean radii, excited \n", ionA, ionB)
        test_R.zero_dR()
        test_R.show_radii()
        print('ion ratio:', test_R.R)
        test_R.ionA.excite_p(exc_radius)
        test_R.ionB.excite_p(exc_radius*1.02)
        #test_R.ionA.excite_m(exc_radius_m*1.02)
        #test_R.ionB.excite_m(exc_radius_m)
        test_R.show_radii()
        test_R.show_ion_frequencies()
        test_R.reset_freq_shifts = True
        test_R.all_frequency_shifts(show_individual_freq_shifts=True)
        #test_R.ionA.all_frequency_shifts()
        #test_R.ionB.all_frequency_shifts()
        #test_R.ionA.show_current_shifts()
        #test_R.ionB.show_current_shifts()
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()

        print('theo ICS dnum A-B', (test_R.ionA.q - test_R.ionB.q)*e/(4*np.pi*epsilon0*test_R.trap['trap_radius']**3*test_R.trap['B0'])/2/np.pi)

        # diffs num A-B, all/ics 0.005287+/-0.000017 -0.005498+/-0.000016
        # diffs num A-B, all/ics 0.010785411794131505 -0.005498+/-0.000016


        #print("num shift", test_R.apply_num_shift(value=0.5*2*np.pi, valueB=0.3*2*np.pi))#-0.5))
        #test_R.show_current_shifts()
    """
    
    """
        print("\n\n --> now ratio, mean radii, cool\n")
        test_R = Ratio(ionA, ionB, example_trap, sameU=True)
        test_R.all_frequency_shifts()
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()

        print("\n\n --> now ratio, mean radii, excited \n")
        test_R.zero_dR()
        print('ion ratio:', test_R.R)
        test_R.ionA.excite_p(exc_radius)
        test_R.ionB.excite_p(exc_radius)
        print('A radii:', test_R.ionA.get_radii())
        print('B radii:', test_R.ionB.get_radii())
        test_R.all_frequency_shifts()
        test_R.show_current_shifts()

        print("\n\n --> now ratio, mean radii, excited, other trap \n")
        example_trap2 = example_trap.copy()
        example_trap2["nu_res"] = ufloat(477000.0, 10.0)
        example_trap2["B2"] = ufloat(0.064, 0.005)
        test_R2 = Ratio(ionA, ionB, example_trap2, sameU=True)

        print('ion ratio:', test_R2.R)
        test_R2.ionA.excite_p(exc_radius)
        test_R2.ionB.excite_p(exc_radius)
        print('A radii:', test_R2.ionA.get_radii())
        print('B radii:', test_R2.ionB.get_radii())
        test_R2.all_frequency_shifts()
        test_R2.show_current_shifts()


        print("\n\n --> now ratio, mean radii, excited axially, only axial freq. effect \n")
        test_R.zero_dR()
        test_R.init_ion_coordinates()

        azA = itp.zamp_sideband_p(exc_radius, test_R.ionA.omegap, test_R.ionA.omegaz)
        azB = itp.zamp_sideband_p(exc_radius, test_R.ionB.omegap, test_R.ionB.omegaz)

        test_R.ionA.excite_z(azA)
        test_R.ionB.excite_z(azB)
        test_R.show_radii()

        test_R.ionA.enable_c_shifts = False
        test_R.ionA.enable_p_shifts = False
        test_R.ionA.enable_m_shifts = False
        test_R.ionB.enable_c_shifts = False
        test_R.ionB.enable_p_shifts = False
        test_R.ionB.enable_m_shifts = False

        test_R.all_frequency_shifts()
        test_R.show_current_shifts()


        print("\n\n --> now ratio, excited axially (sideband relation radius from 15um rho+ excitation)\n")
        test_R.init()
        test_R.init_ion_coordinates()
        azA = itp.zamp_sideband_p(exc_radius, test_R.ionA.omegap, test_R.ionA.omegaz)
        azB = itp.zamp_sideband_p(exc_radius, test_R.ionB.omegap, test_R.ionB.omegaz)
        factor = 1.115
        exc_radiusA = unp.sqrt(azA**2/2) /factor
        exc_radiusB = unp.sqrt(azB**2/2) /factor

        #test_R.ionA.no_Ci_errors()
        #test_R.ionB.no_Ci_errors()

        print("axial radii:", azA, azB)
        print("cyclo radii:", exc_radiusA, exc_radiusB)
        test_R.ionA.excite_z(azA)
        test_R.ionB.excite_z(azB)
        test_R.ionA.excite_p(exc_radiusA)
        test_R.ionB.excite_p(exc_radiusB)
        
        test_R.show_radii()

    """

    """
        rpA, azA, rm = test_R.ionA.get_radii()
        rpA = rpA.n/azA.n * azA * ufloat(1, 0.01)
        test_R.ionA.fix_radii(rpA, azA, rm)
        
        rp, az, rm = test_R.ionB.get_radii()
        az = azA.n/az.n * az * ufloat(1, 0.001)
        rp = rpA.n/rp.n * rp * ufloat(1, 0.001)
        rp = rp.n/az.n * az * ufloat(1, 0.01)
        test_R.ionB.fix_radii(rp, az, rm)
        test_R.show_radii()
        

        test_R.ionA.enable_c_shifts = False
        test_R.ionA.enable_p_shifts = False
        test_R.ionA.enable_m_shifts = False
        test_R.ionB.enable_c_shifts = False
        test_R.ionB.enable_p_shifts = False
        test_R.ionB.enable_m_shifts = False

        test_R.all_frequency_shifts()
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()


        print("\n\n --> now ratio 187Re/Os, mean radii, cool\n")
        test_R = Ratio('187Re29+', '187Os29+', example_trap, sameU=True)
        test_R.all_frequency_shifts()
        test_R.show_current_shifts()
        test_R.show_ion_frequencies()

        print("\n\n --> now ratio, mean radii, excited \n")
        test_R.zero_dR()
        print('ion ratio:', test_R.R)
        test_R.ionA.excite_p(25e-6)
        test_R.ionB.excite_p(25e-6)
        print('A radii:', test_R.ionA.get_radii())
        print('B radii:', test_R.ionB.get_radii())
        test_R.all_frequency_shifts()
        test_R.show_current_shifts()
        """


