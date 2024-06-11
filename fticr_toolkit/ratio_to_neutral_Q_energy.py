#
# Calculation of final results from Cyclotron Frequency Ratios (cfr) to extract:
#   1) neutral mass ratio
#   2) Q value
#   3) Binding energy
#   4) Meta stable state energy (same as 3 actually)
#
# This does NOT include systematic shifts of the ratio, so the ratio it self must be corrected before!
#
# The parameter ratio_measured is always given in:
#     R = ionB / ionA = nucB / nucA = (q/m)_B / (q/m)_A = mA/mB * qB/qA
# For neutral mass calculations, ionB or mB is the reference mass, so if you get the wrong mass out,
# you have to switch ionA<->ionB
# If the ratio_measured is given as the (wrong) inverted ratio, it will be corrected automatically
# according to the ions defined as ionA/ionB.
#
# Author: Menno Door
# 

from uncertainties import ufloat
from matplotlib import pyplot as plt
import matplotlib

import numpy as np
import scipy.constants as cont

from fticr_toolkit import ame

# needed constants
m_e = cont.physical_constants["electron mass in u"] # error is 3e-11
m_e = ufloat(m_e[0], m_e[2])

# naming of conversion variables is based on unit1unit2 = unit2/unit1, e.g. ukg is kg/u (so basically ukg converts u to kg)
ukg = cont.physical_constants["atomic mass unit-kilogram relationship"][0] # the uncertainty of this conversions is bad (3e-10)
ueV = cont.physical_constants["atomic mass unit-electron volt relationship"][0] # the uncertainty of this conversions is bad (3e-10)
eVkg = cont.physical_constants["electron volt-kilogram relationship"][0] # exact

# nicer plots 
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'


def ratio_test_n_flip(uR, ionA, ionB):

    # ame masses and ion parameters
    mionA, dmionA, extra_dataA = ame.get_ion_mass(ion=ionA, binding=None, full=True)
    qA = extra_dataA["q"]

    mionB, dmionB, extra_dataB = ame.get_ion_mass(ion=ionB, binding=None, full=True)
    qB = extra_dataB["q"]

    # (q/m)_B / (q/m)_A = mA/mB * qB/qA
    testR = (qB/qA * mionA / mionB) - 1
    testR2 = uR - 1
    flipped = False
    if testR*testR2 < 0:
        flipped = True
        uR = 1/uR

    return uR, flipped


def neutral_mass(uR, ionB, ionA, Rcompare = None, EbB=None, EbA=None, fig_size=(6,4), font_size=10, export_path="neutral_mass", show=True):
    """
    uR and EbX values have to be given as ufloat including their uncertainty
    """

    plt.rcParams["figure.figsize"] = fig_size
    font = {'size'   : font_size}
    matplotlib.rc('font', **font)

    # ratio flipped?
    uR, flipped = ratio_test_n_flip(uR, ionA, ionB)
    if flipped:
        print("ratio was flipped!")

    # ame masses and ion parameters
    mionA, dmionA, extra_dataA = ame.get_ion_mass(ion=ionA, binding=EbA, full=True)
    mneutralA = extra_dataA["mneutral"]
    qA = extra_dataA["q"]
    AA = extra_dataA["A"]
    EbA = extra_dataA["total_binding"]
    mass_excessA = extra_dataA["mass_excessA"]

    mionB, dmionB, extra_dataB = ame.get_ion_mass(ion=ionB, binding=EbB, full=True)
    mneutralB = extra_dataB["mneutral"]
    qB = extra_dataB["q"]
    EbB = extra_dataB["total_binding"]

    # neutral ratio from AME
    nRatio_ame = mneutralA/mneutralB
    if Rcompare is not None:
        nRatio_ame = Rcompare
    
    # neutral ratio from measured ratio
    nRatio = uR*qA/qB + (qA*m_e)*(1-uR)/mneutralB + ((uR*qA/qB - 1)*EbB/ueV + (EbA-EbB)/ueV)/mneutralB

    # better eV mass excess
    eV_mass_excess = ueV*(uR*qA/qB*mneutralB - AA) + qA*m_e*(1-uR) + uR*qA/qB*EbB - EbA
    #eV_mass_excess = 1/ueV*(uR_sys*charge_ioi/charge*u_ref - isotope_ioi)+uR_sys*charge_ioi/charge*(total_bin_ref-m_e*charge)+m_e*charge_ioi - total_bin_ioi

    # neutral mass TODO
    nmass = None

    if show: 
        print('neutral ratio ame', nRatio_ame, nRatio_ame.s)
        print('neutral ratio ame relative error', nRatio_ame.s/nRatio_ame.n)
        print('neutral ratio measured', nRatio, nRatio.s)
        print('neutral ratio measured relative error', nRatio.s/nRatio.n)
        print("Rneutral / diff to ame in u", nRatio, nRatio_ame, nRatio - nRatio_ame)

        plt.errorbar(["AME"], nRatio_ame.n, nRatio_ame.s, fmt='^', c='k')
        plt.errorbar(["this work"], nRatio.n, nRatio.s, fmt='^', c='y')

        plt.xlim(-0.7,1.7)
        plt.grid(which='both')
        plt.tick_params(direction="in", bottom=True, top=True, left=True, right=True)
        plt.ylabel('neutral mass ratio (1)')
        plt.tight_layout()
        export_path = export_path.split(".")[0]
        plt.savefig(export_path+'_ratio.svg')
        plt.savefig(export_path+'_ratio.png')
        plt.show()

        print('mass excess ame', mass_excessA, mass_excessA.s)
        print('mass excess measured', eV_mass_excess, eV_mass_excess.s)
        print('measured - ame (eV, sqr sum error)', eV_mass_excess.n - mass_excessA.n, np.sqrt(eV_mass_excess.s**2 + mass_excessA.s**2) )
        print('combined sigma difference measured - ame', (eV_mass_excess.n - mass_excessA.n)/np.sqrt(eV_mass_excess.s**2 + mass_excessA.s**2) )

        plt.errorbar(["AME"], mass_excessA.n, mass_excessA.s, fmt='^', c='k')
        plt.errorbar(["this work"], eV_mass_excess.n, eV_mass_excess.s, fmt='^', c='y')
        plt.xlim(-0.7,1.7)
        plt.grid(which='both')
        plt.tick_params(direction="in", bottom=True, top=True, left=True, right=True)
        plt.ylabel('$m_{meas} - A_{isotope}$ (eV)')
        plt.tight_layout()
        plt.savefig(export_path+'_excess.svg')
        plt.savefig(export_path+'_excess.png')
        plt.show()

        """
        plt.errorbar(["AME"],       0, mass_excessA.s, fmt='^', c='k')
        plt.errorbar(["this work"], eV_mass_excess.n - mass_excessA.n, eV_mass_excess.s, fmt='^', c='y')
        
        plt.xlim(-0.7,1.7)
        plt.grid(which='both')
        plt.tick_params(direction="in", bottom=True, top=True, left=True, right=True)
        plt.ylabel('$m_{meas} - m_{AME}$ (eV)')
        plt.tight_layout()
        plt.savefig(export_path+'_excess_diff.svg')
        plt.savefig(export_path+'_excess_diff.png')
        plt.show()
        """

    return nRatio, eV_mass_excess, nmass


def Q_value(ratio_measured, dratio_measured, ionB, ionA, Rcompare = None, EbB=None, EbA=None, fig_size=(6,4), font_size=10, export_path="neutral_mass", show=True):
    pass


def meta_energy(ratio_measured, dratio_measured, ionB, ionA, Rcompare = None, EbB=None, EbA=None, fig_size=(6,4), font_size=10, export_path="neutral_mass", show=True):
    pass


def binding_energy(ratio_measured, dratio_measured, ionB, ionA, Rcompare = None, EbB=None, EbA=None, fig_size=(6,4), font_size=10, export_path="neutral_mass", show=True):
    pass


def neutral_ratio_to_q_charged_ratio(ratio_measured, dratio_measured, ionB, ionA, Rcompare = None, EbB=None, EbA=None, fig_size=(6,4), font_size=10, export_path="neutral_mass", show=True):
    pass
