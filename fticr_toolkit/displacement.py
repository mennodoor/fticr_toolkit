###
# 
# This module allows to calculate the potential on the central axis of a trap tower
# or more of a stack of electrodes with equal inner radius. This is based on the 
# analytical solution presented in the PhD thesis of C. Roux (2012, University Heidelberg)
# and the inital code was written by A. Rischka. The base functionallity of potential 
# simulation is given by the init and potential function. All other methods are for
# translating config dictionaries to voltage lists and fitting the potential to extract
# the effective parameters.
# 
# The SimEval class has a simular functionallity, but is using data created by a Comsol
# simulation and just scales and adds the effective potential together. This is obviously
# limited to the geometries used in the Comsol simulation.
#
###


import glob, copy
from pprint import pprint
import numpy as np
import pathlib
from fticr_toolkit import ideal_trap_physics as itp
from fticr_toolkit import ame
import matplotlib.pyplot as plt

from scipy.special import i0, iv
from scipy.integrate import solve_ivp
from numpy.random import randn

simulation_data_folder = "data\\"
debug = False
this_files_path = pathlib.Path(__file__).parent.absolute()

# ANALYTICAL SOLUTION

class trap_tower:
    def __init__(self, radius=5e-3, gap=0.15e-3, electrode_lengths=None, order = 50, machining_precision = None, random_seed = None, fit_span_mm=1, gap_machining_precision=None):
        
        if electrode_lengths is None:
            lec = 7.040e-3
            lce = 3.932e-3
            lre = 1.457e-3
            electrode_lengths = [lec, lce, lre, lce, lec, lec, lce, lre, lce, lec, lec, lce, lre, lce, lec]
        
        if random_seed is not None:
            np.random.seed(random_seed)

        if machining_precision is not None:
            electrode_lengths += machining_precision * randn(len(electrode_lengths)) + 0

        self.radius = radius
        self.gap = gap
        self.electrode_lenghts = electrode_lengths
        self.order = order
        self.fit_span = fit_span_mm

        self.z = [0.0]

        # z position of edges
        for i, length in enumerate(self.electrode_lenghts):
            self.z.append(self.z[-1] + length)
            if i != (len(self.electrode_lenghts)-1):
                if gap_machining_precision is not None:
                    gap = self.gap + gap_machining_precision * randn(1)[0]
                else:
                    gap = self.gap
                self.z.append(self.z[-1] + self.gap)
            else:
                pass
                #print('last')

        #print(electrode_lengths)

        self.lam = self.z[-1] - self.z[0]
        zs = len(self.z)
        #print(self.z, self.lam/2, (self.z[int(zs/2)]-self.z[int(zs/2)-1])/2+ self.z[int(zs/2)-1])
        self.m = len(self.electrode_lenghts) # number of electrodes

    def potential(self, z_position=None, U=None):

        temp0 = 0.0
        for n in range(1, self.order):
            kn = n*np.pi/self.lam
            uip = (U[0]*np.cos(kn*self.z[0]) - U[self.m - 1]*np.cos(kn*self.lam))/kn

            temp1 = 0.0
            for i in range(2, self.m+1, 1):
                phi_ni = (U[i-1] - U[i-2])/(kn*kn*self.gap) * (np.sin(kn*self.z[2*i-2]) - np.sin(kn*self.z[2*i-3]))
                temp1 += phi_ni

            temp0 += (uip + temp1) * i0(0.0)/i0(kn*self.radius)*np.sin(kn*z_position)

        return 2.0/self.lam * temp0
    
    def Ulist2potential(self, Ulist, span=None):
        if span is None:
            span = self.fit_span
        
        zp = np.linspace((self.lam/2)*1000 - span, (self.lam/2)*1000 + span, 1000) # position in mm
        pot = self.potential(zp/1000, Ulist) 
        
        zp -= self.lam/2*1000 # shift to trap center

        return zp, pot
    
    def trapsettings2Ulist(self, trap_settings, center=2):
        tu = trap_settings[center-1]
        tc = trap_settings[center]
        tl = trap_settings[center+1]
        traps = [tu, tc, tl]

        Ulist = []
        for trap in traps:
            offsets = trap['Offset']
            U0 = trap['U0']
            CE = U0*trap["TR"]
            voltages = np.asarray([0, CE, U0, CE, 0])
            if len(offsets) == 2:
                voltages += [0, offsets[0], 0, offsets[1], 0]
            elif len(offsets) == 4:
                voltages += [offsets[0], offsets[1], 0, offsets[2], offsets[3]]
            elif len(offsets) == 5:
                voltages += [offsets[0], offsets[1], offsets[2], offsets[3], offsets[4]]
                
            Ulist.extend(voltages)

        #print(Ulist)
        #Ulist = [0, tu["U0"]*tu["TR"] + tu["Offset"][0], tu["U0"], tu["U0"]*tu["TR"] + tu["Offset"][1], 0]
        #Ulist.extend( [0, tc["U0"]*tc["TR"] + tc["Offset"][0], tc["U0"], tc["U0"]*tc["TR"] + tc["Offset"][1], 0] )
        #Ulist.extend( [0, tl["U0"]*tl["TR"] + tl["Offset"][0], tl["U0"], tl["U0"]*tl["TR"] + tl["Offset"][1], 0] )
        return Ulist

    def trapsettings2potential(self, trap_settings, trap=2, span = None):
        if span is None:
            span = self.fit_span

        init_U = self.trapsettings2Ulist(trap_settings, trap)

        zp = np.linspace((self.lam/2)*1000 - span, (self.lam/2)*1000 + span, 1000) # position in mm
        pot = self.potential(zp/1000, init_U) 
        
        zp -= self.lam/2*1000 # shift to trap center

        return zp, pot

    def trapsettings2fitminimum(self, trap_settings, trap=2, order=20, full=False, show=False):
        zp, pot = self.trapsettings2potential(trap_settings, trap) # zp is in mm

        #print(zp.shape, pot.shape)
        # do polyfit of potential data
        try:
            popt, pcov = np.polyfit(zp, pot, order, cov=True)
            perr = np.sqrt(np.diag(pcov))
        except AttributeError:
            print("\n*** there was no potential caluclated jet, please do that :) ***\n")
            return 0

        U0 = trap_settings[trap]['U0']
        #print(popt/U0)

        polyobj = np.poly1d(popt) # create polynom object

        # get minimum
        crit = polyobj.deriv().r # roots of the 1st derivative
        r_crit = crit[crit.imag==0].real # only real roots
        test = polyobj.deriv(2)(r_crit) # evaluate 2nd derivative at roots
        x_min = r_crit[test>0] # extrema with positive 2nd derivative are minima
        y_min = polyobj(x_min) # y values

        if show:
            plt.plot(zp, pot, label="potential")
            plt.plot(zp, np.polyval(popt, zp), label="fit")
            plt.legend()
            plt.show()
        #print("found minima :", order, x_min)
        #print("fit minimum  :", y_min[-1], x_min[-1]*1000, "µm")
        #print(perr[-2])
        if full:
             return x_min[-1], popt/U0, perr/abs(U0)
        return x_min[-1] # this is the minimum position in mm

    def trapsettings2fitminimum_at_minimum(self, trap_settings, trap=2, order=20, full=False, show=False):

        # just to get the inital z axis
        zp, pot = self.trapsettings2potential(trap_settings, trap) # zp is in mm

        # find minimum and shift the axis
        xmin = self.trapsettings2fitminimum(trap_settings=trap_settings, trap=trap, order=order, full=False, show=False)
        #print("min pos (shifting)", np.around(xmin, 9))
        zp -= np.around(xmin, 9) # it doesn't matter if the rounding is smaller than the discrete z axis steps, its just an offset

        #print(zp.shape, pot.shape)
        # do polyfit of potential data
        try:
            popt, pcov = np.polyfit(zp, pot, order, cov=True)
            perr = np.sqrt(np.diag(pcov))
        except AttributeError:
            print("\n*** there was no potential caluclated jet, please do that :) ***\n")
            return 0

        U0 = trap_settings[trap]['U0']
        #print(popt/U0)

        polyobj = np.poly1d(popt) # create polynom object

        # get minimum
        crit = polyobj.deriv().r # roots of the 1st derivative
        r_crit = crit[crit.imag==0].real # only real roots
        test = polyobj.deriv(2)(r_crit) # evaluate 2nd derivative at roots
        x_min = r_crit[test>0] # extrema with positive 2nd derivative are minima
        y_min = polyobj(x_min) # y values

        if show:
            plt.plot(zp, pot, label="potential")
            plt.plot(zp, np.polyval(popt, zp), label="fit")
            plt.legend()
            plt.show()
        #print("found minima :", order, x_min)
        #print("fit minimum  :", y_min[-1], x_min[-1]*1000, "µm")
        #print(perr[-2])
        if full:
             return x_min[-1], popt/U0, perr/abs(U0) # the coeffecients of the fit funtion are actually c_i*U0, so they have to be normallized
        return x_min[-1] # this is the minimum position in mm



class SimEval():

    def __init__(self):
        # gather simulation data
        self.sim_data = {}
        print("> loading simulation data... ", end="", flush=True)
        for name in glob.glob(str(this_files_path)+"//data//*"):
            if debug: print("add", name)
            data = np.loadtxt(name)
            idx = int(name.split('\\')[-1][0])
            self.sim_data[  idx  ] = data[:,2]
            self.sim_data[ "pos" ] = data[:,1] # overwritten every time, is the same in all files anyway
        print("done.")
        if debug: print(self.sim_data["pos"])

    def calc_potential(self, trapnumber, trap_settings):

        """
        determines the minimum position of the potential just by searching the minimum value of the potential array

        :param dict trap_settings: trap setting dict of a measurement config {1: {"U0": -45, "TR": 0.88056, "Offset": [0, 0] }, ... }
        """

        centertrap = int(trapnumber)
        uppertrap = int(trapnumber -1)
        lowertrap = int(trapnumber +1)
        if debug:
            pprint(trap_settings)
            print("traps", uppertrap, centertrap, lowertrap)

        self.voltages = {}

        self.voltages[1] = trap_settings[uppertrap]["U0"] * trap_settings[uppertrap]["TR"] + trap_settings[uppertrap]["Offset"][0]
        self.voltages[2] = trap_settings[uppertrap]["U0"] 
        self.voltages[3] = trap_settings[uppertrap]["U0"] * trap_settings[uppertrap]["TR"] + trap_settings[uppertrap]["Offset"][1]

        self.voltages[4] = trap_settings[centertrap]["U0"] * trap_settings[centertrap]["TR"] + trap_settings[centertrap]["Offset"][0]
        self.voltages[5] = trap_settings[centertrap]["U0"] 
        self.voltages[6] = trap_settings[centertrap]["U0"] * trap_settings[centertrap]["TR"] + trap_settings[centertrap]["Offset"][1]

        self.voltages[7] = trap_settings[lowertrap]["U0"] * trap_settings[lowertrap]["TR"] + trap_settings[lowertrap]["Offset"][0]
        self.voltages[8] = trap_settings[lowertrap]["U0"] 
        self.voltages[9] = trap_settings[lowertrap]["U0"] * trap_settings[lowertrap]["TR"] + trap_settings[lowertrap]["Offset"][1]

        self.potential = np.zeros(len(self.sim_data[1]))
        self.set_depth = trap_settings[centertrap]["U0"]

        for i in range(9):
            try:
                self.potential += self.voltages[i+1] * self.sim_data[i+1]
            except Exception as e:
                print(e)

        return self.potential

    def get_min_fast(self):
        try:
            idx = np.argmin(self.potential)
        except AttributeError:
            print("\n*** there was no potential caluclated jet, please do that :) ***\n")
            return 0
        
        value = self.potential[idx]
        pos = self.sim_data[ "pos" ][idx]
        if debug: print("array minimum:", value, pos, "mm")
        return pos

    def fit_potential(self, order=20):
        # do polyfit of potential data
        try:
            self.poly_coeffs, pcov = np.polyfit(self.sim_data[ "pos" ], self.potential, order, cov=True)
            perr = np.sqrt(np.diag(pcov))
        except AttributeError:
            print("\n*** there was no potential caluclated jet, please do that :) ***\n")
            return 0

        polyobj = np.poly1d(self.poly_coeffs) # create polynom object

        # get minimum
        crit = polyobj.deriv().r # roots of the 1st derivative
        r_crit = crit[crit.imag==0].real # only real roots
        test = polyobj.deriv(2)(r_crit) # evaluate 2nd derivative at roots
        x_min = r_crit[test>0] # extrema with positive 2nd derivative are minima
        y_min = polyobj(x_min) # y values

        #print("found minima :", order, x_min)
        #print("fit minimum  :", y_min[-1], x_min[-1]*1000, "µm")
        #print("fit minimum  :", x_min[-1]*1000, "µm")

        return x_min[-1]



if __name__ == "__main__": #[0.013531, -0.013531] 

    ion1 = "174Yb42+"
    ion2 = "174Yb43+"
    #ion1 = "20Ne10+"
    #ion2 = "12C6+"
    A, el, q1 = itp.re_ionstr(ion1)
    A, el, q2 = itp.re_ionstr(ion2)
    m1, dm1 = ame.get_ion_mass(ion1)
    m2, dm2 = ame.get_ion_mass(ion2)
    U1 = itp.U0(q1, m1, 504e3)
    U2 = itp.U0(q2, m2, 504e3)
    trap = 3

    trap_settings = {
            1 : {
                "U0" : -13.96982,
                "TR" : 0.8795,
                "Offset" : [-0.003, 0.003]
            },
            2 : {
                "U0" : -38.0,
                "TR" : 0.8798,
                "Offset" : [0.022, -0.022],
                "OffsetError" : 15e-06
            },
            3 : {
                "U0" : -13.96982,
                "TR" : 0.8795,
                "Offset" : [-0.002710, 0.002710], # +-13e-06
                #"Offset" : [-0.003393, 0.003393], # +-13e-06
                #"Offset" : [0, 0], # +-13e-06
                #"Offset" : [-0.002697, 0.002697], # +-13e-06
                #"Offset" : [-0.002723, 0.002723], # +-13e-06
                "OffsetError" : 13e-06
            },
            4 : {
                "U0" : -30.11500,
                "TR" : 0.8798,
                "Offset" : [0.0, -0.0]
                #"Offset" : [0.021, -0.021]
            },
            5 : {
                "U0" : -13.96982,
                "TR" : 0.8795,
                "Offset" : [-0.003, 0.003]
            }
    }

    tt = trap_tower(radius=5e-3, gap=0.15e-3, electrode_lengths=None, order = 50, machining_precision = 5e-6, random_seed=10)

    xmina = tt.trapsettings2fitminimum(trap_settings, 3, order=10)
    print("fit minimum ana :", xmina, "mm")

    '''

    Ubefore = trap_settings[trap]['U0']
    ts1 = copy.deepcopy(trap_settings)
    ts2 = copy.deepcopy(trap_settings)
    ts1[trap]['U0'] = U1
    ts2[trap]['U0'] = U2
    xmin1 = tt.trapsettings2fitminimum(ts1, trap, order=10)
    xmin2 = tt.trapsettings2fitminimum(ts2, trap, order=10)
    print(ion1, ion2, "diff")
    print(U1, U2, U2-U1)
    print(xmin1, xmin2, xmin2-xmin1)

    sim_eval = SimEval()
    #plt.plot(sim_eval.sim_data[ "pos" ], sim_eval.potential, label="numerical")
    #zp, pot = tt.trapsettings2potential(trap_settings, trap)
    #plt.plot(zp, pot, label="analytical")
    #plt.legend()
    #plt.show()

    offerr = trap_settings[trap]['OffsetError']
    offlist = offerr * randn(100)

    mins = []
    minsn = []
    for off in offlist:
        ts = copy.deepcopy(trap_settings)
        ts[trap]['Offset'] = [ts[trap]['Offset'][0] - off, ts[trap]['Offset'][1] + off]
        #print(ts[trap]['Offset'])
        xmina = tt.trapsettings2fitminimum(ts, trap, order=10)
        mins.append(xmina)
        #print("fit minimum ana :", xmina, "µm")
        sim_eval.calc_potential(trap, ts)
        xmin2 = sim_eval.fit_potential(order=10)
        minsn.append(xmin2)

    mins = np.asarray(mins)
    min_mean = np.mean(mins)
    min_std = np.std(mins)
    print('analytical', min_mean, '+-', min_std, 'um')

    mins = np.asarray(minsn)
    min_mean = np.mean(minsn)
    min_std = np.std(minsn)
    print('numerical', min_mean, '+-', min_std, 'um')

    #xmin1 = sim_eval.get_min_fast()
    #print("array minimum:", xmin1, "µm")

    #xmin2 = sim_eval.fit_potential(order=10)
    #print("fit minimum num :", xmin2, "µm")

    #print("difference   :", (xmina - xmin2), "µm")
    #print("difference   :", (xmin1 - xmin2), "µm")
    #if xmin1 != 0: print("rel error    :", (xmin1 - xmin2)/xmin1)
    #elif xmin2 != 0: print("rel error    :", (xmin1 - xmin2)/xmin2)
    
    """
    xmins = []
    origTR = trap_settings[trap]["TR"]
    TRs =  np.arange(origTR-0.01, origTR+0.01, 0.001)
    for tr in TRs:
        trap_settings[trap]["TR"] = tr
        xmina, popt, perr = tt.trapsettings2fitminimum(trap_settings, trap, order=10, full=True)
        xmins.append( xmina )
        print(tr, popt, perr)
        #print("fit minimum  :", xmin2, "µm")

    plt.plot(TRs, xmins)
    plt.ylabel("min pos [um]")
    plt.xlabel("offset")
    plt.show()

    xminsa = []
    xminsn = []
    diffs = []
    offsets =  np.arange(0, 0.1, 0.01)
    for o in offsets:
        trap_settings[2]["Offset"] = [ o, -o ]
        xmina = tt.trapsettings2fitminimum(trap_settings, 2, order=10)
        xminsa.append( xmina )
        sim_eval.calc_potential(2, trap_settings)
        xminn = sim_eval.fit_potential(order=10)
        xminsn.append( xminn )
        diffs.append(xmina-xminn)
        #print("fit minimum  :", xmin2, "µm")


    plt.plot(offsets, diffs)
    #plt.plot(offsets, xminsa)
    #plt.plot(offsets, xminsn)
    plt.ylabel("min pos [um]")
    plt.xlabel("offset")
    plt.show()

    xmins = []
    orders =  range(10, 30)
    for o in orders:
        xmins.append( sim_eval.fit_potential(order=o) )
        #print("fit minimum  :", xmin2, "µm")


    plt.plot(orders, xmins)
    plt.ylabel("min pos [um]")
    plt.xlabel("polyfit order")
    plt.show()

    xmins = []
    origTR = trap_settings[2]["TR"]
    TRs =  np.arange(origTR-0.01, origTR+0.01, 0.001)
    for tr in TRs:
        trap_settings[2]["TR"] = tr
        sim_eval.calc_potential(2, trap_settings)
        xmins.append( sim_eval.fit_potential(order=25) )
        #print("fit minimum  :", xmin2, "µm")

    plt.plot(TRs, xmins)
    plt.ylabel("min pos [um]")
    plt.xlabel("polyfit order")
    plt.show()

    """
    '''