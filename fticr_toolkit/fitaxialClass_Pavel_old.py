'''
These axial dip fit methods are written by P. Filianin, strongly based on the methods written by D. Lange for his Bachelor Thesis.
'''

import fticr_toolkit
import numpy as np
# suppress error message 'RuntimeWarning: divide by zero encountered in true_divide' in the dip function calculation
np.seterr(divide='ignore', invalid='ignore')

from scipy.signal import savgol_filter
from scipy.integrate import quad
import matplotlib.pyplot as plt
from fticr_toolkit import statistics

import time
from lmfit import Parameters, minimize, fit_report

class AXIAl_FIT(object):
    def __init__(self, init_user = {}):
        init_standard = {
                'nu_z': None,
                'LO': None,
                "dip_span": 40, #Hz
                'resonator_span': 2500, #Hz
                'min_dip_depth': 5, #dB

                'vary_A': True, 
                'vary_off': False, 
                'vary_nu_res': False, 
                'vary_Q': True,

                'vary_slope': False,
                'slope': None, #dB/Hz

                'vary_nu_z_jitter': False,
                'nu_z_jitter': 0, #Hz

                'double_dips': None, #[dip_left, dip_right] #position of the dips
                'equal_widths': True,

                'print_fit_report': False,
                'plot_results': True,
                'fancy_plot': False,
                }

        init_standard.update(init_user)
        self.init_params = init_standard

    def find_nearest(self, array, test_value):
        '''
        find value in the array closest to the test_value and return its index
        '''
        array = np.asarray(array)
        idx = (np.abs(array - test_value)).argmin()
        return idx

    def cut_spec(self, x, y, center, span):
        range_idx = np.where(np.logical_and((center - span/2) <= x, x <= (center + span/2)))
        return x[range_idx], y[range_idx]


    def gaussian(self, nu, sigma):
        return np.exp( - 0.5*(nu / sigma)**2 ) / sigma / np.sqrt(2 * np.pi)

    ##############################################################

    def resonator_func(self, freq, A, Q, offset, slope, nu_res):       
        inner = A / (1 + (Q*(freq/nu_res - nu_res/freq))**2 ) + offset
        return 10*np.log10(inner) + slope*(freq - nu_res)

    def resonator_func_lmfit(self, params, freq, amp = None):
        A, Q, offset, slope, nu_res = params['A'], params['Q'], params['offset'], params['slope'], params['nu_res']
        y = self.resonator_func(freq, A, Q, offset, slope, nu_res)
        if amp is None:
            return y
        return y - amp

    ##############################################################

    def dip_fit_func(self, freq, A, nu_z, nu_res, Q, dip_width, offset, slope):
        inner = A / ( 1 + (dip_width*freq/(freq**2 - nu_z**2) - Q*(freq/nu_res - nu_res/freq))**2 ) + offset
        fit_func = 10*np.log10( inner ) + slope*( freq - nu_res )
        return fit_func

    def convolution_dip(self, freq, A, nu_z, nu_res, Q, dip_width, offset, slope, sigma, a_lim, b_lim):
        return quad(lambda x: self.dip_fit_func(freq, A, nu_z - x, nu_res, Q, dip_width, offset, slope)*self.gaussian(x, sigma), a_lim, b_lim)[0]

    def convoluted_dip_fit_func(self, freq, A, nu_z, nu_res, Q, dip_width, offset, slope, sigma): #this trick of deviding the integration into two intervals speeds up the process by a factor 1.5
        return self.convolution_dip(freq, A, nu_z, nu_res, Q, dip_width, offset, slope, sigma, -3*sigma, 0) + self.convolution_dip(freq, A, nu_z, nu_res, Q, dip_width, offset, slope, sigma, 0, 3*sigma)

    def dip_func_lmfit(self, params, freq, amp = None):
        nu_z, dip_width, A, offset, slope, nu_res, Q = params['nu_z'], params['dip_width'], params['A'], params['offset'], params['slope'], params['nu_res'], params['Q']
        if self.init_params["nu_z_jitter"] == 0:
            y = self.dip_fit_func(freq, A, nu_z, nu_res, Q, dip_width, offset, slope)
        else:
            nu_z_jitter = params['nu_z_jitter']
            y = [self.convoluted_dip_fit_func(f, A, nu_z, nu_res, Q, dip_width, offset, slope, nu_z_jitter) for f in freq]
        if amp is None:
            return y
        return y - amp

    ##############################################################

    def double_dip_fit_func(self, freq, A, nu_res, Q, offset, slope, nu_l, nu_r, dip_width_l = None, dip_width_r = None, dip_widths = None):
        if dip_widths is not None:
            dip_width_r = dip_width_l = dip_widths
        inner = A / (1 + ( dip_width_l*freq/(freq**2 - nu_l**2) + dip_width_r*freq/(freq**2 - nu_r**2) - Q*(freq/nu_res - nu_res/freq) )**2 ) + offset
        fit_func = 10*np.log10( inner ) + slope*( freq - nu_res )
        return fit_func
        
    def convolution_double_dip(self, freq, A, nu_res, Q, offset, slope, nu_l, nu_r, sigma, dip_width_l, dip_width_r, dip_widths, a_lim, b_lim):
        return quad(lambda x: self.double_dip_fit_func(freq, A, nu_res, Q, offset, slope, nu_l - x, nu_r - x, dip_width_l, dip_width_r, dip_widths)*self.gaussian(x, sigma), a_lim, b_lim)[0]

    def convoluted_double_dip_fit_func(self, freq, A, nu_res, Q, offset, slope, nu_l, nu_r, sigma, dip_width_l, dip_width_r, dip_widths): #this trick of deviding the integration into two intervals speeds up the process by a factor 1.5
        return self.convolution_double_dip(freq, A, nu_res, Q, offset, slope, nu_l, nu_r, sigma, dip_width_l, dip_width_r, dip_widths, -3*sigma, 0) + self.convolution_double_dip(freq, A, nu_res, Q, offset, slope, nu_l, nu_r, sigma, dip_width_l, dip_width_r, dip_widths, 0, 3*sigma)

    def double_dip_func_lmfit(self, params, freq, amp = None):
        nu_l, nu_r, A, offset, slope, nu_res, Q = params['nu_l'], params['nu_r'], params['A'], params['offset'], params['slope'], params['nu_res'], params['Q']
        if self.init_params["equal_widths"]:
            dip_widths = params['dip_widths']
            if self.init_params["nu_z_jitter"] == 0:
                y = self.double_dip_fit_func(freq, A, nu_res, Q, offset, slope, nu_l, nu_r, dip_widths = dip_widths)
            else:
                nu_z_jitter = params['nu_z_jitter']
                y = [self.convoluted_double_dip_fit_func(f, A, nu_res, Q, offset, slope, nu_l, nu_r, nu_z_jitter, None, None, dip_widths) for f in freq]
        else:
            dip_width_l, dip_width_r = params['dip_width_l'], params['dip_width_r']
            if self.init_params["nu_z_jitter"] == 0:
                y = self.double_dip_fit_func(freq, A, nu_res, Q, offset, slope, nu_l, nu_r, dip_width_l = dip_width_l, dip_width_r = dip_width_r)
            else:
                nu_z_jitter = params['nu_z_jitter']
                y = [self.convoluted_double_dip_fit_func(f, A, nu_res, Q, offset, slope, nu_l, nu_r, nu_z_jitter, dip_width_l, dip_width_r, None) for f in freq]
        if amp is None:
            return y
        return y - amp

    ##############################################################

    def downsample(self, freqs, amps, factor=3):
        if factor%2 != 1:
            raise ValueError("needs an odd value")
        
        cutlen = int(len(amps)/factor)-1
        subsets = None
        for i in range(factor):
            sub = amps[i::factor][:cutlen]
            #print(len(sub))
            if subsets is None:
                subsets = np.reshape(sub, (1,-1))
            else:
                subsets = np.append(subsets, [sub], axis=0)
        
        amps = np.sum(subsets, axis=0)/factor
        freqs = freqs[int(factor/2)::factor][:cutlen]

        return freqs, amps

    def fit_resonator(self, freq, amp, downsample=1):

        freq, amp = self.downsample(freq, amp, downsample)

        self.freq_resolution = freq[1] - freq[0]

        resfit_start_time = time.time()
        
        #precut the spectrum to the resonator_span
        res_span = self.init_params["resonator_span"]
        if res_span is not None:
            nu_z_init = self.init_params["nu_z"] if self.init_params["nu_z"] is not None else freq[int(len(freq)/2)] #either take nu_z from the config or just take the center of the spectrum
            freq, amp = self.cut_spec(freq, amp, nu_z_init, self.init_params["resonator_span"])

        #empirically it was found that a spectrum is well smoothed if 'window_length' parameter for 'savgol_filter' function showld be equal to 500/resolution, and 'polyorder' = 1
        window_length = int(500 / self.freq_resolution)
        if (window_length % 2) == 0: window_length = window_length + 1 # 'window_length' has to be a positive odd integer according to 'savgol_filter' specification
        amp_smoothed = savgol_filter(amp, window_length, 3)
        t = time.time() - resfit_start_time
        #print(f"--- Res fit smoothing took {int(t/60)} minutes and {round(t - int(t/60)*60)} seconds ---", flush=True)

        #plt.plot(freq, amp)
        #plt.plot(freq, amp_smoothed)
        #plt.show()

        #estimate resonator center frequency 'nu_res'
        amp_max_idx = np.argmax(amp_smoothed)
        estimated_nu_res = freq[amp_max_idx]
        
        #estimate offset
        estimated_off = amp_smoothed[0] #just the first point in the spectrum which value is almost equal to the min value of the spectrum
        estimated_off_abs_scale = 10**(estimated_off/10) #convert from dB to absolute scale

        #estimate resonator span
        amp_max = amp_smoothed[amp_max_idx]
        if res_span is None:
            resonator_FWHM_freqs = freq[amp_smoothed > (amp_max + estimated_off)/2]
            resonator_FWHM = int(resonator_FWHM_freqs[-1] - resonator_FWHM_freqs[0])
            res_span = 3*resonator_FWHM
        
        #estimate the slope using two points "left" and "right", which are symmetrically taken as far from the "estimated_nu_res" frequency as possible
        smoothing_range = (freq[-1] - freq[0])/50 #empirically it was found that the 'smoothing_range' around the two side points 'freq_left' and 'freq_right' showld be equal to res_span/50
        delta = int(smoothing_range/self.freq_resolution/2) #convert Hz to number of points
        spectrum_max_idx = len(freq)
        left_idx = 2*amp_max_idx - spectrum_max_idx
        if left_idx > 0:   #[0 . . . spectrum_center_idx . . . amp_max_idx . . . spectrum_max_idx]
            freq_left = freq[left_idx + delta]
            freq_right = freq[-delta]
            amp_left = np.mean(amp_smoothed[left_idx : left_idx + delta])
            amp_right = np.mean(amp_smoothed[-delta : -1])
        else:           #[0 . . . amp_max_idx . . . spectrum_center_idx . . . spectrum_max_idx]
            freq_left = freq[delta]
            freq_right = freq[2*amp_max_idx - delta]
            amp_left = np.mean(amp_smoothed[0 : delta])
            amp_right = np.mean(amp_smoothed[2*amp_max_idx - delta : 2*amp_max_idx])
        estimated_slope = (amp_right - amp_left) / (freq_right - freq_left)

        #correct amplitude for the slope
        amp_slope_corrected = amp_smoothed - estimated_slope*(freq - estimated_nu_res)

        #estimate resonator Q-value
        bandwidth_points = np.where( amp_slope_corrected > amp_max - 3 )[0] #By definition Q = f / HPBW, where HPBW is half-power bandwidth, and f is the center frequency. HPBW = 3 dB drop
        HPBW = freq[bandwidth_points[-1]] - freq[bandwidth_points[0]]
        estimated_Q = estimated_nu_res / HPBW 

        #estimate amplitude A
        estimated_A = 10**( amp_max /10 ) - 10**( estimated_off /10 )  #rescale amplitude without dB here, because Q and others needed dB scale
        t = time.time() - resfit_start_time
        #print(f"--- Res fit estimates took {int(t/60)} minutes and {round(t - int(t/60)*60)} seconds ---", flush=True)

        #cut spectrum to the resonator span
        freq_cut, amp_cut = self.cut_spec(freq, amp, estimated_nu_res, res_span)

        #heavily smooth the resonator's spectrum
        window_length = int(50 / self.freq_resolution) #empirical value
        if (window_length % 2) == 0: window_length = window_length + 1 # 'window_length' has to be a positive odd integer according to 'savgol_filter' specification
        amp_cut_smoothed = savgol_filter(amp_cut, window_length, 1)
        t = time.time() - resfit_start_time
        #print(f"--- Res fit cut & smoothing2 took {int(t/60)} minutes and {round(t - int(t/60)*60)} seconds ---", flush=True)

        #set constrains to the fit parameters
        params = Parameters()
        params.add('A', min = estimated_A*0.1, max = estimated_A*10, value = estimated_A)
        params.add('Q', min = estimated_Q*0.1, max = estimated_Q*5, value = estimated_Q)
        params.add('offset', min = estimated_off_abs_scale*0.1, max = estimated_off_abs_scale*10, value = estimated_off_abs_scale)
        params.add('slope', min = estimated_slope*-5, max = estimated_slope*5, value = estimated_slope)
        params.add('nu_res', min = estimated_nu_res - 10, max = estimated_nu_res + 10, value = estimated_nu_res)

        #fit the well smoothed resonator
        roughly_fitted_resonator = minimize(self.resonator_func_lmfit, params, args=(freq_cut, amp_cut_smoothed), method='least_squares') #'least_squares' method is much faster, but 'powell' can be more reliable

        t = time.time() - resfit_start_time
        #print(f"--- Res first fit took {int(t/60)} minutes and {round(t - int(t/60)*60)} seconds ---", flush=True)

        #put values of the fitted parameters into the dictionary
        roughly_fitted_resonator_results = {}
        for name, param in roughly_fitted_resonator.params.items(): #put values of the fitted parameters into the dictionary
            roughly_fitted_resonator_results[str(name)] = param.value
        #print(fit_report(roughly_fitted_resonator))

        #remove dip from the spectrum
        nu_res_idx = self.find_nearest(freq_cut, roughly_fitted_resonator_results["nu_res"])
        dip_span = self.init_params["dip_span"]
        dip_span_delta_idx = int(dip_span / 2 / self.freq_resolution)
        dip_indices = np.arange(nu_res_idx - dip_span_delta_idx, nu_res_idx + dip_span_delta_idx + 1, 1)
        freq_without_dip, amp_without_dip = np.delete(freq_cut, dip_indices), np.delete(amp_cut, dip_indices)

        #cut out noise peaks
        window_length = int(10 / self.freq_resolution) #empirical value
        if (window_length % 2) == 0: window_length = window_length + 1 # 'window_length' has to be a positive odd integer according to 'savgol_filter' specification
        amp_without_dip_smoothed = savgol_filter(amp_without_dip, window_length, 1)
        residuals = amp_without_dip - amp_without_dip_smoothed
        residual_var = np.var(residuals)
        residual_distance = np.abs(residuals) - np.mean(residuals)
        noise_peaks_indices = np.argwhere(residual_distance > 4*np.sqrt(residual_var)) #find residual_distance larger than 4 sigma  (The standard deviation is the square root of the variance)
        freq_perfect_resonator, amp_perfect_resonator = np.delete(freq_without_dip, noise_peaks_indices), np.delete(amp_without_dip, noise_peaks_indices)

        t = time.time() - resfit_start_time
        #print(f"--- Res fit dip and noise cut took {int(t/60)} minutes and {round(t - int(t/60)*60)} seconds ---", flush=True)

        #fit the resonator
        params = Parameters()
        for parameter, value in roughly_fitted_resonator_results.items():
            if parameter == 'slope':
                params.add('slope', min = value*-5, max = value*5, value = value)
            elif parameter == 'nu_res':
                params.add('nu_res', min = value - 10, max = value + 10, value = value)
            else:
                params.add(str(parameter), min = value*0.2, max = value*5, value = value)
        #amp_perfect_resonator = savgol_filter(amp_perfect_resonator, 7, 1) #?? not sure that it improves accuracy of the fitted parameters
        fit_perfect_resonator = minimize(self.resonator_func_lmfit, params, args=(freq_perfect_resonator, amp_perfect_resonator), method='least_squares') #'least_squares' method is much faster, but 'powell' can be more reliable
        #print(fit_report(fit_perfect_resonator))

        #put values of the fitted parameters into the dictionary
        perfect_resonator_fit_results = {}
        for name, param in fit_perfect_resonator.params.items(): 
            perfect_resonator_fit_results[str(name)] = param.value
            perfect_resonator_fit_results['d' + str(name)] = param.stderr
        #print(fit_report(fit_perfect_resonator))

        t = time.time() - resfit_start_time
        #print(f"--- Res fit final took {int(t/60)} minutes and {round(t - int(t/60)*60)} seconds ---", flush=True)

        #plt.plot(freq, amp, label = "resonator data")
        #plt.scatter(freq_without_dip[noise_peaks_indices], amp_without_dip[noise_peaks_indices], label = "noise peaks")
        #plt.plot(freq, self.resonator_func_lmfit(perfect_resonator_fit_results, freq), label = "resonator fit")
        #plt.xlim(nu_z_init - dip_span/1.2, nu_z_init + dip_span/1.2) 
        #plt.legend()
        #plt.show()

        return perfect_resonator_fit_results, freq_without_dip, amp_without_dip, noise_peaks_indices, freq_cut, amp_cut # everything except 'perfect_resonator_fit_results' is later needed only for plots


    def fit_axial_dip(self, freq, amp, downsample=1, resonator_fix_parameters={}):
        dummy_fit_data = {}
        dummy_fit_data["nu_z"], dummy_fit_data["dnu_z"], dummy_fit_data["dip_width"], dummy_fit_data["ddip_width"], dummy_fit_data["redchi"], dummy_fit_data["nu_res"], dummy_fit_data["dnu_res"], dummy_fit_data["fit_success"] = 0, 0, 0, 0, 0, 0, 0, False

        fit_start_time = time.time()
        resonator_fit_results, freq_without_dip, amp_without_dip, noise_peaks_indices, freq_cut, amp_cut = self.fit_resonator(freq, amp, downsample)
        dip_span = self.init_params["dip_span"]
        t = time.time() - fit_start_time
        print(f"--- Res fit took {int(t/60)} minutes and {round(t - int(t/60)*60)} seconds ---", flush=True)

        #cut the spectrum down to the size of dip_span around 'nu_res'
        freq_dip, amp_dip = self.cut_spec(freq, amp, resonator_fit_results['nu_res'], dip_span)

        #estimate nu_z
        def estimate_nu_z(freq, amp):
            window_length = int(2 / self.freq_resolution) #empirical value
            if (window_length % 2) == 0: window_length = window_length + 1 # 'window_length' has to be a positive odd integer according to 'savgol_filter' specification
            smooth_residuals = self.resonator_func_lmfit(resonator_fit_results, freq) - savgol_filter(amp, window_length, 1)
            if self.init_params["nu_z"] is not None:
                estimated_nu_z = self.init_params["nu_z"]
                estimated_nu_z_idx = self.find_nearest(freq, estimated_nu_z)
            else:
                estimated_nu_z_idx = np.argmax(smooth_residuals)
                estimated_nu_z = freq_dip[estimated_nu_z_idx]
            return estimated_nu_z, estimated_nu_z_idx, smooth_residuals
        estimated_nu_z, estimated_nu_z_idx, smooth_residuals = estimate_nu_z(freq_dip, amp_dip)

        #recenter spectrum around nu_z (this step is not really necessary since we always try to position the dip at the nu_res)
        freq_dip, amp_dip = self.cut_spec(freq, amp, estimated_nu_z, dip_span)
        estimated_nu_z, estimated_nu_z_idx, smooth_residuals = estimate_nu_z(freq_dip, amp_dip)

        #estimate dip_width
        estimated_dip_depth = smooth_residuals[estimated_nu_z_idx]
        if estimated_dip_depth < self.init_params["min_dip_depth"]: # dB
            print("!!! No dip found !!!")
            if self.init_params["LO"] is not None: #LO is typically given only in the IonWorkGUI mode
                print('Make sure that you provide the correct nu_z frequency')
            return dummy_fit_data
        dip_depth_factor = 0.3
        nu_z_left_idx, nu_z_right_idx = np.where(smooth_residuals > estimated_dip_depth * dip_depth_factor)[0][0], np.where(smooth_residuals > estimated_dip_depth * dip_depth_factor)[0][-1] #two side points where estimated_dip_depth drops by a factor (dip_depth_factor - 1)
        estimated_dip_width = (nu_z_right_idx - nu_z_left_idx) * self.freq_resolution
        check_points = np.array([nu_z_left_idx, nu_z_right_idx, estimated_nu_z_idx])

        t = time.time() - fit_start_time
        print(f"--- Cut Spec and Estimate Params took {int(t/60)} minutes and {round(t - int(t/60)*60)} seconds ---", flush=True)

        #data = self.resonator_func_lmfit(resonator_fit_results, freq_dip) - amp_dip
        #plt.plot(freq_dip, data, label = 'data')
        #plt.plot(freq_dip, smooth_residuals, label = 'residuals')
        #plt.scatter(freq_dip[check_points], residuals[check_points], color = 'black', zorder = 10)
        #plt.legend()
        #plt.show()
        #return

        #initiate fit parameters for the dip fit
        params = Parameters()
        init_value = resonator_fit_results
        init_value.update(resonator_fix_parameters)

        #these parameters are always fitted
        params.add('nu_z', min = estimated_nu_z - estimated_dip_width, max = estimated_nu_z + estimated_dip_width, value = estimated_nu_z)
        params.add('dip_width', min = estimated_dip_width*0.2, max = estimated_dip_width*5, value = estimated_dip_width)
        #these parameters can be fitted or taken from the resonator fit (specified in the config)
        if self.init_params["nu_z_jitter"] != 0:
            params.add('nu_z_jitter', vary = self.init_params["vary_nu_z_jitter"], min = 0.05, max = 9.99, value = self.init_params["nu_z_jitter"])
        params.add('A', vary = self.init_params["vary_A"], min = init_value['A']*0.1, max = init_value['A']*10, value = init_value['A'])
        params.add('Q', vary = self.init_params["vary_Q"], min = init_value['Q']*0.1, max = init_value['Q']*10, value = init_value['Q'])
        params.add('offset', vary = self.init_params["vary_off"], min = init_value['offset']*0.1, max = init_value['offset']*50, value = init_value['offset'])
        params.add('nu_res', vary = self.init_params["vary_nu_res"], min = init_value['nu_res'] - 10, max = init_value['nu_res'] + 10, value = init_value['nu_res'])
        slope = self.init_params["slope"] if self.init_params["slope"] is not None else init_value['slope']
        params.add('slope', vary = self.init_params["vary_slope"], min = init_value['slope']*-10, max = init_value['slope']*10, value = slope)

        #fit the dip
        fit_dip = minimize(self.dip_func_lmfit, params, args=(freq_dip, amp_dip), method='least_squares')

        #put values of the fitted parameters into the dictionary
        dip_fit_results = resonator_fit_results.copy()
        for name, param in fit_dip.params.items(): 
            dip_fit_results[str(name)] = param.value
            if param.stderr != 0: dip_fit_results['d' + str(name)] = param.stderr #if parameter is varied then take its uncertainty, otherwise leave uncertainty from the resonator fit
            else: pass
        dip_fit_results['fit_success'] = fit_dip.success
        dip_fit_results['redchi'] = fit_dip.redchi
        dip_fit_results['LO'] = self.init_params["LO"]

        if self.init_params["print_fit_report"]:
            print(fit_report(fit_dip))

        print('\n \n _______________________ Fit _______________________')
        t = time.time() - fit_start_time
        if t > 2:
            print(f"--- Fit took {int(t/60)} minutes and {round(t - int(t/60)*60)} seconds ---")
        #print(dip_fit_results)


        plot_start_t = time.time() 

        #plot the results
        if self.init_params["plot_results"]:
            if self.init_params["fancy_plot"]:
                plt.figure(figsize = (20, 6))
                grid = plt.GridSpec(3, 3, wspace = 0.1, hspace = 0.05)
                dip_plot = plt.subplot(grid[:2, :2])
                residuals_plot = plt.subplot(grid[2, :2])
                resonator_plot = plt.subplot(grid[0:, 2])
                dip_plot.get_shared_x_axes().join(dip_plot, residuals_plot)
                dip_plot.set_xticklabels([])

                dip_plot.plot(freq_cut, amp_cut, label = "resonator data")
                dip_plot.scatter(freq_without_dip[noise_peaks_indices], amp_without_dip[noise_peaks_indices], label = "noise peaks")
                dip_plot.plot(freq_dip, amp_dip, label = "dip data")
                dip_plot.plot(freq_cut, self.resonator_func_lmfit(resonator_fit_results, freq_cut), label = "resonator fit")
                dip_plot.plot(freq_dip, self.dip_func_lmfit(dip_fit_results, freq_dip), label = "dip fit")
                dip_plot.set_ylim([min(min(self.dip_func_lmfit(dip_fit_results, freq_dip)), min(amp_dip)) - 2, max(amp_dip) + 2])
                dip_plot.scatter(freq_dip[check_points], amp_dip[check_points], label = "check points", color = 'black', zorder=10, s = 12)
                dip_plot.legend()

                fit_residuals = amp_dip - self.dip_func_lmfit(dip_fit_results, freq_dip)
                residuals_plot.plot(freq_dip, fit_residuals, label = "dip fit residuals")
                residuals_plot.set_xlim([estimated_nu_z - dip_span*0.6, estimated_nu_z + dip_span*0.6])
                residuals_plot.axhline(0, color='black')
                residuals_plot.legend()

                resonator_plot.plot(freq_cut, amp_cut, label = "resonator data")
                resonator_plot.scatter(freq_without_dip[noise_peaks_indices], amp_without_dip[noise_peaks_indices], label = "noise peaks")
                resonator_plot.plot(freq_dip, amp_dip, label = "dip data")
                resonator_plot.plot(freq_cut, self.resonator_func_lmfit(resonator_fit_results, freq_cut), label = "resonator fit")
                #resonator_plot.plot(freq_dip, self.dip_func_lmfit(dip_fit_results, freq_dip), label = "dip fit", color = 'C3')
                resonator_plot.plot(freq_cut, self.dip_func_lmfit(dip_fit_results, freq_cut), label = "dip fit")
                resonator_plot.legend()

                t = time.time() - plot_start_t
                print(f"--- Plot took {int(t/60)} minutes and {round(t - int(t/60)*60)} seconds ---")
                plt.show()

                residuals_mean, residuals_dmean = statistics.mean_and_stderror(fit_residuals)
                formatted_residuals_mean, formatted_residuals_dmean = "{:.1e}".format(residuals_mean), "{:.1e}".format(residuals_dmean)
                print(f'mean fit residuals = {formatted_residuals_mean} +/- {formatted_residuals_dmean}')
            else:
                plt.plot(freq_cut, amp_cut, label = "resonator data")
                #plt.scatter(freq_without_dip[noise_peaks_indices], amp_without_dip[noise_peaks_indices], label = "noise peaks")
                plt.plot(freq_dip, amp_dip, label = "dip data", color = 'orange')
                plt.plot(freq_cut, self.resonator_func_lmfit(resonator_fit_results, freq_cut), label = "resonator fit", color = 'green')
                plt.plot(freq_cut, self.dip_func_lmfit(dip_fit_results, freq_cut), label = "dip fit", linewidth=1.5, color = 'red')
                plt.scatter(freq_dip[check_points], amp_dip[check_points], label = "check points", color = 'black', zorder=10, s=5)
                plt.xlim(estimated_nu_z - dip_span*0.6, estimated_nu_z + dip_span*0.6) 
                plt.ylim(min(self.dip_func_lmfit(dip_fit_results, freq_dip)) - 2, max(amp_dip) + 2) 
                plt.legend()
                plt.show()

                #the same but just zoomed in
                plt.plot(freq_cut, amp_cut, label = "resonator data")
                #plt.scatter(freq_without_dip[noise_peaks_indices], amp_without_dip[noise_peaks_indices], label = "noise peaks")
                plt.plot(freq_dip, amp_dip, label = "dip data", color = 'orange')
                plt.plot(freq_cut, self.resonator_func_lmfit(resonator_fit_results, freq_cut), label = "resonator fit", color = 'green')
                plt.plot(freq_cut, self.dip_func_lmfit(dip_fit_results, freq_cut), label = "dip fit", linewidth=1.5, color = 'red')
                plt.scatter(freq_dip[check_points], amp_dip[check_points], label = "check points", color = 'black', zorder=10, s=5)
                plt.xlim(estimated_nu_z - dip_span*0.1, estimated_nu_z + dip_span*0.1) 
                plt.ylim(min(self.dip_func_lmfit(dip_fit_results, freq_dip)) - 2, max(amp_dip) + 2) 
                plt.legend()
                plt.show()
        
        t = time.time() - plot_start_t
        if t > 2:
            print(f"--- Plot took {int(t/60)} minutes and {round(t - int(t/60)*60)} seconds ---")
        #print(dip_fit_results)

        return dip_fit_results

        #except Exception as e:
        #    print(e)
        #    return dummy_fit_data

    


    def fit_double_dip(self, freq, amp):
        dummy_fit_data = {}
        dummy_fit_data["nu_l"], dummy_fit_data["dnu_l"], dummy_fit_data["nu_r"], dummy_fit_data["dnu_r"], dummy_fit_data["dip_width"], dummy_fit_data["ddip_width"], dummy_fit_data["redchi"], dummy_fit_data["nu_res"], dummy_fit_data["dnu_res"], dummy_fit_data["fit_success"] = 0, 0, 0, 0, 0, 0, 0, 0, 0, False

        fit_start_time = time.time()
        resonator_fit_results, freq_without_dip, amp_without_dip, noise_peaks_indices, freq_cut, amp_cut = self.fit_resonator(freq, amp)
        
        #cut the spectrum down to the dip_span
        freq_dips, amp_dips = self.cut_spec(freq, amp, self.init_params["nu_z"], self.init_params["dip_span"])
        dip_span = self.init_params["dip_span"]

        #find the dips
        window_length = int(3 / self.freq_resolution) #empirical value
        if (window_length % 2) == 0: window_length = window_length + 1 # 'window_length' has to be a positive odd integer according to 'savgol_filter' specification
        smooth_residuals = self.resonator_func_lmfit(resonator_fit_results, freq_dips) - savgol_filter(amp_dips, window_length, 5)
        nu_z_idx = self.find_nearest(freq_dips, self.init_params["nu_z"])
        if self.init_params["double_dips"] is not None:
            nu_l, nu_r = self.init_params["double_dips"][0], self.init_params["double_dips"][1]
            nu_l_idx, nu_r_idx = self.find_nearest(freq_dips, nu_l), self.find_nearest(freq_dips, nu_r)
        else:
            nu_l_idx, nu_r_idx = np.argmax(smooth_residuals[0: nu_z_idx + 1]), np.argmax(smooth_residuals[nu_z_idx: -1]) + nu_z_idx
            nu_l, nu_r = freq_dips[nu_l_idx], freq_dips[nu_r_idx]

        #estimate dip_width
        estimated_dip_depth_left, estimated_dip_depth_right = smooth_residuals[nu_l_idx], smooth_residuals[nu_r_idx]
        if estimated_dip_depth_left < self.init_params["min_dip_depth"] or estimated_dip_depth_right < self.init_params["min_dip_depth"]: # dB
            print("!!! No dip found !!!")
            return dummy_fit_data
        dip_depth_factor = 0.4 #relatively high value to be on the safe side
        nu_l_left_idx, nu_l_right_idx = np.where(smooth_residuals[0: nu_l_idx + 1] > estimated_dip_depth_left * dip_depth_factor)[0][0], np.where(smooth_residuals[nu_l_idx: nu_z_idx + 1] > estimated_dip_depth_left * dip_depth_factor)[0][-1] + nu_l_idx #two side points where estimated_dip_depth drops by a factor (dip_depth_factor - 1)
        nu_r_left_idx, nu_r_right_idx = np.where(smooth_residuals[nu_z_idx: nu_r_idx + 1] > estimated_dip_depth_right * dip_depth_factor)[0][0] + nu_z_idx, np.where(smooth_residuals[nu_r_idx: -1] > estimated_dip_depth_right * dip_depth_factor)[0][-1] + nu_r_idx #two side points where estimated_dip_depth drops by a factor (dip_depth_factor - 1)
        estimated_dip_width_l = (nu_l_right_idx - nu_l_left_idx) * self.freq_resolution
        estimated_dip_width_r = (nu_r_right_idx - nu_r_left_idx) * self.freq_resolution            
        check_points = np.array([nu_l_left_idx, nu_l_right_idx, nu_r_left_idx, nu_r_right_idx, nu_z_idx, nu_l_idx, nu_r_idx])

        #data = self.resonator_func_lmfit(resonator_fit_results, freq_dips) - amp_dips
        #plt.plot(freq_dips, data)
        #plt.plot(freq_dips, smooth_residuals)
        #plt.scatter(freq_dips[check_points], smooth_residuals[check_points], color = 'black', zorder = 10)
        #plt.show()
        #return
        
        #double dip fit
        params = Parameters()
        init_value = resonator_fit_results

        #these parameters are always fitted
        params.add('nu_l', min = nu_l - estimated_dip_width_l, max = nu_l + estimated_dip_width_l, value = nu_l)
        params.add('nu_r', min = nu_r - estimated_dip_width_r, max = nu_r + estimated_dip_width_r, value = nu_r)
        if self.init_params["equal_widths"]:
            estimated_dip_widths = (estimated_dip_width_l + estimated_dip_width_r)/2
            params.add('dip_widths', min = estimated_dip_widths*0.1, max = estimated_dip_widths*10, value = estimated_dip_widths)
        else:
            params.add('dip_width_l', min = estimated_dip_width_l*0.1, max = estimated_dip_width_l*10, value = estimated_dip_width_l)
            params.add('dip_width_r', min = estimated_dip_width_r*0.1, max = estimated_dip_width_r*10, value = estimated_dip_width_r)

        #these parameters can be fitted or taken from the resonator fit (can be specified in the config)
        if self.init_params["nu_z_jitter"] != 0:
            params.add('nu_z_jitter', vary = self.init_params["vary_nu_z_jitter"], min = 0.05, max = 1, value = self.init_params["nu_z_jitter"])
        params.add('A', vary = self.init_params["vary_A"], min = init_value['A']*0.2, max = init_value['A']*5, value = init_value['A'])
        params.add('Q', vary = self.init_params["vary_Q"], min = init_value['Q']*0.2, max = init_value['Q']*5, value = init_value['Q'])
        params.add('nu_res', vary = self.init_params["vary_nu_res"], min = init_value['nu_res'] - 5, max = init_value['nu_res'] + 5, value = init_value['nu_res'])
        slope = self.init_params["slope"] if self.init_params["slope"] is not None else init_value['slope']
        params.add('slope', vary = self.init_params["vary_slope"], min = init_value['slope']*-10, max = init_value['slope']*10, value = slope)
        params.add('offset', vary = self.init_params["vary_off"], min = init_value['offset']*0.1, max = init_value['offset']*10, value = init_value['offset'])

        #fit the double dip
        fit_double_dip = minimize(self.double_dip_func_lmfit, params, args=(freq_dips, amp_dips), method='least_squares')

        if self.init_params["print_fit_report"]:
            print(fit_report(fit_double_dip))

        #put values of the fitted parameters into the dictionary
        double_dip_fit_results = resonator_fit_results.copy()
        for name, param in fit_double_dip.params.items(): 
            double_dip_fit_results[str(name)] = param.value
            if param.stderr != 0: double_dip_fit_results['d' + str(name)] = param.stderr #if parameter is varied then take its uncertainty, otherwise leave uncertainty from the resonator fit
            else: pass
        double_dip_fit_results['fit_success'] = fit_double_dip.success
        double_dip_fit_results['redchi'] = fit_double_dip.redchi
        double_dip_fit_results['LO'] = self.init_params["LO"]
        double_dip_fit_results['nu_z'] = self.init_params["nu_z"]

        t = time.time() - fit_start_time
        if t > 2:
            print(f"--- Fit took {int(t/60)} minutes and {round(t - int(t/60)*60)} seconds ---")

        #plot the results
        if self.init_params["plot_results"]:
            if self.init_params["fancy_plot"]:
                plt.figure(figsize = (20, 6))
                grid = plt.GridSpec(3, 3, wspace = 0.1, hspace = 0.05)
                dip_plot = plt.subplot(grid[:2, :2])
                residuals_plot = plt.subplot(grid[2, :2])
                resonator_plot = plt.subplot(grid[0:, 2])
                dip_plot.get_shared_x_axes().join(dip_plot, residuals_plot)
                dip_plot.set_xticklabels([])

                dip_plot.plot(freq_cut, amp_cut, label = "resonator data")
                dip_plot.scatter(freq_without_dip[noise_peaks_indices], amp_without_dip[noise_peaks_indices], label = "noise peaks")
                dip_plot.plot(freq_dips, amp_dips, label = "dip data")
                dip_plot.plot(freq_cut, self.resonator_func_lmfit(resonator_fit_results, freq_cut), label = "resonator fit")
                dip_plot.plot(freq_dips, self.double_dip_func_lmfit(double_dip_fit_results, freq_dips), label = "dip fit")
                dip_plot.set_ylim([min(self.double_dip_func_lmfit(double_dip_fit_results, freq_dips)) - 2, max(amp_dips) + 2])
                dip_plot.scatter(freq_dips[check_points], amp_dips[check_points], label = "check points", color = 'black', zorder=10, s = 12)
                dip_plot.legend()

                fit_residuals = amp_dips - self.double_dip_func_lmfit(double_dip_fit_results, freq_dips)
                residuals_plot.plot(freq_dips, fit_residuals, label = "dip fit residuals")
                residuals_plot.set_xlim([self.init_params["nu_z"] - dip_span*0.6, self.init_params["nu_z"] + dip_span*0.6])
                residuals_plot.axhline(0, color='black')
                residuals_plot.legend()

                resonator_plot.plot(freq_cut, amp_cut, label = "resonator data")
                resonator_plot.scatter(freq_without_dip[noise_peaks_indices], amp_without_dip[noise_peaks_indices], label = "noise peaks")
                resonator_plot.plot(freq_dips, amp_dips, label = "dip data")
                resonator_plot.plot(freq_cut, self.resonator_func_lmfit(resonator_fit_results, freq_cut), label = "resonator fit")
                #resonator_plot.plot(freq_dips, self.double_dip_func_lmfit(double_dip_fit_results, freq_dips), label = "dip fit")
                resonator_plot.plot(freq_cut, self.double_dip_func_lmfit(double_dip_fit_results, freq_cut), label = "dip fit")
                resonator_plot.legend()
                plt.show()

                residuals_mean, residuals_dmean = statistics.mean_and_stderror(fit_residuals)
                formatted_residuals_mean, formatted_residuals_dmean = "{:.1e}".format(residuals_mean), "{:.1e}".format(residuals_dmean)
                print(f'mean fit residuals = {formatted_residuals_mean} +- {formatted_residuals_dmean}')
            else:
                plt.plot(freq_cut, amp_cut, label = "resonator data")
                #plt.scatter(freq_without_dip[noise_peaks_indices], amp_without_dip[noise_peaks_indices], label = "noise peaks")
                plt.plot(freq_dips, amp_dips, label = "dip data", color = 'orange')
                plt.plot(freq_cut, self.resonator_func_lmfit(resonator_fit_results, freq_cut), label = "resonator fit", color = 'green')
                plt.plot(freq_cut, self.double_dip_func_lmfit(double_dip_fit_results, freq_cut), label = "dip fit", linewidth=1.5, color = 'red')
                plt.scatter(freq_dips[check_points], amp_dips[check_points], label = "check points", color = 'black', zorder=10, s=5)
                plt.xlim(self.init_params["nu_z"] - dip_span*0.6, self.init_params["nu_z"] + dip_span*0.6) 
                plt.ylim(min(self.double_dip_func_lmfit(double_dip_fit_results, freq_dips)) - 2, max(amp_dips) + 2)
                plt.legend()
                plt.show()

        return double_dip_fit_results
        

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    '''
    init_user = {'nu_z': None, #504450.98 #501492.7, #8076.75 + 728000, #736074.2
                "dip_span": 70, #Hz
                'resonator_span': 3000, #Hz
                'LO': None,#475000,

                'vary_A': True, 
                'vary_off': False, 
                'vary_slope': False,
                'slope': None,
                'vary_nu_res': True, 
                'vary_Q': True,
                'vary_nu_z_jitter': False,
                'nu_z_jitter': 0,
                'equal_widths': True,

                'print_fit_report': True,
                'plot_results': True,
                'fancy_plot': True,
    }

    from pathlib import Path
    import h5py
    from fticr_toolkit import data_conversion

    #measurement_folder = "\\\\samba1\\PENTATRAP\\_MEMBERS_\\Pavel\\Ne-C_measurements\\12C6+_20Ne10+_12C6+_8"
    measurement_folder = "\\\\samba1\\PENTATRAP\\_MEMBERS_\\Pavel\\analysis\\163Dy40+_163Ho40+_163Dy40+_13"

    input_file = 'pnp_dip_unwrap.fhds'
    output_folder = "./part1_data/"
    use_settings_file = False
    settings = {
        "grouping": [0,0,0,0,0,0,0,0,0,0], # use -1 to not use the axial
        "reuse_averages": False,
        "reuse_fitdata": True,
        "single_axial_fits": False,
        "fixed_phase_readout": False
    }
    data, meas_config, output_folder = data_conversion.load_data(measurement_folder=measurement_folder, 
                          input_data=input_file, 
                          output_folder="part1_data",
                          measurement_script = "unwrap_n_measure_repeat"
                         )

    if use_settings_file:
        settings.update( data_conversion.load_settings(measurement_folder) )

    axial_avg_all_filename = Path(output_folder, "axial_avg_all.hdf5")
    path = 'mcycle1/scycle4/position_2/trap3/avggrp0'

    with h5py.File(axial_avg_all_filename, 'r') as fobj:
        dset = fobj[path]
        spec = data_conversion.Spectrum(dset)

    fit_obj = AXIAl_FIT(init_user)
    print(fit_obj.fit_axial_dip(spec.freqs, spec.amps))

    #plt.plot(spec.freqs, spec.amps)
    #plt.show()

    '''

    init_user = {'nu_z': 707318.854, #501546.034, #12054 + 724000, #8076.75 + 728000, #501492.7
                "dip_span": 20, #Hz
                'resonator_span': 3000, #Hz
                'min_dip_depth': 4.0,
                'LO': 692000, #475000,

                'vary_A': False, 
                'vary_off': False, 
                'vary_slope': False,
                'slope': 0.0,
                'vary_nu_res': False, 
                'vary_Q': False,
                'vary_nu_z_jitter': True,
                'nu_z_jitter': 0.016,
                'equal_widths': False,
                'print_fit_report': True,
                'plot_results': False,
                'fancy_plot': True,
    }

    trap = 2
    downsample = 3

    if trap == 3:
        init_user["nu_z"] = 472658.774
        init_user["dip_span"] = 80
        init_user["vary_A"] = True
        init_user["vary_nu_res"] = True
        init_user["vary_Q"] = True
        init_user["LO"] = 460000

    import pandas as pd

    df = pd.DataFrame()

    for i in range(1, 3):

        #path = '\\\\samba1\\PENTATRAP\\_MEMBERS_\\Pavel\\analysis\\CHAR_B1_Dip_Ne_trap2\\pnp_dip_unwrap\\cycle3\\shift_0.01\\dip_measurement\\cycle1\\position_1\\trap2\\cidx_2_2.spec'
        #path = '\\\\samba1\\PENTATRAP\\_MEMBERS_\\Pavel\\analysis\\test_spectra\\trap3_cyc_0_0.08V.spec'
        path = 'G:\\Yb\\172176_var\\176Yb42+_172Yb42+_176Yb42+_var_8\\pnp_dip_unwrap\\cycle1\\pnp_measurement\\cycle'+str(i)+'\\position_1\\trap'+str(trap)+'\\time_100.05_nu_z_avg_0_atsi_1.spec'
        with open(path, 'r') as file_obj:
                #file_obj = self.open(mode="r")
                data = np.fromfile(file_obj, dtype=np.float32) 
                start, stop, num = data[0:3]
                data = data[3:].reshape((2, int(data[3:].size/2)))
                freq = np.linspace(float(start),float(stop),int(num))
                data = np.asarray([freq, data[0], data[1]])

        #print(data[0][0])
        #plt.plot(data[0], data[1])
        #plt.show()

        fit_obj = AXIAl_FIT(init_user)
        #print(fit_obj.fit_double_dip(data[0], data[1]))
        #print(fit_obj.fit_axial_dip(data[0], data[1]))

        dip_fit_results = fit_obj.fit_axial_dip(data[0], data[1], downsample=1)
        dip_fit_results2 = fit_obj.fit_axial_dip(data[0], data[1], downsample=downsample)
        dfr2 = {}

        for key, val in dip_fit_results2.items():
            dfr2[key+"2"] = val

        dip_fit_results.update(dfr2)

        series1 = pd.Series(dip_fit_results)
        df = df.append(series1, ignore_index=True)

    print(df)
    N = len(df)
    x = np.arange(N)
    x2 = x + 0.1

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("downsample comparison trap"+str(trap))

    diff = df.nu_z2 - df.nu_z
    mdiff, dmdiff = statistics.mean_and_error(diff)
    axs[0, 0].errorbar(x, diff, yerr=df.dnu_z)
    axs[0, 0].hlines(0, min(x), max(x))
    axs[0, 0].errorbar(np.mean(x), mdiff, yerr=dmdiff, capsize=10, elinewidth=3 )
    axs[0, 0].scatter(np.mean(x), mdiff, marker='o', s=10)
    axs[0, 0].set_title("mean(diff) = " + str( np.around(mdiff, 6) ) + "  mean(diff)/sigma(diff) = " + str( np.around(mdiff/dmdiff, 3) ))
    axs[0, 0].set_ylabel("diff nu_z")

    diff = df.nu_res2 - df.nu_res
    mdiff, dmdiff = statistics.mean_and_error(diff)
    axs[0, 1].errorbar(x, diff, yerr=df.dnu_res)
    axs[0, 1].hlines(0, min(x), max(x))
    axs[0, 1].errorbar(np.mean(x), mdiff, yerr=dmdiff, capsize=10, elinewidth=3 )
    axs[0, 1].scatter(np.mean(x), mdiff, marker='o', s=10)
    axs[0, 1].set_title("mean(diff) = " + str( np.around(mdiff, 6) ) + "  mean(diff)/sigma(diff) = " + str( np.around(mdiff/dmdiff, 3) ))
    axs[0, 1].set_ylabel("diff nu_res")

    diff = df.dip_width2 - df.dip_width
    mdiff, dmdiff = statistics.mean_and_error(diff)
    axs[1, 0].errorbar(x, diff, yerr=df.ddip_width)
    axs[1, 0].hlines(0, min(x), max(x))
    axs[1, 0].errorbar(np.mean(x), mdiff, yerr=dmdiff, capsize=10, elinewidth=3 )
    axs[1, 0].scatter(np.mean(x), mdiff, marker='o', s=10)
    axs[1, 0].set_title("mean(diff) = " + str( np.around(mdiff, 6) ) + "  mean(diff)/sigma(diff) = " + str( np.around(mdiff/dmdiff, 3) ))
    axs[1, 0].set_ylabel("diff dip_width")

    diff = df.Q2 - df.Q
    mdiff, dmdiff = statistics.mean_and_error(diff)
    axs[1, 1].errorbar(x, diff, yerr=df.dQ)
    axs[1, 1].hlines(0, min(x), max(x))
    axs[1, 1].errorbar(np.mean(x), mdiff, yerr=dmdiff, capsize=10, elinewidth=3 )
    axs[1, 1].scatter(np.mean(x), mdiff, marker='o', s=10)
    axs[1, 1].set_title("mean(diff) = " + str( np.around(mdiff, 6) ) + "  mean(diff)/sigma(diff) = " + str( np.around(mdiff/dmdiff, 3) ))
    axs[1, 1].set_ylabel("diff Q")

    plt.savefig("fit_compare_trap"+str(trap)+"_downs"+str(downsample)+".png", dpi=300)
    plt.show()

