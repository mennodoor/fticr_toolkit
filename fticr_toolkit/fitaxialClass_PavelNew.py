'''
These axial dip fit methods are written by P. Filianin, strongly based on the methods written by D. Lange for his Bachelor Thesis.
'''

import numpy as np
# suppress error message 'RuntimeWarning: divide by zero encountered in true_divide' in the dip function calculation
np.seterr(divide='ignore', invalid='ignore')

from scipy.signal import savgol_filter
from scipy.integrate import quad
import matplotlib.pyplot as plt

import time
from lmfit import Parameters, minimize, fit_report

class AXIAl_FIT(object):
    def __init__(self, init_user = {}):
        init_standard = {
                'fit_method': 'least_squares', #the 'least_squares' method always estimes the errors well, unlike the 'leastsq' method

                'nu_z': None,
                'LO': None,
                'dip_fit_span': 40, #Hz
                'spec_span': 3000, #Hz
                'dip_width': None, #Hz <- cutout span around nu_z for the resonator fit (if None then 'dip_width' = 'dip_fit_span')
                'double_dip_width': None, #Hz <- cutout span around nu_z for the resonator fit (if None then 'double_dip_width' = 'dip_fit_span')
                'min dip_depth': 0.5, #dB

                'vary A': False, 
                'vary offset': False, 
                'vary nu_res': False, 
                'vary Q': False,

                'vary slope': False,
                'fixed slope': None, #dB/Hz

                'vary nu_z_jitter': False,
                'fixed nu_z_jitter': None, #Hz

                'double dips': None, #[dip_left, dip_right] <- position of the dips
                'equal widths': False,

                'print fit_report': False,
                'plot results': False,

                'plot_save_dir': None,
                'return figure': False,
                'print fit_evaluation_time': False,
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

    def cut_array(self, x, center, span):
        range_idx = np.where(np.logical_and((center - span/2) <= x, x <= (center + span/2)))
        return x[range_idx]


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

        '''
        convolution = []
        x = self.cut_array(freq, nu_z, sigma*6)
        for f in freq:
            if f >= nu_z - 3*sigma and f <= nu_z + 3*sigma:
                h = self.dip_fit_func(f, A, nu_z - x, nu_res, Q, dip_width, offset, slope)*self.gaussian(x, sigma)
                conv = simps(h, x)
            else:
                conv = self.dip_fit_func(freq, A, nu_z, nu_res, Q, dip_width, offset, slope)
            
            
            convolution.append(conv) 
        return conv
        '''

    def dip_func_lmfit(self, params, freq, amp = None):
        nu_z, dip_width, A, offset, slope, nu_res, Q = params['nu_z'], params['dip_width'], params['A'], params['offset'], params['slope'], params['nu_res'], params['Q']
        if not self.init_params['vary nu_z_jitter'] and self.init_params['fixed nu_z_jitter'] is None:
            y = self.dip_fit_func(freq, A, nu_z, nu_res, Q, dip_width, offset, slope)
        elif self.init_params['fixed nu_z_jitter'] is not None:
            nu_z_jitter = self.init_params['fixed nu_z_jitter']
            y = [self.convoluted_dip_fit_func(f, A, nu_z, nu_res, Q, dip_width, offset, slope, nu_z_jitter) for f in freq]
        
        y = np.array(y)
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
        if self.init_params['equal widths']:
            dip_widths = params['dip_widths']
            if not self.init_params['vary nu_z_jitter'] and self.init_params['fixed nu_z_jitter'] is None:
                y = self.double_dip_fit_func(freq, A, nu_res, Q, offset, slope, nu_l, nu_r, dip_widths = dip_widths)
            elif not self.init_params['vary nu_z_jitter'] and self.init_params['fixed nu_z_jitter'] is not None:
                nu_z_jitter = self.init_params['fixed nu_z_jitter']
                y = [self.convoluted_double_dip_fit_func(f, A, nu_res, Q, offset, slope, nu_l, nu_r, nu_z_jitter, None, None, dip_widths) for f in freq]
            elif self.init_params['vary nu_z_jitter']:
                nu_z_jitter = params['nu_z_jitter']
                y = [self.convoluted_double_dip_fit_func(f, A, nu_res, Q, offset, slope, nu_l, nu_r, nu_z_jitter, None, None, dip_widths) for f in freq]
        else:
            dip_width_l, dip_width_r = params['dip_width_l'], params['dip_width_r']
            if not self.init_params['vary nu_z_jitter'] and self.init_params['fixed nu_z_jitter'] is None:
                y = self.double_dip_fit_func(freq, A, nu_res, Q, offset, slope, nu_l, nu_r, dip_width_l = dip_width_l, dip_width_r = dip_width_r)
            elif not self.init_params['vary nu_z_jitter'] and self.init_params['fixed nu_z_jitter'] is not None:
                nu_z_jitter = self.init_params['fixed nu_z_jitter']
                y = [self.convoluted_double_dip_fit_func(f, A, nu_res, Q, offset, slope, nu_l, nu_r, nu_z_jitter, dip_width_l, dip_width_r, None) for f in freq]
            elif self.init_params['vary nu_z_jitter']:
                nu_z_jitter = params['nu_z_jitter']
                y = [self.convoluted_double_dip_fit_func(f, A, nu_res, Q, offset, slope, nu_l, nu_r, nu_z_jitter, dip_width_l, dip_width_r, None) for f in freq]
        
        y = np.array(y)
        if amp is None:
            return y
        return y - amp

    ##############################################################


    def fit_resonator(self, freq, amp, fit_double_dip = False, show = False):
        self.freq_resolution = freq[1] - freq[0]

        #smooth the raw spectrum
        window_length = int(30 / self.freq_resolution) #number 30 quickly and sufficiently smooths the spectrum (the larger this number, the smoother the spectrum and the longer smoothening takes)
        if (window_length % 2) == 0: window_length = window_length + 1 # 'window_length' has to be a positive odd integer according to 'savgol_filter' specification
        if window_length < 3: window_length = 3
        amp_smoothed = savgol_filter(amp, window_length, 1)

        #plt.plot(freq, amp)
        #plt.plot(freq, amp_smoothed)
        #plt.show()

        #estimate the resonator center frequency
        amp_max_idx = np.argmax(amp_smoothed)
        estimated_nu_res = freq[amp_max_idx]

        #precut the spectrum to the spec_span
        res_span = self.init_params["spec_span"]
        dip_fit_span = self.init_params['dip_fit_span']
        if res_span is not None:
            freq, amp = self.cut_spec(freq, amp, estimated_nu_res, self.init_params["spec_span"])

            #after cutting the spectrum find the resonator center frequency again
            amp_max_idx = self.find_nearest(freq, estimated_nu_res)
            estimated_nu_res = freq[amp_max_idx]

            #smooth the precut spectrum
            window_length = int(30 / self.freq_resolution)
            if (window_length % 2) == 0: window_length = window_length + 1 # 'window_length' has to be a positive odd integer according to 'savgol_filter' specification
            if window_length < 3: window_length = 3
            amp_smoothed = savgol_filter(amp, window_length, 1)
            
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

        #cut spectrum to the resonator span
        if not fit_double_dip:
            freq_cut, amp_cut = self.cut_spec(freq, amp, estimated_nu_res, res_span)
            _, amp_smoothed = self.cut_spec(freq, amp_smoothed, estimated_nu_res, res_span)
        else:
            freq_cut, amp_cut = self.cut_spec(freq, amp, self.init_params['nu_z'], res_span)
            _, amp_smoothed = self.cut_spec(freq, amp_smoothed, self.init_params['nu_z'], res_span)

        #set constrains to the fit parameters
        params = Parameters()
        params.add('A', min = estimated_A*0.1, max = estimated_A*10, value = estimated_A)
        params.add('Q', min = estimated_Q*0.1, max = estimated_Q*5, value = estimated_Q)
        params.add('offset', min = estimated_off_abs_scale*0.1, max = estimated_off_abs_scale*10, value = estimated_off_abs_scale)
        params.add('slope', min = estimated_slope*-5, max = estimated_slope*5, value = estimated_slope)
        params.add('nu_res', min = estimated_nu_res - 10, max = estimated_nu_res + 10, value = estimated_nu_res)

        #fit the well smoothed resonator
        roughly_fitted_resonator = minimize(self.resonator_func_lmfit, params, args=(freq_cut, amp_smoothed), method='least_squares')

        #put values of the fitted parameters into the dictionary
        roughly_fitted_resonator_results = {}
        for name, param in roughly_fitted_resonator.params.items(): #put values of the fitted parameters into the dictionary
            roughly_fitted_resonator_results[str(name)] = param.value
        #print(fit_report(roughly_fitted_resonator))

        
        dip_exists = True
        estimated_dip = self.init_params['nu_z']
        #search for the dip (this step is done inside the resonator fit function because if the dip exists then cut it out for better fit of the resonator)
        if not fit_double_dip:
            window_length = int(2 / self.freq_resolution) #empirical value
            if (window_length % 2) == 0: window_length = window_length + 1 # 'window_length' has to be a positive odd integer according to 'savgol_filter' specification
            if window_length < 3: window_length = 3
            smooth_residuals = self.resonator_func_lmfit(roughly_fitted_resonator_results, freq_cut) - savgol_filter(amp_cut, window_length, 1)

            if estimated_dip is not None:
                estimated_nu_z_idx = self.find_nearest(freq, estimated_dip)
            else:
                estimated_nu_z_idx = np.argmax(smooth_residuals)
                estimated_dip = freq[estimated_nu_z_idx]
                estimated_dip_depth = smooth_residuals[estimated_nu_z_idx]
                if estimated_dip_depth < self.init_params['min dip_depth']: # dB
                    dip_exists = False

            #if dip exists, then cutout 'dip_width' span around nu_z
            if dip_exists:
                cutout_span = self.init_params['dip_width'] if self.init_params['dip_width'] is not None else dip_fit_span
                cutout_delta_idx = int(cutout_span / 2 / self.freq_resolution)
                dip_indices = np.arange(estimated_nu_z_idx - cutout_delta_idx, estimated_nu_z_idx + cutout_delta_idx + 1, 1)
                freq_cut, amp_cut = np.delete(freq_cut, dip_indices), np.delete(amp_cut, dip_indices)

        #search for the double dip
        else:
            #cut the spectrum down to the dip_fit_span
            freq_dips, amp_dips = self.cut_spec(freq, amp, self.init_params['nu_z'], dip_fit_span)
            
            window_length = int(3 / self.freq_resolution) #empirical value
            if (window_length % 2) == 0: window_length = window_length + 1 # 'window_length' has to be a positive odd integer according to 'savgol_filter' specification
            if window_length < 7: window_length = 7
            smooth_residuals = self.resonator_func_lmfit(roughly_fitted_resonator_results, freq_dips) - savgol_filter(amp_dips, window_length, 5)

            nu_z_idx = self.find_nearest(freq_dips, self.init_params['nu_z'])
            if self.init_params['double dips'] is not None:
                nu_l, nu_r = self.init_params['double dips'][0], self.init_params['double dips'][1]
                nu_l_idx, nu_r_idx = self.find_nearest(freq_dips, nu_l), self.find_nearest(freq_dips, nu_r)
                min_dip_distance = 0.1 #Hz <- the minimum distance between two dips to consider that the double dip exists
            else:
                nu_l_idx, nu_r_idx = np.argmax(smooth_residuals[0: nu_z_idx + 1]), np.argmax(smooth_residuals[nu_z_idx: -1]) + nu_z_idx
                nu_l, nu_r = freq_dips[nu_l_idx], freq_dips[nu_r_idx]
                min_dip_distance = 1 #Hz <- the minimum distance between two dips to consider that the double dip exists

            estimated_dip_depth_left, estimated_dip_depth_right = smooth_residuals[nu_l_idx], smooth_residuals[nu_r_idx]
            if estimated_dip_depth_left < self.init_params['min dip_depth'] or estimated_dip_depth_right < self.init_params['min dip_depth'] or (nu_r - nu_l) < min_dip_distance: # dB
                dip_exists = False

            #if dips exist, then cutout 'double_dip_width' span around nu_z
            if dip_exists:
                estimated_dip = nu_l, nu_r, nu_z_idx, nu_l_idx, nu_r_idx
                nu_z_idx = self.find_nearest(freq_cut, self.init_params['nu_z'])
                cutout_span = self.init_params['double_dip_width'] if self.init_params['double_dip_width'] is not None else dip_fit_span
                cutout_delta_idx = int(cutout_span / 2 / self.freq_resolution)
                dip_indices = np.arange(nu_z_idx - cutout_delta_idx, nu_z_idx + cutout_delta_idx + 1, 1)
                freq_cut, amp_cut = np.delete(freq_cut, dip_indices), np.delete(amp_cut, dip_indices)

        #cut out noise peaks
        noise_peaks_indices = []
        window_length = int(10 / self.freq_resolution) #empirical value
        if (window_length % 2) == 0: window_length = window_length + 1 # 'window_length' has to be a positive odd integer according to 'savgol_filter' specification
        if window_length < 3: window_length = 3
        amp_without_dip_smoothed = savgol_filter(amp_cut, window_length, 1)
        residuals = amp_cut - amp_without_dip_smoothed
        residual_var = np.var(residuals)
        residual_distance = np.abs(residuals) - np.mean(residuals)
        noise_peaks_indices = np.argwhere(residual_distance > 4*np.sqrt(residual_var)) #find residual_distance larger than 4 sigma  (The standard deviation is the square root of the variance)
        freq_without_noise, amp_without_noise = np.delete(freq_cut, noise_peaks_indices), np.delete(amp_cut, noise_peaks_indices)
        noise_peaks = (freq_cut[noise_peaks_indices], amp_cut[noise_peaks_indices])

        #fit the resonator
        params = Parameters()
        for parameter, value in roughly_fitted_resonator_results.items():
            if parameter == 'slope':
                params.add('slope', min = value*-5, max = value*5, value = value)
            elif parameter == 'nu_res':
                params.add('nu_res', min = value - 30, max = value + 30, value = value)
            else:
                params.add(str(parameter), min = value*0.1, max = value*10, value = value)

        fit_perfect_resonator = minimize(self.resonator_func_lmfit, params, args=(freq_without_noise, amp_without_noise), method='least_squares')
        #print(fit_report(fit_perfect_resonator))

        #put values of the fitted parameters into the dictionary
        resonator_fit_results = {}
        for name, param in fit_perfect_resonator.params.items(): 
            resonator_fit_results[str(name)] = param.value
            resonator_fit_results['d' + str(name)] = param.stderr
        #print(fit_report(fit_perfect_resonator))

        if show:
            if self.init_params['print fit_report']:
                print(fit_report(fit_perfect_resonator))

            if self.init_params['LO'] is not None: #basically if done in the IonWork_GUI (LO provided only in the GUI)
                fig = plt.figure(figsize = (13, 7))
            else:
                fig = plt.figure(figsize = (24, 8))

            grid = plt.GridSpec(3, 1, wspace = 0.1, hspace = 0.05)
            resonator_plot, residuals_plot = plt.subplot(grid[:2, 0]), plt.subplot(grid[2, 0])
            resonator_plot.get_shared_x_axes().join(resonator_plot, residuals_plot)
            resonator_plot.set_xticklabels([])

            freq_cut_with_dip, _ = self.cut_spec(freq, amp, estimated_nu_res, res_span)
            resonator_plot.plot(freq_cut, amp_cut, label = 'data')
            resonator_plot.scatter(noise_peaks[0], noise_peaks[1], label = 'noise peaks', color = 'blue')
            resonator_plot.plot(freq_cut_with_dip, self.resonator_func_lmfit(resonator_fit_results, freq_cut_with_dip), label = 'fit', color = 'lime')
            resonator_plot.legend()

            fit_residuals = amp_without_noise - self.resonator_func_lmfit(resonator_fit_results, freq_without_noise)
            residuals_plot.plot(freq_without_noise, fit_residuals, label = 'residuals')
            residuals_plot.axhline(0, color='black')
            residuals_plot.legend()
            plt.show()

            return resonator_fit_results, fig

        return resonator_fit_results, dip_exists, estimated_dip, noise_peaks


    def fit_axial_dip(self, freq, amp):
        #make the dummy fit results to return in case the dip does not exist
        dummy_keys = ['nu_z', 'dnu_z', 'dip_width', 'ddip_width', 'redchi', 'nu_res_glob', 'dnu_res_glob', 'A', 'dA', 'Q', 'dQ' ,'slope', 'dslope', 'offset', 'doffset', 'nu_z_jitter', 'dnu_z_jitter', 'fit_success']
        dummy_values = [0]*17 + [False]
        dummy_fit_data = dict(zip(dummy_keys, dummy_values))

        fit_start_time = time.time()
        resonator_fit_results, dip_exists, estimated_dip, noise_peaks = self.fit_resonator(freq, amp)
        dummy_fit_data.update(resonator_fit_results)
        dip_fit_span = self.init_params['dip_fit_span']

        if not dip_exists:
            print(f"No dip found in  ->  {self.init_params['plot_save_dir']}")
            center = resonator_fit_results['nu_res'] if self.init_params['nu_z'] is None else self.init_params['nu_z']
            freq_dip, amp_dip = self.cut_spec(freq, amp, center, dip_fit_span*3)
            fig = plt.figure(figsize = (13, 7))
            plt.plot(freq_dip, amp_dip, label = 'data')
            plt.plot(freq_dip, self.resonator_func_lmfit(resonator_fit_results, freq_dip), label = 'resonator fit', color = 'green')
            #plt.xlim(resonator_fit_results['nu_res'] - dip_fit_span*1.5, resonator_fit_results['nu_res'] + dip_fit_span*1.5)
            #plt.ylim( min(amp_dip)-1, max(amp_dip)+1 )
            plt.legend()
            if self.init_params['plot_save_dir'] is not None:
                plt.savefig(self.init_params['plot_save_dir'] +'.png', bbox_inches='tight')
                plt.close()
            if self.init_params['plot results']: 
                plt.show()

            if self.init_params["return figure"]:
                dummy_fit_data['figure'] = fig

            return dummy_fit_data


        #cut spectrum down to the resonator span
        freq_cut, amp_cut = self.cut_spec(freq, amp, resonator_fit_results['nu_res'], self.init_params["spec_span"])

        #cut spectrum down to the dip span
        freq_dip, amp_dip = self.cut_spec(freq_cut, amp_cut, estimated_dip, dip_fit_span)
        if self.init_params['dip_width'] is not None and self.init_params['dip_width'] < dip_fit_span:
            freq_dip_cut, amp_dip_cut = self.cut_spec(freq_dip, amp_dip, estimated_dip, self.init_params['dip_width'])
        else:
            freq_dip_cut, amp_dip_cut = freq_dip, amp_dip

        #remove noise peaks
        noise_peaks_indices = []
        for idx, f in enumerate(freq_dip):
            if f in noise_peaks[0]:
                noise_peaks_indices.append(idx)
        freq_dip, amp_dip = np.delete(freq_dip, noise_peaks_indices), np.delete(amp_dip, noise_peaks_indices)
        
        #estimate dip_width
        dip_depth_factor = 0.4
        window_length = int(0.5 / self.freq_resolution) #empirical value
        if (window_length % 2) == 0: window_length = window_length + 1 # 'window_length' has to be a positive odd integer according to 'savgol_filter' specification
        if window_length < 3: window_length = 3
        smooth_residuals = self.resonator_func_lmfit(resonator_fit_results, freq_dip_cut) - savgol_filter(amp_dip_cut, window_length, 1)
        nu_z_idx = np.argmax(smooth_residuals)
        estimated_dip = freq_dip_cut[nu_z_idx]
        estimated_dip_depth = smooth_residuals[nu_z_idx]
        nu_z_left_idx, nu_z_right_idx = np.where(smooth_residuals > estimated_dip_depth * dip_depth_factor)[0][0], np.where(smooth_residuals > estimated_dip_depth * dip_depth_factor)[0][-1] #two side points where estimated_dip_depth drops by a factor (dip_depth_factor - 1)
        estimated_dip_width = (nu_z_right_idx - nu_z_left_idx) * self.freq_resolution

        #check_points = np.array([nu_z_left_idx, nu_z_right_idx, nu_z_idx])
        #data = self.resonator_func_lmfit(resonator_fit_results, freq_dip_cut) - amp_dip_cut
        #plt.plot(freq_dip_cut, data, label = 'data')
        #plt.plot(freq_dip_cut, smooth_residuals, label = 'residuals')
        #plt.scatter(freq_dip_cut[check_points], smooth_residuals[check_points], color = 'black', zorder = 10)
        #plt.legend()
        #plt.show()
        #return

        #initiate fit parameters for the dip fit
        params = Parameters()
        init_value = resonator_fit_results

        #these parameters are always fitted
        params.add('nu_z', min = estimated_dip - estimated_dip_width, max = estimated_dip + estimated_dip_width, value = estimated_dip)
        params.add('dip_width', min = estimated_dip_width*0.2, max = estimated_dip_width*5, value = estimated_dip_width)

        #these are optional
        nu_z_jitter = 0.2
        if self.init_params['fixed nu_z_jitter'] is not None:
            nu_z_jitter = self.init_params['fixed nu_z_jitter']
            self.init_params['vary nu_z_jitter'] = False

        if self.init_params['vary nu_z_jitter'] or self.init_params['fixed nu_z_jitter'] is not None:
            params.add('nu_z_jitter', vary = self.init_params['vary nu_z_jitter'], min = 0.05, max = 2.0, value = nu_z_jitter)

        #these parameters can be fitted or taken from the resonator fit (specified in the config)
        slope = self.init_params['fixed slope'] if self.init_params['fixed slope'] is not None else init_value['slope']
        params.add('slope', vary = self.init_params['vary slope'], min = init_value['slope']*0.1, max = init_value['slope']*10, value = slope)
        params.add('A', vary = self.init_params['vary A'], min = init_value['A']*0.1, max = init_value['A']*10, value = init_value['A'])
        params.add('Q', vary = self.init_params['vary Q'], min = init_value['Q']*0.5, max = init_value['Q']*2, value = init_value['Q'])
        params.add('offset', vary = self.init_params['vary offset'], min = init_value['offset']*0.1, max = init_value['offset']*50, value = init_value['offset'])
        params.add('nu_res', vary = self.init_params['vary nu_res'], min = init_value['nu_res'] - 10, max = init_value['nu_res'] + 10, value = init_value['nu_res'])


        #fit the dip
        fit_dip = minimize(self.dip_func_lmfit, params, args=(freq_dip, amp_dip), method = self.init_params["fit_method"])

        #put values of the fitted parameters into the dictionary
        resonator_fit_results['nu_res_glob'] = resonator_fit_results['nu_res'] #save nu_res from the resonator fit (not just from the dip fit)
        resonator_fit_results['dnu_res_glob'] = resonator_fit_results['dnu_res'] #save dnu_res from the resonator fit (not just from the dip fit)
        dip_fit_results = resonator_fit_results.copy()
        for name, param in fit_dip.params.items(): 
            dip_fit_results[str(name)] = param.value
            if param.stderr != 0: dip_fit_results['d' + str(name)] = param.stderr #if parameter is varied then take its uncertainty, otherwise leave uncertainty from the resonator fit
            else: pass
        dip_fit_results['LO'] = self.init_params['LO']

        dip_fit_results["dip_shift"] = dip_fit_results["nu_z"] - dip_fit_results["nu_res_glob"]
        dip_fit_results["ddip_shift"] = np.sqrt(dip_fit_results["dnu_z"]**2 + dip_fit_results["dnu_res_glob"]**2)

        try:
            dip_fit_results['fit_success'] = fit_dip.success
            dip_fit_results['redchi'] = fit_dip.redchi
        except:
            pass
        

        if self.init_params['LO'] is not None: #basically if done in the IonWork_GUI (LO provided only in the GUI)
            print('\n \n _______________________ Fit _______________________')

        if self.init_params['print fit_report']:
            print(fit_report(fit_dip))

        if self.init_params['print fit_evaluation_time']:
            t = time.time() - fit_start_time
            print(f"--- Fit took {int(t/60)} minutes and {round(t - int(t/60)*60)} seconds ---")

        #plot and/or save the results if necessary
        if self.init_params['plot_save_dir'] is not None or self.init_params['plot results'] is True or self.init_params["return figure"]:
            nu_z = dip_fit_results['nu_z']
            if self.init_params['LO'] is not None: #basically if done in the IonWork_GUI (LO provided only in the GUI)
                fig = plt.figure(figsize = (13, 7))
            else:
                fig = plt.figure(figsize = (24, 8))
            grid = plt.GridSpec(3, 4, wspace = 0.02, hspace = 0.05)
            dip_plot, residuals_plot, dip_plot_zoom, residuals_plot_zoom, resonator_plot = plt.subplot(grid[:2, :2]), plt.subplot(grid[2, :2]), plt.subplot(grid[:2, 2]), plt.subplot(grid[2, 2]), plt.subplot(grid[0:, 3])
            dip_plot.get_shared_x_axes().join(dip_plot, residuals_plot)
            dip_plot.set_xticklabels([])

            plt.setp(dip_plot_zoom.get_yticklabels(), visible=False)
            plt.setp(dip_plot_zoom.get_xticklabels(), visible=False)
            dip_plot_zoom.axes.get_xaxis().set_visible(False)
            dip_plot.axes.get_xaxis().set_visible(False)
            plt.setp(residuals_plot_zoom.get_yticklabels(), visible=False)
            plt.setp(dip_plot.get_xticklabels(), visible=False)

            #recenter the zoomed plot relative to nu_z
            if self.init_params['dip_width'] is not None and self.init_params['dip_width'] < dip_fit_span:
                freq_dip_cut, amp_dip_cut = self.cut_spec(freq_dip, amp_dip, nu_z, self.init_params['dip_width'])
            
            dip_plot_zoom.plot(freq_dip_cut, amp_dip_cut, label = 'zoom fit', color = 'orange')
            dip_plot_zoom.plot(freq_dip_cut, self.dip_func_lmfit(dip_fit_results, freq_dip_cut), label = 'dip fit', color = 'red')
            dip_plot_zoom.set_ylim([min(min(self.dip_func_lmfit(dip_fit_results, freq_dip_cut)), min(amp_dip_cut)) - 2, max(amp_dip_cut) + 2])
            #dip_plot_zoom.scatter(freq_dip_cut[check_points], self.dip_func_lmfit(dip_fit_results, freq_dip_cut)[check_points], label = 'check points', color = 'black', zorder=10, s = 12)

            dip_plot.plot(freq_cut, amp_cut, label = 'spec data')
            dip_plot.scatter(noise_peaks[0], noise_peaks[1], label = 'noise peaks', color = 'blue')
            dip_plot.plot(freq_dip, amp_dip, label = 'dip data', color = 'orchid')
            dip_plot.plot(freq_dip_cut, amp_dip_cut, label = 'cutout for res fit', color = 'orange')
            dip_plot.plot(freq_cut, self.resonator_func_lmfit(resonator_fit_results, freq_cut), label = 'res fit', color = 'lime')
            dip_plot.plot(freq_cut, self.dip_func_lmfit(dip_fit_results, freq_cut), label = 'dip fit', color = 'red')
            dip_plot.set_ylim([min(min(self.dip_func_lmfit(dip_fit_results, freq_dip)), min(amp_dip)) - 2, max(amp_dip) + 2])
            #dip_plot.scatter(freq_dip_cut[check_points], self.dip_func_lmfit(dip_fit_results, freq_dip_cut)[check_points], label = 'check points', color = 'black', zorder=10, s = 12)
            dip_plot.legend()

            fit_residuals = amp_dip - self.dip_func_lmfit(dip_fit_results, freq_dip)
            residuals_plot.plot(freq_dip, fit_residuals, label = 'dip fit residuals')
            residuals_plot.set_xlim([nu_z - dip_fit_span*0.6, nu_z + dip_fit_span*0.6])
            residuals_plot.axhline(0, color='black')
            residuals_plot.legend()

            residuals_plot_zoom.plot(freq_dip, fit_residuals, label = 'dip fit residuals')
            residuals_plot_zoom.axhline(0, color='black')
            if self.init_params['dip_width'] is not None and self.init_params['dip_width'] < dip_fit_span:
                residuals_plot_zoom.set_xlim([nu_z - self.init_params['dip_width']*0.5, nu_z + self.init_params['dip_width']*0.5])
            else:
                residuals_plot_zoom.set_xlim([nu_z - dip_fit_span*0.5, nu_z + dip_fit_span*0.5])
            #residuals_plot_zoom.legend()

            resonator_plot.plot(freq_cut, amp_cut, label = 'res data')
            resonator_plot.scatter(noise_peaks[0], noise_peaks[1], label = 'noise peaks', color = 'blue')
            resonator_plot.plot(freq_dip, amp_dip, label = 'dip data', color = 'orchid')
            resonator_plot.plot(freq_cut, self.resonator_func_lmfit(resonator_fit_results, freq_cut), label = 'res fit', color = 'lime')
            resonator_plot.plot(freq_cut, self.dip_func_lmfit(dip_fit_results, freq_cut), label = 'dip fit', color = 'red') #, color = 'C3')
            resonator_plot.yaxis.tick_right()
            resonator_plot.legend()
            if self.init_params['plot_save_dir'] is not None:
                plt.savefig(self.init_params['plot_save_dir'] +'.png', bbox_inches='tight')
                plt.close()
            if self.init_params['plot results']:
                #residuals_mean, residuals_dmean = statistics.mean_and_stderror(fit_residuals)
                #formatted_residuals_mean, formatted_residuals_dmean = '{:.1e}'.format(residuals_mean), '{:.1e}'.format(residuals_dmean)
                #print(f'mean fit residuals = {formatted_residuals_mean} +/- {formatted_residuals_dmean}')
                plt.show()

            # else: #simple plot
            #     plt.plot(freq_cut, amp_cut, label = 'spec data')
            #     plt.plot(freq_dip, amp_dip, label = 'dip data', color = 'orange')
            #     plt.plot(freq_cut, self.resonator_func_lmfit(resonator_fit_results, freq_cut), label = 'resonator fit', color = 'green')
            #     plt.plot(freq_cut, self.dip_func_lmfit(dip_fit_results, freq_cut), label = 'dip fit', linewidth=1.5, color = 'red')
            #     #plt.scatter(freq_dip[check_points], amp_dip[check_points], label = 'check points', color = 'black', zorder=10, s=5)
            #     plt.xlim(nu_z - dip_fit_span*0.6, nu_z + dip_fit_span*0.6) 
            #     plt.ylim(min(self.dip_func_lmfit(dip_fit_results, freq_dip)) - 2, max(amp_dip) + 2) 
            #     plt.legend()
            #     if self.init_params['plot_save_dir'] is not None:
            #         plt.savefig(self.init_params['plot_save_dir'] +'.png', bbox_inches='tight')
            #     if self.init_params['plot results']:
            #         plt.show()

            #         #the same but just zoomed in
            #         plt.plot(freq_cut, amp_cut, label = 'resonator data')
            #         plt.plot(freq_dip, amp_dip, label = 'dip data', color = 'orange')
            #         plt.plot(freq_cut, self.resonator_func_lmfit(resonator_fit_results, freq_cut), label = 'resonator fit', color = 'green')
            #         plt.plot(freq_cut, self.dip_func_lmfit(dip_fit_results, freq_cut), label = 'dip fit', linewidth=1.5, color = 'red')
            #         #plt.scatter(freq_dip[check_points], amp_dip[check_points], label = 'check points', color = 'black', zorder=10, s=5)
            #         plt.xlim(nu_z - dip_fit_span*0.1, nu_z + dip_fit_span*0.1) 
            #         plt.ylim(min(self.dip_func_lmfit(dip_fit_results, freq_dip)) - 2, max(amp_dip) + 2) 
            #         plt.legend()
            #         plt.show()


        if self.init_params["return figure"]:
            dip_fit_results["figure"] = fig

        return dip_fit_results
    


    def fit_double_dip(self, freq, amp):
        dummy_keys = ['nu_l', 'dnu_l', 'nu_r', 'dnu_r' 'dip_width', 'ddip_width', 'redchi', 'nu_res', 'dnu_res', 'fit_success']
        dummy_values = [0]*9 + [False]
        dummy_fit_data = dict(zip(dummy_keys, dummy_values))

        fit_start_time = time.time()

        resonator_fit_results, dip_exists, estimated_dip, noise_peaks = self.fit_resonator(freq, amp, fit_double_dip = True)

        #cut spectrum down to the dips span
        dip_fit_span = self.init_params['dip_fit_span']
        freq_dips, amp_dips = self.cut_spec(freq, amp, self.init_params['nu_z'], dip_fit_span)
        

        if not dip_exists:
            print('!!! No double dip found !!!')
            fig = plt.figure(figsize = (13, 7))
            plt.plot(freq_dips, amp_dips, label = 'data (fit span)')
            plt.plot(freq_dips, self.resonator_func_lmfit(resonator_fit_results, freq_dips), label = 'fit', color = 'green')

            plt.legend()
            if self.init_params['plot_save_dir'] is not None:
                plt.savefig(self.init_params['plot_save_dir'] +'.png', bbox_inches='tight')
                plt.close()
            if self.init_params['plot results']: 
                plt.show()
            if self.init_params["return figure"]:
                return dummy_fit_data, fig
            else:
                return dummy_fit_data

        #cut spectrum down to the resonator span
        freq_cut, amp_cut = self.cut_spec(freq, amp, resonator_fit_results['nu_res'], self.init_params["spec_span"])

        nu_l, nu_r, nu_z_idx, nu_l_idx, nu_r_idx = estimated_dip
        window_length = int(3 / self.freq_resolution) #empirical value
        if (window_length % 2) == 0: window_length = window_length + 1 # 'window_length' has to be a positive odd integer according to 'savgol_filter' specification
        if window_length < 7: window_length = 7
        smooth_residuals = self.resonator_func_lmfit(resonator_fit_results, freq_dips) - savgol_filter(amp_dips, window_length, 5)
        estimated_dip_depth_left, estimated_dip_depth_right = smooth_residuals[nu_l_idx], smooth_residuals[nu_r_idx]

        #estimate dip_width
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
        if self.init_params['equal widths']:
            estimated_dip_widths = (estimated_dip_width_l + estimated_dip_width_r)/2
            params.add('dip_widths', min = estimated_dip_widths*0.1, max = estimated_dip_widths*10, value = estimated_dip_widths)
        else:
            params.add('dip_width_l', min = estimated_dip_width_l*0.1, max = estimated_dip_width_l*10, value = estimated_dip_width_l)
            params.add('dip_width_r', min = estimated_dip_width_r*0.1, max = estimated_dip_width_r*10, value = estimated_dip_width_r)

        #these parameters can be fitted or taken from the resonator fit (can be specified in the config)
        nu_z_jitter = self.init_params['fixed nu_z_jitter'] if self.init_params['fixed nu_z_jitter'] is not None else 0.2
        params.add('nu_z_jitter', vary = self.init_params['vary nu_z_jitter'], min = 0.05, max = 0.6, value = nu_z_jitter)

        slope = self.init_params['fixed slope'] if self.init_params['fixed slope'] is not None else init_value['slope']
        params.add('slope', vary = self.init_params['vary slope'], min = init_value['slope']*0.1, max = init_value['slope']*10, value = slope)

        params.add('A', vary = self.init_params['vary A'], min = init_value['A']*0.2, max = init_value['A']*5, value = init_value['A'])
        params.add('Q', vary = self.init_params['vary Q'], min = init_value['Q']*0.2, max = init_value['Q']*5, value = init_value['Q'])
        params.add('nu_res', vary = self.init_params['vary nu_res'], min = init_value['nu_res'] - 5, max = init_value['nu_res'] + 5, value = init_value['nu_res'])
        params.add('offset', vary = self.init_params['vary offset'], min = init_value['offset']*0.1, max = init_value['offset']*10, value = init_value['offset'])

        #fit the double dip
        fit_double_dip = minimize(self.double_dip_func_lmfit, params, args=(freq_dips, amp_dips), method = self.init_params["fit_method"])

        if self.init_params['print fit_report']:
            print(fit_report(fit_double_dip))

        #put values of the fitted parameters into the dictionary
        double_dip_fit_results = resonator_fit_results.copy()
        for name, param in fit_double_dip.params.items(): 
            double_dip_fit_results[str(name)] = param.value
            if param.stderr != 0: double_dip_fit_results['d' + str(name)] = param.stderr #if parameter is varied then take its uncertainty, otherwise leave uncertainty from the resonator fit
            else: pass
        double_dip_fit_results['fit_success'] = fit_double_dip.success
        double_dip_fit_results['redchi'] = fit_double_dip.redchi
        double_dip_fit_results['LO'] = self.init_params['LO']
        double_dip_fit_results['nu_z'] = self.init_params['nu_z']

        if self.init_params['LO'] is not None: #basically if done in the IonWork_GUI (LO provided only in the GUI)
            print('\n \n _______________________ Fit _______________________')

        if self.init_params['print fit_report']:
            print(fit_report(fit_double_dip))

        if self.init_params['print fit_evaluation_time']:
            t = time.time() - fit_start_time
            print(f'--- Fit took {int(t/60)} minutes and {round(t - int(t/60)*60)} seconds ---')

        #plot and/or save the results if necessary
        if self.init_params['plot_save_dir'] is not None or self.init_params['plot results'] is True:
            if self.init_params['LO'] is not None: #basically if done in the IonWork_GUI (LO provided only in the GUI)
                fig = plt.figure(figsize = (13, 7))
            else:
                fig = plt.figure(figsize = (24, 8))
            grid = plt.GridSpec(3, 3, wspace = 0.1, hspace = 0.05)
            dip_plot, residuals_plot, resonator_plot = plt.subplot(grid[:2, :2]), plt.subplot(grid[2, :2]), plt.subplot(grid[0:, 2])
            dip_plot.get_shared_x_axes().join(dip_plot, residuals_plot)
            dip_plot.set_xticklabels([])

            dip_plot.plot(freq_cut, amp_cut, label = 'resonator data')
            dip_plot.scatter(noise_peaks[0], noise_peaks[1], label = 'noise peaks', color = 'blue')
            dip_plot.plot(freq_dips, amp_dips, label = 'dips data', color = 'orchid')

            if self.init_params['double_dip_width'] is not None and self.init_params['double_dip_width'] < dip_fit_span:
                freq_dips_cut, amp_dips_cut = self.cut_spec(freq_dips, amp_dips, self.init_params['nu_z'], self.init_params['double_dip_width'])
                dip_plot.plot(freq_dips_cut, amp_dips_cut, label = 'cutout for res fit', color = 'orange')

            dip_plot.plot(freq_cut, self.resonator_func_lmfit(resonator_fit_results, freq_cut), label = 'resonator fit', color = 'lime')
            dip_plot.plot(freq_cut, self.double_dip_func_lmfit(double_dip_fit_results, freq_cut), label = 'dip fit', color = 'red')
            dip_plot.set_ylim([min(self.double_dip_func_lmfit(double_dip_fit_results, freq_dips)) - 2, max(amp_dips) + 2])
            dip_plot.scatter(freq_dips[check_points], amp_dips[check_points], label = 'check points', color = 'black', zorder=10, s = 12)
            dip_plot.legend()

            fit_residuals = amp_dips - self.double_dip_func_lmfit(double_dip_fit_results, freq_dips)
            residuals_plot.plot(freq_dips, fit_residuals, label = 'residuals')
            residuals_plot.set_xlim([self.init_params['nu_z'] - dip_fit_span*0.6, self.init_params['nu_z'] + dip_fit_span*0.6])
            residuals_plot.axhline(0, color='black')
            residuals_plot.legend()

            resonator_plot.plot(freq_cut, amp_cut, label = 'resonator data')
            resonator_plot.scatter(noise_peaks[0], noise_peaks[1], label = 'noise peaks', color = 'blue')
            resonator_plot.plot(freq_dips, amp_dips, label = 'dips data', color = 'orchid')
            resonator_plot.plot(freq_cut, self.resonator_func_lmfit(resonator_fit_results, freq_cut), label = 'resonator fit', color = 'lime')
            resonator_plot.plot(freq_cut, self.double_dip_func_lmfit(double_dip_fit_results, freq_cut), label = 'dip fit', color = 'red')
            resonator_plot.legend()
            if self.init_params['plot_save_dir'] is not None:
                plt.savefig(self.init_params['plot_save_dir'] +'.png', bbox_inches='tight')
                plt.close()
            if self.init_params['plot results']:
                #residuals_mean, residuals_dmean = statistics.mean_and_stderror(fit_residuals)
                #formatted_residuals_mean, formatted_residuals_dmean = '{:.1e}'.format(residuals_mean), '{:.1e}'.format(residuals_dmean)
                #print(f'mean fit residuals = {formatted_residuals_mean} +- {formatted_residuals_dmean}')
                plt.show()

            # else:
            #     plt.plot(freq_cut, amp_cut, label = 'resonator data')
            #     plt.plot(freq_dips, amp_dips, label = 'dip data', color = 'orange')
            #     plt.plot(freq_cut, self.resonator_func_lmfit(resonator_fit_results, freq_cut), label = 'resonator fit', color = 'green')
            #     plt.plot(freq_cut, self.double_dip_func_lmfit(double_dip_fit_results, freq_cut), label = 'dip fit', linewidth=1.5, color = 'red')
            #     plt.scatter(freq_dips[check_points], amp_dips[check_points], label = 'check points', color = 'black', zorder=10, s=5)
            #     plt.xlim(self.init_params['nu_z'] - dip_fit_span*0.6, self.init_params['nu_z'] + dip_fit_span*0.6) 
            #     plt.ylim(min(self.double_dip_func_lmfit(double_dip_fit_results, freq_dips)) - 2, max(amp_dips) + 2)
            #     plt.legend()
            #     if self.init_params['plot_save_dir'] is not None:
            #         plt.savefig(self.init_params['plot_save_dir'] +'.png', bbox_inches='tight')
            #         plt.close()
            #     if self.init_params['plot results']:
            #         plt.show()

        if self.init_params["return figure"]:
            double_dip_fit_results["figure"] = fig

        return double_dip_fit_results





        

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    '''
    init_user = {'nu_z': None, #504450.98 #501492.7, #8076.75 + 728000, #736074.2
                'dip_fit_span': 70, #Hz
                'spec_span': 3000, #Hz
                'LO': None,#475000,

                'vary A': True, 
                'vary offset': False, 
                'vary slope': False,
                'slope': None,
                'vary nu_res': True, 
                'vary Q': True,
                'vary nu_z_jitter': False,
                'fixed nu_z_jitter': None,
                'equal widths': True,

                'print fit_report': True,
                'plot results': True,
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

    init_user = {'nu_z': 26545.73+475000, #501546.034, #12054 + 724000, #8076.75 + 728000, #501492.7
                'fit_method': 'least_squares', #the 'least_squares' method always estimes the errors well, unlike the 'leastsq' method

                #'nu_z': None,
                'LO': None,
                'dip_fit_span': 90, #Hz
                'spec_span': 3000, #Hz
                'dip_width': 60, #Hz <- cutout span around nu_z for the resonator fit (if None then 'dip_width' = 'dip_fit_span')
                'double_dip_width': 60, #Hz <- cutout span around nu_z for the resonator fit (if None then 'double_dip_width' = 'dip_fit_span')
                'min dip_depth': 5, #dB

                'vary A': True,
                'vary Q': True,
                'vary nu_res': False, 
                'vary offset': False, 
                
                'vary slope': False,
                'fixed slope': None, #dB/Hz

                'vary nu_z_jitter': False,
                'fixed nu_z_jitter': None, #Hz

                'double dips': None, #[dip_left, dip_right] <- position of the dips
                'equal widths': True,

                'print fit_report': False,
                'plot results': True,

                'plot_save_dir': None,
                'return figure': False,
                'print fit_evaluation_time': False,
    }

    #path = '\\\\samba1\\PENTATRAP\\_MEMBERS_\\Pavel\\analysis\\CHAR_B1_Dip_Ne_trap2\\pnp_dip_unwrap\\cycle2\\shift_0.01\\dip_measurement\\cycle1\\position_1\\trap2\\cidx_2_2.spec'
    path = '\\\\samba1\\PENTATRAP\\_MEMBERS_\\Pavel\\analysis\\test_spectra\\trap3_cyc_0_0.08V.spec'
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
    print(fit_obj.fit_double_dip(data[0], data[1]))
    #print(fit_obj.fit_axial_dip(data[0], data[1]))


    
    

