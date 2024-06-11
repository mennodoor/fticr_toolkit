from cmath import phase
from socket import NI_NUMERICHOST
from tracemalloc import start
from weakref import ref
from cv2 import exp
import numpy as np
from scipy.optimize import curve_fit
import math
from numpy.lib.ufunclike import fix
#from scipy.optimize import minimize
import pandas as pd
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import plotly.express as px

from fticr_toolkit.statistics import mean_and_stderror
from fticr_toolkit import statistics
from fticr_toolkit import filtering

### FORMALLY KNOWN AS UNWRAPPING... THIS IS ABOUT THE N DETERMINATION BEFORE THE ACTUAL MEASUREMENT

def calc_N(nu, acc_time, phase):
    return acc_time * nu  - (phase/(2*np.pi))

def calc_nu(N, acc_time, phase):
    return ( N  + (phase/(2*np.pi)) ) / acc_time

def calc_dnu(acc_time, dphase):
    return (dphase/(2*np.pi)) / acc_time

def unwrap(array):
    """
    This is just here if you want to change to another method for unwrapping
    """
    return np.unwrap(array)

def drift_unwrap_method(array, pi_span = 1.5, times=None, fixit=True, slope=None):
    """
    This is a super cool drift unwrap for the case if there is an outliner and the normal
    numpy unwrap fails, this will fit the drift of the phase and does the unwrap additionally
    by calculating an expected value and unwrap using the delta to this value. 
    
    The linear fit to calculate the expected value starts if there are at least 3 (min_fit) values
    given, this is meant to protect against super wierd data... please check if everything looks normal.

    This feature should not be used light-minded, since it may removes features in 
    your data which would be a hint on actuall bad measurement points!!!

    TODO: this feature works with indexes where it would be better to work with actual times. That
    means if we ever decide to have non-equal time-differences between phase measurements (like LIONTRAP)
    this feature has to be fixed :)

    """

    if times is not None:
        try:
            xarray = times.tolist()
        except:
            xarray = times
    else:
        xarray = list(range(len(array)+1))

    for idx in range(len(array)):
        if idx == 0:
            continue

        x = xarray[:idx]
        dx = xarray[idx] - xarray[idx-1]

        val_here = array[idx]
        val_before = array[idx-1]
        delta = val_here - val_before
        #print(delta)

        # simple unwrap:
        if idx == 1:
            if slope is not None:
                expected_value = val_before + dx*slope
                delta = val_here - expected_value
                print('fixed slope unwrap, before/here/expected:', val_before, val_here, expected_value)

            while abs(delta) > np.pi:
                #print("idx1, before, here", val_before, val_here)
                if delta > np.pi:
                    array[idx:] -= 2*np.pi
                elif delta < -np.pi:
                    array[idx:] += 2*np.pi
                if abs(delta)>np.pi:
                    print("simple unwrap! idx", idx, "value before", val_before, "value here", val_here, "new", array[idx])
                
                val_here = array[idx]
                if slope:
                    expected_value = val_before + dx*slope
                    delta = val_here - expected_value
                else:
                    delta = val_here - val_before

            # DONT DO THE REST!
            continue
        
        y = array[:idx]
        try:
            y = [item for sublist in y for item in sublist]
        except TypeError:
            pass

        order = 1
        if idx == 2:
            val_before_before = array[idx-2]
            dx_before = xarray[idx-1] - xarray[idx-2]
            dydx = (val_before - val_before_before) / dx_before
            expected_value = val_before + dydx*dx
            if slope is not None:
                expected_value = val_before + dx*slope
                print('fixed slope unwrap, before/here/expected:', val_before, val_here, expected_value)
            #print("idx2, before before, before, here, expected", val_before_before, val_before, val_here, expected_value)
        else:
            #print("idx", idx, end="")
            #print(x, y)
            if idx > len(array)/2:
                order = 3
            coef = np.polyfit(x,y,order)
            #print(coef)
            #if abs(coef[0]) > max_slope: # NOTE: This did not behave great, just as a hint.
            #    coef[0] = math.copysign(max_slope, coef[0])
            #print(coef)
            poly1d_fn = np.poly1d(coef) 
            expected_value = poly1d_fn(xarray[idx])
            #print("idx", idx, ", before, here, expected", val_before, val_here, expected_value)
            if slope is not None and idx<6:
                expected_value2 = val_before + dx*slope
                print('fixed slope unwrap, before/here/expected:', val_before, val_here, expected_value)
            if False: # for debugging
                plt.plot(x, y)
                plt.plot(x, poly1d_fn(x))
                plt.show()

        # you have to get the val_here again, since the normal unwrap may already did the trick.
        val_here = float(array[idx])
        delta = val_here - expected_value
        if delta > (np.pi*pi_span):
            array[idx:] -= 2*np.pi
        elif delta < (-np.pi*pi_span):
            array[idx:] += 2*np.pi
        if abs(delta)>(np.pi*pi_span): 
            print("drift unwrap! x", xarray[idx], "value before", val_before, "expected", expected_value, "current", val_here, "new", array[idx])

        if fixit:
            val_here = float(array[idx])
            delta = val_here - expected_value
            dup = abs(delta + 2*np.pi)
            ddown = abs(delta - 2*np.pi)
            minddd = min([abs(delta), dup, ddown])
            if minddd == ddown:
                array[idx:] -= 2*np.pi
                print("fixit unwrap! x", xarray[idx], "value before", val_before, "expected", expected_value, "current", val_here, "new", array[idx])
            elif minddd == dup:
                array[idx:] += 2*np.pi
                print("fixit unwrap! x", xarray[idx], "value before", val_before, "expected", expected_value, "current", val_here, "new", array[idx])

    return array

def leastsquare_unwrap(df, column='phase', pi_span=1.0, timesort='time', median_group=['cycle'], correlate=False, fit_order=3, show=False, iterations=1000):
    # time sort and add epoch time
    df.sort_values(by=timesort, inplace=True)
    df["epoch"] = df['time'].astype('int64')//1e9

    # show original data:
    if show:
        plt.plot(df.epoch, df[column])
        plt.show()

    if correlate:
        # we want to use both position data sets to unwrap, following steps need to be done:
        # 1) normalize the starting phases to 0, the median is actually enough.
        # 2) use both trap phases (and there times) to make the proper unwrap
        median_group.append("position")
        phase_offsets = {}

    # new df for new data and keep list of median phases for unwrap:
    results = pd.DataFrame()
    times = []
    medians = []
    indxes = []
    start_time = None

    # loop over cycles for mean and substraction of reference as well as global unwrap
    groupby = df.groupby(median_group)
    for gname, grp in groupby:

        phases = grp[column].to_numpy()
        idx = grp.index.to_numpy()

        # get median from phases
        non_masked_phases = grp[grp["masked"]==False][column]
        median = np.median(non_masked_phases)

        # 0 starting phase and adjust all others
        if correlate:
            corr = gname[1]
            if corr not in phase_offsets.keys():
                phase_offsets[corr] = median
            median -= phase_offsets[corr]

        time = grp.epoch.mean()
        if start_time is None:
            start_time = time

        medians.append(median)
        times.append(time)
        indxes.append(idx)
        #print(idx, time, median)

    medians = np.asarray(medians)
    times = np.asarray(times)
    indxes = np.asarray(indxes)
    results = df.copy(deep=True)
    history = []

    for i in range(iterations):
        _, resi, _, _, _ = np.polyfit(times, medians, fit_order, rcond=None, full=True, w=None, cov=False)
        ls_start = np.sum(resi**2)
        pick = np.random.choice(len(times))
        up_down = np.random.choice([-1, 1])
        medians[pick] += up_down*2*np.pi*pi_span
        _, resi, _, _, _ = np.polyfit(times, medians, fit_order, rcond=None, full=True, w=None, cov=False)
        ls_new = np.sum(resi**2)

        if ls_new < ls_start:
            results.loc[indxes[pick], column] += up_down*2*np.pi*pi_span
            history.append(ls_new)
        else:
            medians[pick] -= up_down*2*np.pi*pi_span
            history.append(ls_start)

        #if len(history) > 50:
        #    recent = history[-50:-1]
        #    rtol = np.std(recent)/np.mean(recent)
        #    print(recent)
        #    if rtol < 1e-7:
        #        break 
    return results



def grouped_unwrap(df, column='phase', start_phase_time=None, reverse=False, pi_span=1.0, skip=0,
                 timesort='time', median_group=['cycle'], correlate=False, fit_N=99, fit_order=3, show=False, first_expected_offset=0):

    # df is already a data subset of a mcycle, trap, position kind with only long phases (one acc_time)

    # time sort and add epoch time
    df.sort_values(by=timesort, inplace=True)
    df["epoch"] = df['time'].astype('int64')//1e9

    # unwrap of subsets, median_group is typically "cycle", so each cycle is unwrapped
    #df_subunwrapped = unwrap_subsets(df, median_group, column, False, 1.0, True, None, False)

    # show original data:
    if show:
        plt.plot(df.epoch, df[column])
        plt.show()

    if correlate:
        # NOTE: WORK IN PROGRESS!!! NEVER USE WITH START_PHASE!!!
        if start_phase_time is not None:
            print("sorry, there is no support for correlate feature AND start_phase_time")
            raise(TypeError)
        # we want to use both position data sets to unwrap, following steps need to be done:
        # 1) normalize the starting phases to 0, the median is actually enough.
        # 2) use both trap phases (and there times) to make the proper unwrap
        median_group.append("position")
        phase_offsets = {}

    # new df for new data and keep list of median phases for unwrap:
    results = pd.DataFrame()
    times = []
    medians = []
    start_time = None
    # if there is a start phase and time, add it to the median history
    if start_phase_time is not None:
        medians.append(start_phase_time[0])
        start_time = int(start_phase_time[1].astype('int64')//1e9)
        times.append(start_time)

    # loop over cycles for mean and substraction of reference as well as global unwrap
    groupby = df.groupby(median_group)
    if reverse:
        groupby = reversed(tuple(groupby))

    for gname, grp in groupby:
        #print(gname)#, times, medians)
        
        # unwrap phases # NOTE: DONT YOU DARE! this was carefully fixed before and should not be tried with this "simpelton" unwrap!
        #phases = unwrap(grp[column].to_numpy())
        phases = grp[column].to_numpy()

        # get median from phases
        non_masked_phases = grp[grp["masked"]==False][column]
        median = np.median(non_masked_phases)

        # 0 starting phase and adjust all others
        if correlate:
            corr = gname[1]
            if corr not in phase_offsets.keys():
                phase_offsets[corr] = median
            median -= phase_offsets[corr]

        time = grp.epoch.mean()
        if start_time is None:
            start_time = time

        N_history = len(medians)

        if N_history == 0 or N_history < skip:
            # first value, just store it and keep the phases
            medians.append(median)
            times.append(time)
            print(N_history, time, median)

        else:
            # if negative fit_N we want to just take the first fit_N
            # values as they are and then start fitting and calculating 
            # the expectance value. Also it will be a running fitting window
            # of only the latest fit_N values. NOTE: STUPID IDEA!
            #if fit_N < 0 and N_history < abs(fit_N):
            #    expected_value = median

            # not yet fitting, to low number of values, just take the last
            # phase median as an expectation value
            if N_history < abs(fit_N):
                expected_value = medians[-1] + first_expected_offset

            # fitting saved data and estimate the next phase median by
            # extrapolation
            else:
                if fit_N < 0:
                    x = np.asarray(times)[fit_N:] - start_time
                    y = medians[fit_N:]
                else:
                    x = np.asarray(times) - start_time
                    y = medians
                    
                order = np.clip(int(N_history/3), 1, fit_order)

                """
                expos = np.asarray(range(0, order+1))[::-1]
                slope_bounds = [-5e-3, 5e-3]
                def poly(x, *p):
                    derivative_p = p[:-1]*expos[:-1]
                    mean_slope = np.mean(np.polyval(derivative_p, x))
                    if mean_slope < slope_bounds[0] or mean_slope > slope_bounds[1]:
                        return np.polyval(p, x)*mean_slope/slope_bounds[0]*10
                    return np.polyval(p, x)
                p0 = np.asarray([0]*(order+1)) # highest to lowest coefficient
                bounds = (
                    [-np.inf]*(order+1), # lower
                    [np.inf]*(order+1), # upper
                )
                bounds[0][-1] = -5e-3
                bounds[1][-1] = 5e-3
                #print(bounds[0])
                #print(bounds[1])
                coef, pcov = curve_fit(poly, x, y, p0, bounds=bounds)
                print("fit results", coef)
                """

                coef = np.polyfit(x, y, order)

                poly1d_fn = np.poly1d(coef)
                # get expected aat current phase time
                expected_value = poly1d_fn(time - start_time)
            
            # if negative fit_N, we want to 
            # unwrap phases relative to the expected value:
            delta = median - expected_value
            # do it multiple times, since we do not effect the whole dataset afterwards
            # meaning that after an unwrap jump happend the next jump is probably 4 pi 
            # and not only 2
            while abs(delta) > np.pi*pi_span:
                #print("unwrap! group, expected, median, delta", gname, expected_value, median, delta)
                if delta > np.pi:
                    phases -= 2*np.pi
                elif delta < -np.pi:
                    phases += 2*np.pi

                # recalc median and delta
                median = np.median(phases)
                delta = median - expected_value

            # store the median and time
            times.append(time)
            medians.append(median)

        if show:
            plt.plot(grp.epoch, phases, ".", label="phases")
            plt.plot(times, medians, "o", label="medians")
            if len(medians) > abs(fit_N):
                if fit_N < 0:
                    x = np.asarray(times)[fit_N:] - start_time
                else:
                    x = np.asarray(times) - start_time
                plt.plot(x + start_time, poly1d_fn(x), label="fit N/order "+str(N_history)+" "+str(order))
            plt.legend()
            plt.grid()
            plt.show()

        # store the unwrapped phases in results df
        grp[column] = phases
        results = results.append(grp)

    return results


def unwrap_dset(dset, column="phase", start_phase = None, reverse = False, drift_unwrap = False, drift_pi_span=1.5, 
                timesort = False, start_phase_time = None, slope=None, median_group=None, show=False):
    """
    Unwraps the phase of the given column in a full pandas DataFrame.

    A start phase can be given as well to use it as a handle for the first phase.

    Before you use drift_unwrap, please read the notes in the docstring of the 
    drift_unwrap_method() ! ! !
    """

    #print(dset.columns)
    if "masked" in dset.columns:
        masked_data = dset[dset["masked"] == True]
        #print(masked_data)
        dset = dset[dset["masked"] == False]
    
    # just to be save... sort by timestamp so the unwrap is correct.
    if timesort:
        if reverse:
            # this is used if you want to unwrap using the N and last phase of the post_unwrap:
            # we take the last phase of the post unwrap as a starting phase for the reversed
            # array of the phase data. After unwrapping, we remove this phase again and
            # flip the array back to the original order. The phases will now start with an high
            # absolute value, but it does not matter. Just remember to take the N also from the
            # post-unwrap for the freqency determination!!
            dset = dset.sort_values(by='time', ascending = False )
            if slope is not None:
                slope *= -1
        else:
            dset = dset.sort_values(by='time')

    # get the phase data as a single flat array
    phases = dset[column].to_numpy()

    # add the startphase to the beginning of the array
    if start_phase is not None:
        phases = np.insert(phases, 0, start_phase)

    # use or not use drift_unwrap
    if drift_unwrap:
        if isinstance(drift_unwrap, str) and drift_unwrap == "timex" :
            times = dset['time'].astype('int64')//1e9
            times = times.to_numpy()

            if start_phase is not None:
                start_phase_time= start_phase_time.astype('int64')//1e9
                times = np.insert(times, 0, start_phase_time )

            # normalize time to float array starting at 0
            start_time = times[0]
            times -= start_time
            
            # basically if reversed... because the first would be the latest (biggest) time.
            if times.mean() < 0:
                times *= -1
            #print(times)
            unwrapped_phases = drift_unwrap_method(phases, pi_span=drift_pi_span, times=times, fixit=False, slope=slope)
        elif isinstance(drift_unwrap, str) and drift_unwrap == "justfixit" :
            unwrapped_phases = drift_unwrap_method(phases, pi_span=drift_pi_span, fixit=True)
        else:
            unwrapped_phases = drift_unwrap_method(phases, pi_span=drift_pi_span, fixit=False)


    else:
        unwrapped_phases = unwrap(phases)
    # remove the starting phase again
    if start_phase is not None:
        unwrapped_phases = unwrapped_phases[1:]
        if start_phase_time is not None:
            times = times[1:]

    # update the original dataset
    dset[column] = unwrapped_phases

    # reinsert masked data:
    if "masked" in dset.columns:
        dset = dset.append(masked_data)

    # resort to normal time
    if timesort:
        dset = dset.sort_values(by='time')

    if show:
        #plt.clf()
        if isinstance(drift_unwrap, str) and drift_unwrap == "timex":
            plt.plot(times, unwrapped_phases)
        else:
            plt.plot(unwrapped_phases)
        plt.show()

    return dset


def unwrap_subsets(dset, groupby=["cycle"], column="phase", drift_unwrap = False, drift_pi_span=1.0, 
                timesort = False, slope=None, residuals_unwrap=False, mean_positive=True, show=False):
    """
    Here we will loop over subsets like cycle and unwrap a column (sorted by time) e.g. phase
    """
    acc_time_groups = dset.groupby(groupby)
    new_data = pd.DataFrame()
    for gname, group in acc_time_groups:

        group = unwrap_dset(group, column=column, drift_unwrap=drift_unwrap, drift_pi_span=drift_pi_span, timesort=timesort, slope=slope, show=show)

        if residuals_unwrap:

            while True:
                data_arr = group[column].to_numpy()
                indexes = group.index.to_numpy()
                mean_val = np.median(data_arr)
                residuals = data_arr - mean_val

                if all( residuals < np.pi*drift_pi_span) and all( -np.pi*drift_pi_span < residuals):
                    break

                print(mean_val, residuals)

                for i, res in enumerate(residuals):
                    val_before = group.at[indexes[i], column]
                    if res > np.pi:
                        group.at[indexes[i], column] = group.at[indexes[i], column] - 2*np.pi
                        print("residuals unwrap: idx, mean, resi, val_before, val_new ", indexes[i], mean_val, res, val_before, group.at[indexes[i], column])
                    elif res < -np.pi:
                        group.at[indexes[i], column] = group.at[indexes[i], column] + 2*np.pi
                        print("residuals unwrap: idx, mean, resi, val_before, val_new ", indexes[i], mean_val, res, val_before, group.at[indexes[i], column])

        if mean_positive:

            while True:
                data_arr = group[column].to_numpy()
                indexes = group.index
                mean_val = np.median(data_arr)
                if mean_val < 0:
                    group.loc[indexes, column] = group.loc[indexes, column] + 2*np.pi
                elif mean_val > 2*np.pi:
                    group.loc[indexes, column] = group.loc[indexes, column] - 2*np.pi
                else:
                    break

        new_data = new_data.append(group)


    return new_data


def substract_ref_phase(dset, identifier="acc_time", y="phase", groupby="cycle", mean=False, show=True):
    """
    This will substract the reference phase values from the long accumulation time phase
    values. 

    The dset must include the following coloumns: acc_time, phase, cycle 

    There have to be at least 2 diffrerent accumulation times, and all different accumulation
    times have to have the same number of phase data (same number of rows).
    """

    # split first if there are still reference and measurment phase in the dataset
    # NOTE: The reset index is needed for the substraction, since the columns will only
    # substract in same panda dataset indexes. But we store the original index so we can
    # re-assign it.
    subset_ref = dset[dset[identifier] == dset[identifier].min()].reset_index(drop=True)
    subset_meas = dset[dset[identifier] != dset[identifier].min()].reset_index(drop=True)
    #display(subset_meas)

    for gname, grp in subset_meas.groupby(groupby):
        subsub_ref = subset_ref[subset_ref[groupby] == gname]

        # we assign only the error of the mean of the reference phases as an error to the long
        # phases in this step. Getting the real error of the long phases is done later using the
        # whole main cycle, undrifted and then taking the max of inner/outer error.
        if mean:
            subsub_ref = subsub_ref[subsub_ref["masked"] == False] # TODO: this is actually a problem: if the ref phases are not averaged, a masked ref phase reduces the length of that array and substraction does not match
            ref_phases = subsub_ref[y].to_numpy()
            meanval, err = statistics.mean_and_error(ref_phases)
            grp[y] -= meanval
            grp["d"+y] = err
        else:
            try:
                ref_phases = subsub_ref[y].to_numpy()
                grp[y] = grp[y].to_numpy() - ref_phases
                grp["masked"] = grp["masked"] | subsub_ref["masked"]
            except:
                print("Error in ref-phase substraction: Does mean ref phase work?")
                grp[y] = grp[y].to_numpy() - np.median(ref_phases)
                print("YES!")

        # remove the reference acc_time! do not forget that!!!
        grp[identifier] -= subset_ref[identifier]
        # put it back to where you got it from
        subset_meas.iloc[grp.index] = grp
    """
    try:
        subset_meas[y] -= float(subset_ref[y])
        subset_meas[identifier] -= float(subset_ref[identifier])
    except Exception as e:
        if mean:
            print("mean ref phase", subset_ref[y].mean())
            subset_meas[y] -= subset_ref[y].mean()
        else:
            subset_meas[y] -= subset_ref[y]
        
        subset_meas[identifier] -= subset_ref[identifier]
    
    subset_meas.set_index( index )
    """
    if show:
        plt.plot(subset_meas["time"], subset_meas[y])
        plt.grid()
        plt.show()

    return subset_meas

def prepare_unwrap_phases(dset, val="phase", show=True, std=False):

    # loop over different acc_times to unwrap and average the phases
    averaged_data = pd.DataFrame(columns = ["acc_time", "phase", "dphase", "time", 'n'])
    for acc_time, group in dset.groupby(['acc_time']):

        group = unwrap_dset(group, column=val)
        group["masked"] = False
        group["index"] = list(range(len(group)))
        group = filtering.three_sigma(group, val, around="median", max_std=0.5, show=False)
        masked_ones =  group[ group["masked"]==True ]
        if not masked_ones.empty:
            print('filtered', len(masked_ones), "in acc_time", acc_time)
        group = group[ group["masked"]==False ]
        #display(group)
        if show:
            plt.plot(group["index"].to_numpy(), group[val].to_numpy(), label=acc_time)
        n = len(group)
        if n < 2:
            dval = np.pi*100/180
        else:
            dval = None
        mphase, dphase = statistics.mean_and_error(group[val].to_numpy(), dval)
        if std:
            dphase = np.std(group[val].to_numpy())

        # time 
        mean_epoch = np.mean((group["time"].astype('int64')//1e9).to_numpy())
        mean_dt = pd.to_datetime(mean_epoch, unit='s')
        # put it together
        new_row = pd.Series([acc_time, mphase, dphase, mean_dt, n], index=averaged_data.columns )
        averaged_data = averaged_data.append(new_row, ignore_index=True)

    #display(averaged_data)
    if show:
        plt.legend()
        plt.title("unwrapped filtered averaged phases")
        plt.show()

    return averaged_data



def determine_N(dset, nu_guess, resolution=0.001, negative=False, nu_range=2, show=True, val="phase", **kwargs):
    """
    This method will calculate the number N of full revolutions for the given evolution time (this should be t_acc - t_ref)

    the dset has to be pre-filtered, meaning: only containing data for one trap and one position (so one specific ion in a specific trap)

    the nu_p_guess should be just a float value of the reduced cyclotron frequency which was used n the measurement config

    the dset has to have following columns: acc_time, phase, time (a timestamp)

    procedure:
    - unwrap phases with same accumulation time (you could have by chance a "border phase", meaning that its flipping between >0 and <360, thats why you need this.)
    - remove the reference phase (smallest acc time)
    - average

    """
    pd.options.mode.chained_assignment = None  # default='warn'

    if show:
        print("\n °°° N determination °°° ")

    phase_offset = kwargs.get("phase_offset", 0)

    if negative:
    #    nu_guess *= -1
        dset["phase"] *= -1
        #phase_offset *= -1

    averaged_data = dset

    drop = kwargs.get("drop_accs", None)
    if drop is not None:
        print("DROPPING ACC ROWS", drop)
        averaged_data.drop(axis='index', index=drop, inplace=True)
    if show:
        display(averaged_data)


    # split off the reference phase from the rest of the data
    reference = averaged_data[averaged_data["acc_time"] == averaged_data["acc_time"].min()]
    #print(reference)
    phase_ref = float(reference["phase"])
    try:
        dphase_ref = float(reference["dphase"])
    except:
        pass
    acc_time_ref = float(reference["acc_time"])
    averaged_data = averaged_data[averaged_data["acc_time"] != acc_time_ref].reset_index(drop=True)

    averaged_data["acc_time"] -= acc_time_ref 
    averaged_data["phase"] -= phase_ref + phase_offset
    try:
        averaged_data["dphase"] = np.sqrt( ( (averaged_data["dphase"])**2 + dphase_ref**2 ).to_numpy() )
    except:
        pass

    #averaged_data["acc_time"] = np.around(averaged_data.acc_time.to_numpy(), 3)
    digits = kwargs.get("acc_time_digits", 3)
    averaged_data["acc_time"] = np.around(averaged_data.acc_time.to_numpy(), digits)
    #print(averaged_data.acc_time.to_numpy() - acc_rounded)

    #display(averaged_data)
    fitphases = averaged_data["phase"]
    fittimes = averaged_data["acc_time"]

    # this calculates the correctness of the nu guess (the error sum of all N calculations)
    def N_error_sum(nu):
        #n_sum = 0
        #for index, row in averaged_data.iterrows():
        #    N = calc_N(nu, row["acc_time"], row["phase"])
        #    N_int = np.around(N, decimals=0)
        #    n_sum += (N - N_int)**2
        N = calc_N(nu, fittimes, fitphases)
        N_int = np.around(N, decimals=0)
        n_sum2 = np.sum((N - N_int)**2)
        #if np.around(n_sum2, decimals=13) != np.around(n_sum, decimals=13):
        #    print(n_sum, n_sum2)
        #    raise ValueError
        return n_sum2

    def N_error_sum2(nu):
        matrix = np.zeros((len(nu), len(averaged_data)))
        for index, row in averaged_data.iterrows():
            Ns = calc_N(nu, row["acc_time"], row["phase"])
            Nints = np.around(Ns, decimals=0)
            #Nints = np.floor(Ns) # Nope!
            errors = Ns-Nints
            matrix[:,index] = errors**2
        errors = np.sum(matrix, axis=1)
        return errors

    # NOTE: !!! minimizing algorithms do not work so well due to multiple local minima
    # rather use a kind of brute force way: calculate the n_error in a 2 Hz range of the
    # guessed frequency and get the minimum.
    if resolution is None:
        resolution = round(1/averaged_data["acc_time"].max()/25, 6) # 25 samples per period.
        if show:
            print("choose resolution automatically:", resolution)

    nu_arange = np.arange(nu_guess-(nu_range/2), nu_guess+(nu_range/2), resolution)
    #error_values = []
    #for nu_test in nu_arange:
    #    error_values.append(N_error_sum(nu_test))
    #error_values = np.asarray(error_values)
    error_values = N_error_sum2(nu_arange)

    # get the correct frequency (with minimal N error)
    nu = nu_arange[np.argmin(error_values)]
    if show: print(nu)
    
    if error_values.min() > 0.2:
        print("\n !!! W A R N I N G !!! \nA minimum was found, but the sum of N-error is too big, >0.2. The Unwrap did in fact fail!!!")
    
    #if negative:
    #    averaged_data["phase"] *= -1

    # get the corresponding N and the phase value, this is everything whats returned
    acc_time_max = averaged_data["acc_time"].max()
    phase_for_N = averaged_data[averaged_data["acc_time"] == acc_time_max]
    phase_for_N = float(phase_for_N["phase"])

    mean_time = averaged_data[averaged_data["acc_time"] == acc_time_max].time.iloc[0].value
    N = round(calc_N(nu, acc_time_max, phase_for_N))

    # NOTE: new too, fix low phase values: NOTE: removed again, not neccessary to do
    #if phase_for_N < -2*np.pi:
    #    print("fix low last phase")
    #    N -= 1
    #    phase_for_N += 2*np.pi

    Nvalues = calc_N(nu, averaged_data["acc_time"][:], averaged_data["phase"][:])

    averaged_data["N"] = Nvalues
    averaged_data["N_intN"] = Nvalues - np.asarray(np.round(Nvalues), dtype=np.int64)
    Nquality = np.sqrt(np.mean(averaged_data["N_intN"].to_numpy()**2))

    # some visuallisation for ease of mind ;)
    if show:
        plt.axvline(nu, color='r')
        plt.plot(nu_arange, error_values)
        #plt.plot(nu_arange, error_values2)
        plt.title("Sum of N error vs nu")
        plt.show()
        display(averaged_data)

        plt.plot(averaged_data["acc_time"], averaged_data["N_intN"])
        plt.title("phase offset")
        plt.show()

    #if negative:
    #    nu *= -1
    #    nu_guess *= -1

    if show:
        print("")
        print(" °°° N =", float(N), " last phase =", phase_for_N, " nu =", nu, " °°° " )
        print(" °°° guess =", nu_guess, " diff =", nu_guess-nu, " °°° " )
        print(" °°° root mean square N-int(N) (2nd order banana) =", Nquality, " °°° " )

    if kwargs.get("residuals", False):
        nu_test = calc_nu(N, acc_time_max, phase_for_N)
        #print(nu_test)
        #nu_test = nu
        Nphase = nu_test * averaged_data["acc_time"].to_numpy()
        resi = Nphase - np.around(Nphase, 0) - averaged_data["phase"].to_numpy()/2/np.pi
        return N, phase_for_N, nu, mean_time, acc_time_max, Nquality, resi*2*np.pi

    return N, phase_for_N, nu, mean_time, acc_time_max, Nquality
