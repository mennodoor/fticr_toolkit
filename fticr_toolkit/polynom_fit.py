from pprint import pprint
import random
from datetime import datetime, timedelta

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev, cheb2poly

import scipy as sp
from scipy.optimize import curve_fit
from scipy.optimize import minimize

import pandas as pd

from matplotlib import pyplot as plt
import plotly.graph_objs as go

from fticr_toolkit import statistics

#------------------------------------------------------------------------------------------------------------------------------
# AUTO GROUPING
#------------------------------------------------------------------------------------------------------------------------------

def auto_group(list, n):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(list), n):
        yield list[i:i + n]

def auto_group_subset(dset, n_min, sub_min, sortby=["cycle", "time", "trap"], groupby="position", group_key="group", indexer="cycle"):
    """
    This function will take a subset of result data, sort it and loop over it to assign group indexes. The sorting and
    the n minimum and sub minimum are basically defining what kind of groups you are going to build with it. n_min is
    the minimum size for one group, if this value is reached and the other group is at least sub_min in size, the group
    index will advance.
    """
    # keep manual:
    if n_min == -1:
        return dset

    # if the n_min value is 0, we want all subcycles of the main cycle to be in one group, group 1
    if n_min == 0:
        dset[group_key] = 1
        print("all in one group...")
        return dset

    # This warning is very annoying... I think everything should be fine here. I think it raises in the sortby function...
    pd.options.mode.chained_assignment = None  # default='warn'

    # Sorting the data, in the default way, we sort by cycle, then time and then trap. This way we should have the "natural"
    # sequence for the data, the same as in the measurement. NOTE: sorting by position instead of time is not so good, since
    # the measurement could have started with position_2 instead of position_1 and would then be wrongly sorted.
    dset.sort_values(by=sortby, inplace=True)
    dset.reset_index(drop=True, inplace=True)
    #display(dset.index.to_numpy())

    # add the column for the group index and prepare some indexes and numbers
    dset[group_key] = np.NaN
    group_number = 1
    positions = dset[groupby].unique()
    counter = {}
    for pos in positions:
        counter[pos] = 0
    real_row_counter = 0

    # since we want to have ideally equal sized groups, we have to switch group index when either both reach the n_min size
    # or we do it a little more carefull: only one group needs n_min, but we will add at least one more of the other group.
    # This way we end up with 5/5 normally but if there are data points masked its still possible to get 5/3 and countinue 
    # with the next group.
    just_one_more_thing = True

    last_position = None
    groups = dset.groupby([indexer, groupby])
    dset_length = len(groups)
    #print(dset_length)

    for grpname, grp in groups:
        #display(grp.position.to_numpy())

        real_row_counter += 1

        # set the group indext to the original dataFrame
        dset.loc[grp.index, group_key] = int(group_number)

        # count only if the data is not masked
        if not all(grp["masked"].to_numpy()):
            counter[grpname[1]] += 1

        # check if minimal group numbers are reached and if the rest of the dset allows for a full group. If the rest is to less, we make the current
        # group bigger then normal.
        counter_list = list(counter.values())
        if (max(counter_list) >= n_min) and (min(counter_list) >= sub_min) and ( (dset_length-real_row_counter) >= n_min*len(positions) ):
            # add just one more from the next row (if ts not masked) to level out countings
            if just_one_more_thing:
                just_one_more_thing = False
                continue
            
            # next group
            group_number += 1
            # reset group intern counter
            for key, item in counter.items():
                counter[key] = 0

            # reset the "one more tag"
            just_one_more_thing = True
    
    return dset

""" NO COUNTER CYCLE
    last_position = None
    for i, row in dset.iterrows():
        real_row_counter += 1

        # set the group indext to the original dataFrame
        dset.at[i, group_key] = int(group_number)

        # count only if the data is not masked
        if not row["masked"]:
            if i == 0:
                counter[row[groupby]] += 1
            else:
                if last_position != row[groupby]: # real new position (e.g. for non average data, this is really needed...)
                    counter[row[groupby]] += 1
        last_position = row[groupby]
        
        # check if minimal group numbers are reached and if the rest of the dset allows for a full group. If the rest is to less, we make the current
        # group bigger then normal.
        counter_list = list(counter.values())
        if (max(counter_list) >= n_min) and (min(counter_list) >= sub_min) and ( (dset_length-real_row_counter) >= n_min*len(positions) ):
            # add just one more from the next row (if ts not masked) to level out countings
            if just_one_more_thing:
                just_one_more_thing = False
                continue
            
            # next group
            group_number += 1
            # reset group intern counter
            for key, item in counter.items():
                counter[key] = 0

            # reset the "one more tag"
            just_one_more_thing = True
    
    return dset
"""
#------------------------------------------------------------------------------------------------------------------------------
# MODEL TESTING
#------------------------------------------------------------------------------------------------------------------------------

def fit_model_testing(residuals, num_fit_parameters, yerr=None):
    """
    Takes residuals, error bars and number of fit parameters to calculate fit quality and model
    quality estimators
    
    returns
    chi2, chi2red, AIC, AICc, BIC
    """
    N = len(residuals)
    k = num_fit_parameters

    if yerr is None:
        yerr = np.ones(N)

    chi2 = sum((residuals / yerr) ** 2)
    chi2red = chi2 / (N-k)
    
    AIC = N*np.log( np.sum( residuals**2 ) ) + 2*k
    AICc = AIC + (2*k**2 + 2*k) / (N-k-1)
    BIC = N*np.log( np.sum( residuals**2 ) / N ) + np.log(N)*k
    #print(chi2, chi2red, AIC, AICc, BIC)
    return chi2, chi2red, AIC, AICc, BIC

#------------------------------------------------------------------------------------------------------------------------------
# Fit function
#------------------------------------------------------------------------------------------------------------------------------

def fit_sharedpoly_ratio(df, R_guess, y="nu_c", yerr="dnu_c", data_identifier="position", invert=False, groupsize=[4], degree=[3], mode='curvefit', x="time", keep_columns=["mc", "trap"],
                        bestfit="AICc", bestgroupsize="mindR", show=False):
    """Performs a shared polynom fit on y data in the pandas table. The data is mixed in the table and identified using a seperate column 
    (data_identifier) for us typically 'position' or in case of the cancellation method 'trap'. The group size and number of degrees of the
    polynom function can be lists to loop over these possible settings. The results will include the fit results of all settings with columns
    identifying the groupsizes, group indexes, number of polynom degree. Also all the polynomial parameters and fit quality estimators will
    be included.

    The structure of the input df should be (at least):
    | idx | keep_columns (e.g. mc, trap, ...) | x (e.g. time) | y (e.g. nu_c) | yerr (e.g. dnu_c) | data_identifier (e.g. position) |

    The structure of the output df will be:
    | idx | keep_columns | groupsize | degree | group | cycle_start | cycle_stop | x | R | dR | chi2 | chi2red | AIC | AICc | BIC | ion_numer | ion_denom | fit_parameter ... |

    Some notes on the mode options:
    polyfit: one side half of the data will be multiplied by a scanned/guessed Ratio value to overlapp the data and fit a single polynomial function on all data
             points using np.polyfit. Robust and very accurate, but slow. The error of the ratio is estimated with the chi2 + 1 rule.
    polyfit_fast: Same as polyfit, but the ratio is not simply scanned, the chi2 as a function of a guessed ratio is minimized using scipy.optimize.minimize. 
                  The error of the ratio is estimated with the chi2min + 1 rule, with the limitation that only very few points of chi2 vs. Ratio are known. These points
                  are fitted with a parabola and its fit parameter are used to estimate dR (cool).
    polyfit_sqrt: Same as polyfit, just both datasets are shifted to the "center": dataA*np.sqrt(R) and dataB*np.sqrt(R). This is just to test that this function gives the same
                  results as the original because we maybe need this function for curvefit.
    curvefit: This is a simultaneous fit of two datasets with shared parameter. In this case the data is not modified, the polynomial fit is performed two both datasets using
              deltafunctions as descriminators.
    curvefit_sqrt: I had the fear, that in the case of the simple fit function R*poly*deltafunction(dataA)+poly*deltafunction(dataB), a variation of the Ratio parameter
                   only changes the residuals of one(!) dataset. To get rid of this effect, I tried to use the fit function 
                   np.sqrt(R)*poly*deltafunction(dataA)+1/np.sqrt(R)*poly*deltafunction(dataB) but apparently the results are the same.
    curvefit_chebyshev: (NOT RECOMMENDED!) Implementation using Chebyshev polynomials. Doesn't work, maybe conceptional/programming problem.

    Args:
        df (pandas DataFrame): [Dataframe containing data to fit. Must include a column "masked" to drop points].
        R_guess (float): [Guessed ratio value].
        y (str, optional): [column name of y data]. Defaults to "nu_c".
        yerr (str, optional): [column name of yerr data]. Defaults to "dnu_c".
        invert (bool, optional): [to invert the ratio result]. Defaults to False.
        groupsize (list, optional): [list of group sizes to devide the data into]. Defaults to [4].
        degree (list, optional): [list of polynom degrees to use for fits]. Defaults to [3].
        mode (str, optional): [Fit-mode: Either numpy.polyfit with seperatly minimized R (R modifies data). Or scipy.curvefit
        using a simple implementation of a shared fit. All options: polyfit, polyfit_fast, polyfit_sqrt, 
        curvefit, curvefit_sqrt, curvefit_chebyshev]. Defaults to 'curvefit'.
        x (str, optional): [column name of x data]. Defaults to "time".
        keep_columns (list, optional): [list of column names to adopt from the input df]. Defaults to ["mc", "trap"].
        bestfit (str, optional): [method for best fit descision, options are: chi2, ftest, AIC, AICc, BIC]. Defaults to "AICc".
        bestgroupsize (str, optional): [method for best group size descision, options are: mindR, chi2red, birgeratio. If you performed
        multiple measurements, this "best" grouping is rather analysed for all of the measurements.]. Defaults to "mindR".
        show (bool, optional): [to plot more or less results of substeps]. Defaults to False.

    Returns:
        [pandas DataFrame]: [Dataframe with original data]
        [pandas DataFrame]: [Dataframe with all fit results. When multiple groupsizes or degrees are tested, this results table will include
        all these results and meaningfull final Ratio has to be calculated as an average of results from different fit parameters!]
        [pandas DataFrame]: [Dataframe with best poly degree fit results. When multiple groupsizes or degrees are tested, this results table 
        will include the best results for each group size, but still different results for different group sizes.]
        [pandas DataFrame]: [Dataframe with best fit results regarding group size and polynomial degree.]
    """
    # check input mode in the beginning
    mode_list = ["polyfit", "polyfit_fast", "polyfit_sqrt", "curvefit", "curvefit_sqrt", "curvefit_chebyshev"]
    if mode not in mode_list:
        print('please use one of the following modes for fitting:', mode_list)
        return None

    # inverting needs inverted guess
    if invert:
        R_guess = 1/R_guess

    # sort and remove offset of x axis
    if x.startswith("time"):
        df[x] = pd.to_datetime(df[x])    
    df.sort_values(by=x, inplace=True)
    x0 = df[x].iloc[0]
    xrel = x+'_rel'
    df[xrel] = df[x] - x0
    if x.startswith("time"):
        df[xrel] = df[xrel].dt.total_seconds()
    df.sort_values(by=x, inplace=True)

    # get the positions, we need to get the datasets accordingly
    data_idents = df[data_identifier].unique()
    if len(data_idents) != 2:
        print("there are less or more than 2 unique *data_identifiers* (positions?) in this subset! It has to be 2!")
        return None

    # lets prepare the results already to see what we want to have in the end
    results_columns = keep_columns.copy()
    results_columns.extend(["groupsize", "degree", "group", "cycle_start", "cycle_stop", x, "R", "dR", "chi2", "chi2red", "AIC", "AICc", "BIC", "ion_numer", "ion_denom"])
    results_columns.extend( ["c"+str(num) for num in range(max(degree) + 1)] )
    results_df = pd.DataFrame()
    best_df = pd.DataFrame() # for best fits

    # lets loop over the groupe sizes, this is the first level of settings variation.
    for gsize in groupsize:
        # group the data
        df = auto_group_subset(df, gsize, gsize-1, sortby=[x, data_identifier], groupby=data_identifier, group_key="group")
        #print(df.group)

        # removed masked data for analysis
        #df_masked = df[df["masked"] == True]
        df = df[df["masked"] == False]

        # to create a "best fit in this group size" df, we store the groups results in a seperated df and 
        # extract the best fit afterwards.
        gsize_df = pd.DataFrame()

        # looping over poly settings
        for polydegree in degree:

            # NOTE: in the next loop we already check for enough data to fit the polynomial
            #if (2*gsize - 2 - polydegree) < 2 and not gsize<=0:
            #    continue
            #if gsize==0 and (len(df) - polydegree - 2) < 2:
            #    continue
            print(">> group sizes:", gsize, "polynom degree:", polydegree)

            # iterate over each group and fit it with the polynomial given
            for group_name, df_group in df.groupby("group"):
                N = len(df_group)
                freedom = N - polydegree - 2 # poly + 1 parameter + 1 for ratio
                if freedom < 2:
                    continue
                if show: print(">>> group index:", group_name, "data points:", N, "freedom:", freedom)

                cycle_start = df_group["cycle"].min()
                cycle_stop = df_group["cycle"].max()
                mean_x = df_group[x].mean()
                data_idents = df[data_identifier].unique()
                if len(data_idents) != 2:
                    print("there are less or more than 2 unique *data_identifiers* (positions?) in this subset! It has to be 2!")
                    continue
                # Sorting positions is important to dont accidentally flip the ratio by having a different 
                # position temporally first due to unfortunate grouping/filtering
                data_idents.sort()
                
                if show: print(" °°° subfit for", gsize, group_name, polydegree, cycle_start, cycle_stop, data_idents, " °°° ")

                # new relative position on x axis (WHY: The fit works better if you are near to x=0)
                start_x = df_group[xrel].iloc[0]
                df_group["grp_xrel"] = df_group[xrel] - start_x
                df_group.sort_values(by="grp_xrel", inplace=True)

                # ion specii
                if not invert:
                    iona = df[ df[data_identifier] == data_idents[0] ]["ion"].iloc[0] # numerator
                    ionb = df[ df[data_identifier] == data_idents[1] ]["ion"].iloc[0] # denominator
                else:
                    iona = df[ df[data_identifier] == data_idents[1] ]["ion"].iloc[0] # numerator
                    ionb = df[ df[data_identifier] == data_idents[0] ]["ion"].iloc[0] # denominator

                # constant data
                x_data = df_group["grp_xrel"].to_numpy()
                yerr_data = df_group[yerr].to_numpy()

                # fit settings
                resolution = 1e-12
                Rrange = 4e-10

                # choose the mode: polyfit, polyfit_fast, polyfit_sqrt, curvefit, curvefit_sqrt, curvefit_chebyshev

                #### POLYFIT #### (plain and submethods) #################################################################################################################
                # PLAIN and SQRT
                if mode=="polyfit" or mode=="polyfit_sqrt": 
                    # prepare data lets try it the pandas way
                    sigma = 1/yerr_data

                    chi2_list = []
                    fitp_list = []
                    ratio_list = []

                    df_group['ytest'] = df_group[y]
                    
                    # run through a list of test ratios around the guessed one, multiply the one half of the data up the other, fit and store according chi2
                    Rtest_list = np.arange(R_guess-Rrange/2, R_guess+Rrange/2, resolution)
                    for Rtest in Rtest_list:
                        if not mode.endswith("_sqrt"):
                            df_group['ytest'] = np.where(df_group[data_identifier] == data_idents[1], # multiply denominator with guessed ratio
                                                                    df_group[y] * Rtest,
                                                                    df_group[y])
                        else:
                            df_group['ytest'] = np.where(df_group[data_identifier] == data_idents[1], # multiply denominator with guessed ratio sqrt 
                                                                    df_group[y] * np.sqrt(Rtest),
                                                                    df_group[y] / np.sqrt(Rtest)) # and devide nominator with guessed ratio sqrt 
                                                                    # this should move the data together from both sides
                        
                        y_data = df_group['ytest'].to_numpy()
                        if show and Rtest==R_guess:
                            plt.plot(x_data, y_data)
                            plt.show()

                        fit_params = np.polyfit( x_data, y_data, polydegree, w=sigma)
                        residuals = y_data - np.polyval(fit_params, x_data)
                        ratio_list.append(Rtest)
                        fitp_list.append(fit_params)
                        chi2_list.append( np.sum( (residuals)**2/yerr_data**2 ) )                
                
                    # optimal values from analysis of residuals
                    minimal_chi2 = min(chi2_list)
                    idx = chi2_list.index(minimal_chi2)
                    optimal_ratio = ratio_list[idx]

                    if show:
                        # chi2 vs ratio
                        print(minimal_chi2, idx, optimal_ratio-1) 
                        plt.plot(ratio_list, chi2_list)
                        plt.plot(ratio_list, [minimal_chi2]*len(ratio_list))
                        plt.show()

                    # calculation of ratio error from chi2
                    chi2_list = np.asarray(chi2_list)
                    posdR1 = np.argmin(np.abs( chi2_list - (minimal_chi2 + 1) )) # nearest
                    chi2_list = np.delete(chi2_list, posdR1) # second ... 
                    posdR2 = np.argmin(np.abs( chi2_list - (minimal_chi2 + 1) )) # ... nearest
                    dR1 = abs( ratio_list[posdR1] - optimal_ratio )
                    dR2 = abs( ratio_list[posdR2] - optimal_ratio )
                    dR = max([dR1, dR2])
                    if show: print('chi2 error estimation: (should be "the same")', posdR1, posdR2, dR1, dR2)

                # FAST
                if mode=="polyfit_fast": 
                    sigma = 1/yerr_data
                    bounds = (R_guess-20*Rrange, R_guess+20*Rrange)
                    fast_guess = random.uniform(R_guess-Rrange/10, R_guess+Rrange/10) # this leads to better results, since more variation happens

                    df_group['ytest'] = df_group[y]
                    
                    chi2_list = []
                    ratio_list = []

                    def function(Rtest):
                        nonlocal ratio_list, chi2_list

                        df_group['ytest'] = np.where(df_group[data_identifier] == data_idents[1], # multiply denominator with guessed ratio
                                        df_group[y] * Rtest,
                                        df_group[y])
                        #df_group['ytest'] = np.where(df_group[data_identifier] == data_idents[1], # multiply denominator with guessed ratio sqrt 
                        #                                        df_group[y] * np.sqrt(Rtest),
                        #                                        df_group[y] / np.sqrt(Rtest)) # and devide nominator with guessed ratio sqrt 
                                                                # this should move the data together from both sides
                        y_data = df_group['ytest'].to_numpy()

                        fit_params = np.polyfit( x_data, y_data, polydegree, w=sigma)
                        residuals = y_data - np.polyval(fit_params, x_data)
                        chi2 = np.sum( (residuals)**2/yerr_data**2 )

                        if Rtest > bounds[0] and Rtest < bounds[1]:# and chi2 < 1e5:
                            ratio_list.append(Rtest[0])
                            chi2_list.append(chi2)
                        return chi2

                    # run through a list of test ratios around the guessed one, multiply the one half of the data up the other, fit and store according chi2
                    
                    res = minimize(function, [fast_guess], bounds=[bounds], tol=1e-21)
                    minimal_chi2 = function(res.x)
                    optimal_ratio = res.x[0]

                    # fit parabola to chi2 data:
                    def parabola(x, c):
                        return c*(x-optimal_ratio)**2+minimal_chi2

                    dRtest = np.ptp(ratio_list)
                    dchi2 = np.ptp(chi2_list)
                    p0 = [(dchi2/dRtest)**2]
                    popt, pcov = curve_fit(parabola, ratio_list, chi2_list, p0, absolute_sigma=True, method='lm', ftol=1e-18) #, full_output=True)
                    perr = np.sqrt(np.diag(pcov))
                    dR = 1/np.sqrt( popt[0] )
                    ddR = 1/2 * 1/np.sqrt( popt[0] ) / popt[0] * perr[0]
                    if show: print('chi2 vs. R fit error estimation:', dR, ddR)

                    if show:
                        plt.scatter(ratio_list, chi2_list)
                        rlin = np.linspace(min(ratio_list), max(ratio_list), 200)
                        plt.plot(rlin, parabola(rlin, popt[0]))
                        plt.hlines(minimal_chi2+1, min(ratio_list), max(ratio_list) )
                        plt.show()

                # FINALIZE FOR ALL POLYFIT METHODS
                if mode.startswith("polyfit"):
                    # fit best ratio again to get residuals
                    if mode.endswith("_sqrt"):
                        df_group['ytest'] = np.where(df_group[data_identifier] == data_idents[1], # multiply denominator with guessed ratio
                                                                                        df_group[y] * np.sqrt(optimal_ratio),
                                                                                        df_group[y] / np.sqrt(optimal_ratio))
                    else:
                        df_group['ytest'] = np.where(df_group[data_identifier] == data_idents[1], # multiply denominator with guessed ratio
                                                                df_group[y] * (optimal_ratio),
                                                                df_group[y])
                    y_data = df_group['ytest'].to_numpy()
                    optimal_fit_results = np.polyfit( x_data, y_data, polydegree, w=sigma)
                    residuals = y_data - np.polyval(optimal_fit_results, x_data)

                    if show:
                        # modified data and poly fit
                        xlin = np.linspace(np.min(x_data), np.max(x_data), 1000)
                        plt.plot(xlin, np.polyval(optimal_fit_results, xlin), '-')
                        plt.errorbar(x_data, y_data, yerr=yerr_data, fmt='o')
                        plt.show()

                    # get all estimators for comparison
                    chi2, chi2red, AIC, AICc, BIC = fit_model_testing(residuals, polydegree+1, yerr=yerr_data)

                #### ####### ###############################################################################################################################


                #### CURVEFIT ##############################################################################################################################
                if mode.startswith("curvefit"):
                    y_data = df_group[y].to_numpy()
                    positions = df_group[data_identifier].to_numpy()

                    # get an array of the positions, this should work vor any type of postion indicator, str or int
                    __, pos = np.unique(positions, return_inverse=True)
                    #if not invert:
                    #    pos = 1 - pos


                    # NOTE: WHY HAVE THE SQRT VERSION OF THE FIT FUNCTION? 
                    #       np.multiply( pos, np.polyval(p, x) ) + np.multiply( (1-pos), R*np.polyval(p, x) ) 
                    #       np.multiply( pos, 1/( np.sqrt(R) )*np.polyval(p, x) ) + np.multiply( (1-pos), ( np.sqrt(R) )*np.polyval(p, x) ) 
                    # ANSWER: The function above is basically a case decision: at one position a polynom is fitted and on the 
                    # the other position the same polynom * R. The question with this was: Will minimizing the chi2 (y-yfit) by modifying R 
                    # effect the residuals (chi2) of both positions or just one position? If it just changes the residuals of one position
                    # that might be a problem for the minimizer in the fit because it might underestimate the error of the fit parameter R.
                    # The solution is the function below, where R is included in both datasets/polynomials. This should either result in 
                    # an higher error if these concerns are correct or are just the same if the above is wrong.
                    def polyratio(x, *p):
                        """
                        p is the list of fit paramters, high polynom order to low, last element is R
                        """
                        p = np.asarray(p)
                        p = p.flatten()
                        R = p[-1]
                        p = p[:-1]
                        # to check scipy curve fit:
                        #print('minimizing...\t', p, '\t', R)
                        poly_y = np.polyval(p, x)
                        return np.multiply( pos, poly_y ) + np.multiply( (1-pos), R*poly_y )

                    def polyratio_sqrt(x, *p):
                        """
                        p is the list of fit paramters, high polynom order to low, last element is R
                        """
                        p = np.asarray(p)
                        p = p.flatten()
                        R = p[-1]
                        p = p[:-1]
                        # to check scipy curve fit:
                        #print('minimizing...\t', p, '\t', R)
                        poly_y = np.polyval(p, x)
                        return np.multiply( pos, 1/( np.sqrt(R) )*poly_y ) + np.multiply( (1-pos), ( np.sqrt(R) )*poly_y )

                    def polyratio_chebyshev(x, *p):
                        """
                        p is the list of fit paramters, high polynom order to low, last element is R
                        """
                        p = np.asarray(p)
                        p = p.flatten()
                        R = p[-1]
                        p = p[:-1]
                        # to check scipy curve fit:
                        #print('minimizing...\t', p, '\t', R)
                        poly_y = Chebyshev(p)(x) 
                        return np.multiply( pos, poly_y ) + np.multiply( (1-pos), R*poly_y )

                    # start parameters
                    p0 = [0]*(polydegree+1)
                    p0[-1] = y_data[0] 
                    #p0.append(1) # R
                    p0.append(R_guess) # R

                    # choose the function
                    if mode=='curvefit':
                        fit_function = polyratio
                    elif mode=='curvefit_sqrt':
                        fit_function = polyratio_sqrt
                    elif mode=='curvefit_chebyshev':
                        fit_function = polyratio_chebyshev

                    # fitting
                    popt, pcov = curve_fit(fit_function, x_data, y_data, p0, yerr_data, absolute_sigma=True, method='lm', ftol=1e-15) #, full_output=True)
                    perr = np.sqrt(np.diag(pcov))
                    #print(popt, perr)
                    #pprint(pcov)
                    R, dR = popt[-1], perr[-1]
                    optimal_ratio = R

                    # correct fit results parameters according to function
                    if mode=='curvefit':
                        popt_corr = popt * R
                    elif mode=='curvefit_sqrt':
                        popt_corr = popt * np.sqrt(R)
                    elif mode=='curvefit_chebyshev':
                        c_cheby = popt[:-1] * R
                        c_poly = cheb2poly(c_cheby)[::-1] # returned coeff order is inverse (from lowest to highest)
                        popt_corr = np.append(c_poly, R)
                    optimal_fit_results = popt_corr[:-1]

                    ### NOTE: estimate dR if fit does not supply:
                    if np.isinf(dR) or dR == 0.0 or np.isnan(dR):
                        popt_test = popt
                        R_span = np.arange(R - 1e-10, R + 1e-10, 1e-12)
                        chi2_list = []
                        for test_R in R_span: # This is probably optimizable to get rid of the loop
                            popt_test[-1] = test_R
                            chi2_list.append( np.sum( (fit_function(x_data, popt_test) - y_data)**2/(yerr_data**2) ) )
                        chi2_list = np.asarray(chi2_list)
                        min_chi2 = chi2_list.min()
                        half_span_size = int(len(chi2_list)/2)
                        left = chi2_list[:half_span_size]
                        right = chi2_list[half_span_size:]
                        left_idx = (np.abs(left - min_chi2 - 1)).argmin()
                        right_idx = (np.abs(right - min_chi2 - 1)).argmin() + half_span_size
                        Rleft, Rright = R_span[int(left_idx)], R_span[int(right_idx)]
                        dR = max(R-Rleft, Rright-R)
                        if show:
                            plt.plot(R_span, chi2_list)
                            plt.plot(R_span, [min_chi2+1]*len(R_span))
                            plt.title("estimating dR via chi2.min()+1 method")
                            plt.show()
                        #print('no dR, fixing...', left_idx, right_idx, Rleft, Rright, dR)
                        print('no dR from pcov, fixed with chi2 vs. R error estimation: dR=', dR)

                    residuals = y_data - fit_function(x_data, popt)
                    #print('residuals', np.sum( residuals**2 ))
                    chisq = sum((residuals / yerr_data) ** 2)

                    if show:
                        xlin = np.linspace(np.min(x_data), np.max(x_data), 1000)
                        plt.plot(xlin, np.polyval(popt_corr[:-1], xlin), '-', label=mode+", degree "+str(polydegree))
                        #plt.plot(xlin, np.polyval(popt[:-1], xlin), '-')
                        ycorr = []
                        for yi, posi in zip(y_data, pos):
                            if posi == 1:
                                ycorr.append(yi*R)
                            else:
                                ycorr.append(yi)
                        plt.errorbar(x_data, ycorr, yerr=yerr_data, fmt='o', label="data, group size "+str(gsize)+", index "+str(group_name))
                        plt.legend()
                        plt.title("ratio via shared polynom fit")
                        plt.show()

                    # get all estimators for comparison
                    chi2, chi2red, AIC, AICc, BIC = fit_model_testing(residuals, polydegree+2, yerr=yerr_data)

                if invert:
                    optimal_ratio = 1/optimal_ratio
                    dR = dR*optimal_ratio**2

                # summing up the data
                data = []
                for kc in keep_columns:
                    try:
                        val = df_group[kc].mean()
                    except:
                        val = df_group[kc].unique()[0]
                    data.append(val)

                data.extend( [ int(gsize), int(polydegree), int(group_name), int(cycle_start), int(cycle_stop), mean_x, optimal_ratio, dR, chi2, chi2red, AIC, AICc, BIC, iona, ionb ] )
                data.extend( optimal_fit_results[::-1] )
                data.extend( np.full(max(degree) - polydegree, np.nan) )
                data_series = pd.Series(data=data, index=results_columns)
                #print(data_series)
                results_df = results_df.append( data_series, ignore_index=True )
                gsize_df = gsize_df.append( data_series, ignore_index=True )

        # back to loop level of group sizes
        # choose best fit of each group:
        if gsize_df.empty:
            continue
        
        for gname, grp in gsize_df.groupby("group"):
            # TODO: add f-test here?
            # get best
            min_best = grp[bestfit].min()
            best_data = grp[ grp[bestfit] == min_best ]
            #print(len(best_data))
            if len(best_data) > 1:
                print('multiple best results, taking the first one!')
                best_data = best_data.iloc[0]
            best_df = best_df.append( best_data, ignore_index=True )
            
            # plot ratios vs. poly degree:
            if show:
                plt.errorbar(grp.degree.to_numpy(), grp.R.to_numpy(), grp.dR.to_numpy())
                plt.xlabel("poly degree")
                plt.ylabel("R")
                plt.title("groupsize "+str(gsize)+" group index "+str(gname))
                plt.show()
                plt.plot(grp.degree.to_numpy(), grp[bestfit])
                plt.xlabel("poly degree")
                plt.ylabel(bestfit)
                plt.title("groupsize "+str(gsize)+" group index "+str(gname))
                plt.show()
            print("> best in group idx/size: ", gname, gsize, "with criterion:", bestfit, "degree:", int(best_data['degree']) )
        
    # get also the best results regarding grouping
    if best_df.empty:
        best_results = pd.DataFrame()
    elif len(best_df.groupsize.unique()) == 1:
        print('only one groupsize, no optimization possible', len(best_df.groupsize.unique()))
        best_results = best_df.copy(deep=True)
    else:
        best_results = statistics.find_best_ratio_results(best_df, bestcriterion=bestgroupsize, show=show)

    return df, results_df, best_df, best_results

def plot_best_results(nu_c_data, best_results, groupby=['trap'], x="time_p", y="nu_c", yerr="dnu_c", ratio='R', line_width=15):
    nu_c_data = nu_c_data[ nu_c_data["masked"] == False ]
    for grpname, subset_results in best_results.groupby(groupby):
        subset_data = nu_c_data.copy()
        try:
            iterator = iter(grpname)
        except:
            grpname = [grpname]
        for key, val in zip(groupby, np.asarray(grpname)):
            subset_data = subset_data[ subset_data[key] == val ]

        plot_starttime = subset_data[x].min()
        
        for parameters, grp in subset_results.groupby(["mcycle", "group"]):
            mc = parameters[0]
            group_name = parameters[1]

            R = grp[ratio].iloc[0]
            dR = grp["d"+ratio].iloc[0]

            cycle_start = grp["cycle_start"].iloc[0]
            cycle_stop = grp["cycle_stop"].iloc[0]
            
            grp_data = subset_data[ (subset_data["mcycle"] == mc) & (subset_data["cycle"] >= cycle_start) & (subset_data["cycle"] <= cycle_stop)]

            timetostart = (grp_data[x].min() - plot_starttime).total_seconds()
            timeduration = (grp_data[x].max() - grp_data[x].min()).total_seconds()
            timemax = grp_data[x].max()
            
            x_data = grp_data[x].to_numpy()

            multiply = R # inverted
            pos = "position_1"
            
            original_std = grp_data[y].std()
            y_data = np.where(grp_data["position"] == pos, # multiply denominator with guessed ratio
                                                    grp_data[y] * R,
                                                    grp_data[y])  
            if y_data.std() > original_std:
                pos = "position_2"
                y_data = np.where(grp_data["position"] == pos, # multiply denominator with guessed ratio
                                        grp_data[y] * R,
                                        grp_data[y])
                multiply = 1


            yerr_data = grp_data[yerr].to_numpy()
            
            plt.errorbar(x_data, y_data, yerr_data, fmt=".")
            
            # now the fit:
            fitparams = []
            for i in range(99):
                try:
                    par = float(grp["c"+str(i)])*multiply
                    if np.isnan(par): break
                    fitparams.append(par)
                except KeyError:
                    break
            
            xlin = np.arange(0, timeduration, 200)  
            ydata = np.polyval(fitparams[::-1], xlin)

            startdatetime = plot_starttime+timedelta(seconds=timetostart)
            xlindtime = np.array([startdatetime + timedelta(seconds=i) for i in xlin])
            plt.plot(xlindtime, ydata, linewidth=line_width, label="degree:"+str(len(fitparams)-1))
            
        plt.legend()
        plt.show()
        

def compare_groupsizes_fits(nu_c_data, all_results, groupsizes=[4,0], groupby=['trap'], x="time_p", y="nu_c", yerr="dnu_c", ratio='R', line_width=15):
    nu_c_data = nu_c_data[ nu_c_data["masked"] == False ]

    all_results = all_results[ all_results["groupsize"].isin(groupsizes) ]

    # get results of this groupby (trap)
    for grpname, subset_results in all_results.groupby(groupby):

        # get data of this groupby (trap)
        subset_data = nu_c_data.copy()
        try:
            iterator = iter(grpname)
        except:
            grpname = [grpname]
        for key, val in zip(groupby, np.asarray(grpname)):
            subset_data = subset_data[ subset_data[key] == val ]

        plot_starttime = subset_data[x].min()
        #for groupsize, subsubset_results:

        for parameters, grp in subset_results.groupby(["mcycle", "group"]):
            mc = parameters[0]
            group_name = parameters[1]

            R = grp[ratio].iloc[0]
            dR = grp["d"+ratio].iloc[0]

            cycle_start = grp["cycle_start"].iloc[0]
            cycle_stop = grp["cycle_stop"].iloc[0]
            
            grp_data = subset_data[ (subset_data["mcycle"] == mc) & (subset_data["cycle"] >= cycle_start) & (subset_data["cycle"] <= cycle_stop)]

            timetostart = (grp_data[x].min() - plot_starttime).total_seconds()
            timeduration = (grp_data[x].max() - grp_data[x].min()).total_seconds()
            timemax = grp_data[x].max()
            
            x_data = grp_data[x].to_numpy()

            multiply = R # inverted
            pos = "position_1"
            
            original_std = grp_data[y].std()
            y_data = np.where(grp_data["position"] == pos, # multiply denominator with guessed ratio
                                                    grp_data[y] * R,
                                                    grp_data[y])  
            if y_data.std() > original_std:
                pos = "position_2"
                y_data = np.where(grp_data["position"] == pos, # multiply denominator with guessed ratio
                                        grp_data[y] * R,
                                        grp_data[y])
                multiply = 1


            yerr_data = grp_data[yerr].to_numpy()
            
            plt.errorbar(x_data, y_data, yerr_data, fmt=".")
            
            # now the fit:
            fitparams = []
            for i in range(99):
                try:
                    par = float(grp["c"+str(i)])*multiply
                    if np.isnan(par): break
                    fitparams.append(par)
                except KeyError:
                    break
            
            xlin = np.arange(0, timeduration, 200)  
            ydata = np.polyval(fitparams[::-1], xlin)

            startdatetime = plot_starttime+timedelta(seconds=timetostart)
            xlindtime = np.array([startdatetime + timedelta(seconds=i) for i in xlin])
            plt.plot(xlindtime, ydata, linewidth=line_width, label="degree:"+str(len(fitparams)-1))
            
        plt.legend()
        plt.show()
        
