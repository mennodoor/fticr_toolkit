import numpy as np
import pandas as pd
from scipy.stats import t, norm
from matplotlib import pyplot as plt
from copy import deepcopy

def mean_std_dmean(values, dvalus):
    pass

def mean_and_stderror(array, axis=None):
    """
    This returns the mean and error of the mean
    """
    mean = np.mean(array, axis=axis)
    std = np.std(array, axis=axis, ddof=1)/np.sqrt( np.size(array, axis=axis) )
    return mean, std

def student_68(number_of_points):
    df = number_of_points - 1
    return t.interval(0.6827, df)[1]

def mean_and_error(value, dvalue = None, student_limit = 10):
    N = len(value)

    student = False
    if N < student_limit:
        student = True

    R, err_in, err_out, chi2red = complete_mean_and_error(value, dvalue, student=student)
    err = max([err_in, err_out])

    return R, err

def complete_mean_and_error(value, dvalue = None, student = False):

    # the resulting mean variable is called R here, due to ratio stuff, but its usable
    # for everything

    N = len(value)
    if N == 1:
        return float(value), float(dvalue), np.NaN, np.NaN

    value = np.array(value)
    if dvalue is not None:
        dvalue = np.array(dvalue)


    nan_flag = np.isnan(np.sum(value))
    #print(nan_flag, value, dvalue)
    nan_idx = np.argwhere(np.isnan(value))

    if dvalue is not None:
        dvalue[dvalue == 0] = np.nan
        nan_idx = np.append(nan_idx, np.argwhere(np.isnan(dvalue)))
        nan_flag = nan_flag or np.isnan(np.sum(dvalue))

    if nan_flag:
        value = np.delete(value, nan_idx)
        if dvalue is not None:
            dvalue = np.delete(dvalue, nan_idx)
        #print(value, dvalue)

    # if there are errors for the values given, we calculate the weighted mean/error
    if dvalue is not None:
        dvalue = np.array(dvalue)

        # We are using variance weights, meaning the weights for calculating error
        # (and also chi2 and such) are calculated by taking the inverse of the single
        # value error squared.
        # This implicates our assumption that our errors assigned to the single values
        # are sigma-like errors. This can be tested in the end with the 'birge ratio'
        # which will show if the individual errors might be under or overestimated.

        # variance weights
        # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Variance_weights
        weight = 1 / dvalue**2

        # weighted mean
        # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Mathematical_definition
        R = np.sum(weight * value) / np.sum(weight)
        residual = value-R

        # this is the standard error of the weighted mean
        # its just the sqrt of the qaudratic sum of errors, so normal error propagation
        # NOTE: often refered to in 5trap thesises as the "internal error"
        # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Mathematical_definition
        # also Roux PhD, (9.25)
        err_in = np.sqrt( 1/np.sum(weight) )
        
        # weighted sample variance
        # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
        sigma_weighted = np.sqrt( np.sum( weight * residual**2 ) / np.sum(weight) )

        # Roux PhD, (9.26)
        # NOTE: often refered to in 5trap thesises as the "external error"
        #err_out = np.sqrt( np.sum( weight * residual**2 ) / ( (N-1) * np.sum(weight) ) )
        err_out = sigma_weighted / np.sqrt(N-1)
        
        # reduced chi2, 
        # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Correcting_for_over-_or_under-dispersion
        chi2red = np.sum( weight * residual**2) / (N-1)
    else:
        # normal mean
        R = np.sum(value) / N
        residual = value-R

        # infinite errors...
        err_in = float("-inf")
        chi2red = float("-inf")

        # Roux PhD, (9.27)
        # TODO: Here i think it should also be just devided by sqrt(N) not N
        err_out = np.sqrt( np.sum( residual**2 ) / ( (N-1) * N ) )

    # multiply student distribution factor
    if student:
        cf = student_68(N)
    else:
        cf = 1
    
    return R, err_in*cf, err_out*cf, chi2red



def assign_stderr_subsets(dset, groupby=["cycle"], val="phase", dval="dphase", global_undrift=False, x="time", degree=5, show=False):
    """Calculates stds of column val in each group and assigns this value to column dval.

    Args:
        dset (pandas DataFrame): input data.
        groupby (list, optional): Defaults to ["cycle"].
        val (list, optional): Defaults to ["phase"].
        dval (list, optional): Defaults to ["dphase"].

    Returns:
        pandas DataFrame: updated Dataframe
    """
 

    dset_temp = dset.copy(deep=True)
    if global_undrift:
        values = dset_temp[val].to_numpy()
        valuesc = deepcopy(values)

        try:
            if x.startswith("time"):
                xdata = (dset_temp[x].astype('int64')//1e9).to_numpy()
                xdata -= xdata.min()
            else:
                xdata = dset_temp[x].to_numpy()
        except:
            xdata = range(len(dset))

        p = np.polyfit(xdata, values, degree)
        dset_temp[val] = values - np.polyval(p, xdata)
        if show:
            plt.plot(xdata, valuesc)
            plt.plot(xdata, np.polyval(p, xdata))
            plt.show()

            plt.plot(xdata, dset_temp[val].to_numpy())
            plt.show()

    stds = []
    for grpname, grp in dset_temp.groupby(groupby):

        idx = grp.index
        stdval = float(grp[val].std())

        # check inner error and use that if its bigger
        try:
            errors = grp[dval].to_numpy()
            err_in = np.sqrt( 1/np.sum(1/errors**2) )
            if err_in > stdval:
                stdval = err_in
        except:
            pass

        stds.append(stdval)
        dset[dval].loc[idx] = stdval

    #print(np.mean(stds), stds)
    return dset


def assign_stderr_undrift(dset, x="time", val="phase", dval="dphase", degree=5, show=False):
    """Calculates stds of column val in each group and assigns this value to column dval.

    Args:
        dset (pandas DataFrame): input data.
        groupby (list, optional): Defaults to ["cycle"].
        val (list, optional): Defaults to ["phase"].
        dval (list, optional): Defaults to ["dphase"].

    Returns:
        pandas DataFrame: updated Dataframe
    """

    dset_temp = dset.copy(deep=True)

    values = dset_temp[val].to_numpy()
    valuesc = deepcopy(values)
    if x.startswith("time"):
        xdata = (dset_temp[x].astype('int64')//1e9).to_numpy()
        xdata -= xdata.min()
    else:
        xdata = dset_temp[x].to_numpy()
    p = np.polyfit(xdata, values, degree)
    dset_temp[val] = values - np.polyval(p, xdata)
    if show:
        plt.plot(xdata, valuesc)
        plt.plot(xdata, np.polyval(p, xdata))
        plt.show()

        plt.plot(xdata, dset_temp[val].to_numpy())
        plt.show()
    
    stdval = float(dset_temp[val].std())
    print(stdval)
    dset[dval] = stdval
    return dset

def average_subsets(df, groupby=["cycle"], errortype="stderror", columns=["phase", "time"], dcolumns=["dphase", None], masked=True):
    """
    Here we will loop over subsets like cycle and average for example the phase and the time.

    Values / columns which are not averaged will get the first value (according to initial sorting). 
    """
    pd.options.mode.chained_assignment = None  # default='warn'
    
    dset = df.copy()
    if masked:
        dset = dset[dset["masked"]==False]

    new_data = pd.DataFrame()
    for groupname, group in dset.groupby(groupby):
        this_first = group.iloc[0]
        
        for column, dcolumn in zip(columns, dcolumns): 
            if 'datetime' in str(group[column].dtype):
                #print(column, 'is a datetime column, converting to epoch and back')
                array = (group[column].astype('int64')//1e9).to_numpy()
            else:
                array = group[column].to_numpy()

            try:
                darray = group[dcolumn].to_numpy()
            except:
                #print("error column does not exist")
                darray = None

            #mean_val, err = mean(array)
            if errortype == "stderror" or darray is None:
                mean_val, err = mean_and_stderror(array)
            elif errortype == "weighted":
                mean_val, err = mean_and_error(array, dvalue = darray)
                #print(array, darray, mean_val, err)

            if 'datetime' in str(group[column].dtype):
                mean_val = pd.to_datetime(mean_val, unit='s')

            this_first[column] = mean_val

            if 'datetime' not in str(group[column].dtype):
                this_first["d"+column] = err
 
        new_data = new_data.append(this_first, ignore_index=True)

    #pd.options.mode.chained_assignment = chained_assignment_set  # default='warn'
    return new_data

#######################################################
### this is from alex analysis package, plus corrections plus comments


def test_normality(sample, dsample):
    pass




def merge_odd_even_results(df_odd, df_even, dRtype='max'):
    """
    If we have an analysis method like the naive method, the ratios can be calculated by data from one cycle (odd)
    or from data of two cycles (even, the second position and the first from the next cycle). Odd and even ratios are
    both shifted by a fixed offset corresponding to the common and constant linear drift of the data, but shifted in 
    opposite directions, e.g.:
    R_odd  = R_real + dR
    R_even = R_real - dR
    R_real = (R_odd + R_even) / 2

    We will try to merge the data here.
    """
    #if 'masked' in df_odd:
    #    df_odd['masked'] = df_odd['masked'].astype('boolean')
    #    df_even['masked'] = df_even['masked'].astype('boolean')

    new_df = pd.DataFrame(columns=df_odd.columns)
    new_df.astype(df_odd.dtypes.to_dict())

    try:
        traps = df_odd.trap.unique() # raise when no trap column
        other_t = traps[1] # raise when single

        #new_df = pd.DataFrame()
        for t in traps:
            new_df = new_df.append( merge_odd_even_results( df_odd[ df_odd['trap']==t ], df_even[ df_even['trap']==t ]), ignore_index=True )
        if 'masked' in new_df:
            print(df_odd.masked.unique(), new_df.masked.unique())
            new_df['masked'] = new_df['masked'].astype('bool')
        return new_df
    except IndexError:
        pass
    except:
        raise

    try:
        measures = df_odd.measurement.unique() # raise when no trap column
        other_m = measures[1] # raise when single

        #new_df = pd.DataFrame()
        for m in measures:
            new_df = new_df.append( merge_odd_even_results( df_odd[ df_odd['measurement']==m ], df_even[ df_even['measurement']==m ]), ignore_index=True )
        #display(new_df)
        if 'masked' in new_df:
            print(df_odd.masked.unique(), new_df.masked.unique())
            new_df['masked'] = new_df['masked'].astype('bool')
        return new_df
    except AttributeError:
        pass
    except IndexError:
        pass
    except:
        raise

    #df_odd['cycle'] = df_odd['cycle'].apply(np.floor)
    #df_even['cycle'] = df_even['cycle'].apply(np.floor)
    #print(df_odd.position.unique(), df_even.position.unique())

    #new_df = pd.DataFrame()
    counter = 0

    #df = pd.concat( [df_odd, df_even], axis=0, ignore_index=True, sort=False)

    df = df_odd.append( df_even, ignore_index=True )
    #display(df)
    grouping = ['mcycle', 'cycle']
    if "subcycle" in df.columns:
        grouping = ['mcycle', 'cycle', 'subcycle']
    for grpname, grp in df.groupby(grouping):
        #display(grp)
        if len(grp) == 2:
            sub = grp.iloc[0]
            sub.R = (grp.R.iloc[0] + grp.R.iloc[1]) / 2
            sub.dR = max(grp.dR.iloc[0], grp.dR.iloc[1])
            new_df = new_df.append(sub, ignore_index=True)
        else:
            #display(grp)
            #print("lost", grpname)
            counter += 1
    print('lost', counter, 'len(odd/even/merged)', len(df_odd), len(df_even), len(new_df))

    """
    #df_odd.sort_values("time", inplace=True) # TODO: is this an issue? do i nead time sorting?
    #df_even.sort_values("time", inplace=True)

    while len(df_odd) > len(df_even):
        print("lost one odd")
        df_odd.drop(df_odd.tail(1).index,inplace=True) # drop last row

    while len(df_odd) < len(df_even):
        print("lost one even")
        df_even.drop(df_even.tail(1).index,inplace=True) # drop last row

    new_df = df_odd.copy()
    new_df.reset_index(drop=True, inplace=True)
    #print(len(df_odd), len(df_even), len(new_df))
    new_df['R'] = (df_odd.R.to_numpy() + df_even.R.to_numpy()) / 2.0
    
    if dRtype == 'max':
        dRmax = np.asarray([df_odd.dR.to_numpy(), df_even.dR.to_numpy()]).max(axis=0)
        new_df['dR'] = dRmax
    elif dRtype == 'mean':
        dRmean = (df_odd.dR.to_numpy() - df_even.dR.to_numpy()) / 2.0
        new_df['dR'] = dRmean
    else:
        print("dRtype not supported")
        raise TypeError
    """
    new_df['Rminus'] = new_df['R'] - 1
    return new_df



def find_best_ratio_results(best_df, groupby=["groupsize"], bestcriterion="mindR", show=False):
    best_results = pd.DataFrame() 

    test=1e9
    bestsize = None
    size_list = []
    ratio_list = []
    dratio_list = []
    for gsize, grp in best_df.groupby(groupby):
        R, inner, outer, chi2red = complete_mean_and_error(grp['R'].to_numpy(), grp['dR'].to_numpy(), student=True)
        dR = np.nanmax([inner, outer])
        size_list.append(gsize)
        ratio_list.append(R)
        dratio_list.append(dR)
        print("groupsize", gsize, "R=", R, "dR=", dR, "chi2red=", chi2red)

        if bestcriterion == 'mindR':
            if dR < test:
                best_results = grp.copy()
                bestsize = gsize
                test = dR
        if bestcriterion == "chi2red":
            diff = abs(1-chi2red)
            if diff < test:
                best_results = grp.copy()
                bestsize = gsize
                test = diff
        if bestcriterion == "birgeratio":
            if len(grp)==1:
                diff = np.sqrt( chi2red ) - 1
            else:
                diff = np.sqrt( chi2red/(len(grp)-1) ) - np.sqrt( 1/(len(grp)-1) )
            diff = abs(diff)
            if diff < test:
                best_results = grp.copy()
                bestsize = gsize
                test = diff
        if bestcriterion == 'AICc':
            mean = grp['AICc'].mean()
            if mean < test:
                best_results = grp.copy()
                bestsize = gsize
                test = mean
    print("> best "+str(groupby)+" was:", bestsize)
    if show:
        plt.errorbar(size_list, ratio_list, dratio_list)
        plt.show()

    return best_results