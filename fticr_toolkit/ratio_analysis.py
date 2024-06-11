import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import plotly.graph_objs as go
from fticr_toolkit import polynom_fit

np.set_printoptions(precision=12)

# this function can be used for naive analysis, for calculating ratios after interpolation or preparing trap ratio data for
# the cancellation method
def calc_ratios(dset, y=['nu_c'], yerr=['dnu_c'], groupbys=['cycle', 'position'], identifier="trap", ident_types=[2,3],
                keep_columns=['mcycle', 'ion'], additional_identifiers=['ion'], mean_columns=['time', 'epoch']):
    """Calcultes the ratio of nu_c frequencies (or the ratio of any other column) for the traps or the positions given
    by the identifier value (or any other identifier). The cycle and position (or any other two groupbys)
    have to be the same value.

    If we do this with the idea of applying the cancellation method, the ratio can be calculated the following way when using the default parameters:
    The trap is the identifier and we group by cycle and postion, so we end up with trap ratio for each position in each cycle (a ratio of the two nu_c measured
    at the same time). Two ratios of one cycle, called r_p1 & r_p2 can be used to calculate the mass ratio the 'naive way':
    R = np.sqrt(r_p1*r_p2)
    Polynomial fit method would be possible too, since:
    r_p1 and r_p2 should follow a polynomial form (TODO: actually a poly/poly function, so called rational function), so you might fit:
    y[r_p1 & r_p2] = polyfunction(x_p1, params) * deltafunction(p1)  +  mass_ratio * polyfunction(x_p2, params) * deltafunction(p2)
    
    Args:
        dset (pandas DataFrame): Input data, only one main cycle at a time!
        y (str, optional): Column name of y data. Defaults to 'nu_c'.
        yerr (str, optional): Column name of yerr data. Defaults to 'dnu_c'.
        groupbys (list, optional): Column names for grouping the data. Defaults to ['cycle', 'position'].
        identifier (str, optional): Column name of the identifier which marks the different data origings.
        ident_types (list, optional): The two values we want to ratio for this identifier. Typicall the trap integers. The first is the
                                      numerator the second the denominator of the resulting ratio. Defaults to [2, 3].
        keep_columns (list, optional): Columns names which should be somehow transfered to the results dataset. Defaults to ['mcycle', 'time', 'ion'].
    
    Returns:
        [pandas DataFrame]: Resulting dataset.
    """
    if not isinstance(y, list) or not isinstance(yerr, list):
        print("Wrong parameter type, please check doc string or check default values to match the right parameter types!")
        raise ValueError

    # prepare the results dataframe
    columns = keep_columns.copy()
    columns.extend(groupbys)
    for ycol in y:
        columns.extend(["ratio_"+ycol, "dratio_"+ycol])
    columns.extend([identifier+"_numer", identifier+"_denom"])
    for addident in additional_identifiers:
        columns.extend([addident+"_numer", addident+"_denom"])
    columns.extend(mean_columns)

    df_results = pd.DataFrame(columns=columns)

    for grpname, grp in dset.groupby(groupbys):
        if len(grp) < 2:
            print("not enough data in this group", groupbys, grpname)
            continue
        if len(grp[identifier].unique()) < 2:
            print("only the one ident_type in this gourp", groupbys, grpname)
            continue

        try:
            masked = grp["masked"].any()
            if masked:
                #print("masked", grpname)
                continue
        except:
            pass # no masked data here, so what?
            #print('something went wrong looking for masked data tags')
            #raise

        num = grp[ grp[identifier]==ident_types[0] ]
        den = grp[ grp[identifier]==ident_types[1] ]
        #display(num)
        #display(den)
        #print(num['epoch'].to_numpy()-den['epoch'].to_numpy())

        ratios, dratios = [], []

        for ycol, yerrcol in zip(y, yerr):
            ynum = float( num[ycol] )
            dynum = float( num[yerrcol] )
            yden = float( den[ycol] )
            dyden = float( den[yerrcol] )

            ratio = ynum/yden
            ratios.append( ratio )
            dratios.append( ratio*np.sqrt( (dynum/ynum)**2 + (dyden/yden)**2 ) )

        data = []
        for kc in keep_columns:
            val = grp[kc].unique()[0]
            #if len(val) > 1:
            #    print("to many different values for column", kc, val, 'taking the first...')
            #    val = val[0]
            data.append(val)
        for col in groupbys:
            data.append(num[col].values[0])
        for ycol, ratio, dratio in zip(y, ratios, dratios):
            data.extend([ratio, dratio])
        data.extend([ident_types[0], ident_types[1]])
        for ai in additional_identifiers:
            ainum = num[ai].values[0]
            aiden = den[ai].values[0]
            data.extend([ainum, aiden])
        for col in mean_columns:
            timecol = False
            if 'datetime' in str(num[col].dtype):
                timecol = True
                vnum = (num[col].astype('int64')//1e9).values[0]
                vden = (den[col].astype('int64')//1e9).values[0]
            else:
                vnum = num[col].values[0]
                vden = den[col].values[0]
            mean = (vnum + vden)/2
            if timecol:
                mean = pd.to_datetime(mean, unit='s')
            data.append(mean)

        series_data = pd.Series(data=data, index=columns)
        df_results = df_results.append(series_data, ignore_index=True)

    return df_results

def interpolate_values(y1, dy1, y2, dy2, t1, t2, tcenter, non_linear_uncertainty_per_second=0):
    """Interpolates values, see e.g. Roux chapter 9.1 for a description of this in 
    regard to mass measurements in Penning traps.

    Args:
        y1 ([type]): [description]
        dy1 ([type]): [description]
        y2 ([type]): [description]
        dy2 ([type]): [description]
        t1 ([type]): [description]
        t2 ([type]): [description]
        tcenter ([type]): [description]

    Returns:
        [type]: [description]
    """
    deltay = y2 - y1
    deltat12 = t2 - t1
    deltat1c = tcenter - t1
    deltatc2 = t2 -tcenter
    #print(deltat12, deltat1c, deltatc2)
    yinter = y1 + deltay/deltat12*deltat1c

    del1 = (deltatc2 / deltat12)**2 * dy1**2
    del2 = (deltat1c / deltat12)**2 * dy2**2
    dyinter = np.sqrt( del1 + del2 )

    # add non-linear error
    dyinter = np.sqrt( dyinter**2 + (non_linear_uncertainty_per_second*deltat1c)**2 )
    return yinter, dyinter


def interpolate(dset, y=['nu_c'], yerr=['dnu_c'], groupbys=['mcycle', 'trap'], x="time",
                identifier="position", id_reference="position_1", id_interpolate="position_2",
                get_center_columns=['mcycle', 'cycle', 'time', 'time_p'], 
                non_linear_uncertainty_per_second=0, newycol=None, newxcol=None, keepfirstlast=False, keeprow=False):
    """Interpolating two data points of one identifier to the time(x-axis) of another identifier.
    Only works with A-B-A sequence, so only averaged data!

    Args:
        dset ([type]): [description]
        y (list, optional): [description]. Defaults to ['nu_c'].
        yerr (list, optional): [description]. Defaults to ['dnu_c'].
        groupbys (list, optional): [description]. Defaults to ['mcycle', 'trap'].
        x (str, optional): [description]. Defaults to "time".
        identifier (str, optional): [description]. Defaults to "position".
        id_reference (str, optional): [description]. Defaults to "position_1".
        id_interpolate (str, optional): [description]. Defaults to "position_2".
        mean_columns (list, optional): [description]. Defaults to ['time', 'epoch'].
        non_linear_incertainty_per_second (int, optional): [description]. Defaults to 0.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """

    if not isinstance(y, list) or not isinstance(yerr, list):
        print("Wrong parameter type, please check doc string or check default values to match the right parameter types!")
        raise ValueError

    # make epoch axis if needed:
    timecol = False
    float_x = x
    if 'datetime' in str(dset[x].dtype):
        timecol = True
        dset['epoch'] = dset[x].astype('int64')//1e9
        float_x = 'epoch'

    # sort dset and reset index to loop clean over rows
    sortby = groupbys + [float_x]
    dset.sort_values(sortby, inplace=True)
    new_dset = dset.reset_index(drop=True)
    results_df = pd.DataFrame(columns=dset.columns)
    results_df[newycol] = None
    results_df[newxcol] = None
    #display(new_dset['position'].head(10))

    # we do it mcycle by mcycle and trap by trap
    for grpname, grp in new_dset.groupby(groupbys):
        grp_length = len(grp)
        if grp_length < 3:
            #print("not enough data in this group", grpname)
            continue

        #print(groupbys, grpname)

        # we loop over each row, everytime we hit a reference row
        # we add that row and a new row with interpolated data from
        # row-1 and row+1
        for idx, row in grp.iterrows():
            
            if idx == 0 or idx == (grp_length-1):
                if keepfirstlast:
                    results_df = results_df.append(row, ignore_index=True)
                    continue
                else:
                    continue

            # only reference row and only unmasked
            if row[identifier] != id_reference or row["masked"]:
                continue 

            # get row before and after, if existing...
            try:
                before = grp.iloc[idx-1]
                after = grp.iloc[idx+1]
            except IndexError as e:
                print('Index not availabe before after maxidx', idx-1, idx+1, grp_length-1, e)
                continue

            # check if positions are correct:
            if before[identifier] != id_interpolate or after[identifier] != id_interpolate:
                print("not the correct identifier before/after", before[identifier], after[identifier])
                continue

            # check if these rows are masked
            if before["masked"] or after["masked"]:
                continue

            #display(before)
            #display(row)

            # get the times
            t1 = before[float_x]
            tc = row[float_x]
            t2 = after[float_x]

            # prepare new row TODO: copy before or after row?
            if keeprow:
                new_row = row.copy()
            else:
                new_row = before.copy()
            if newxcol is None:
                new_row[float_x] = tc
            else:
                new_row[newxcol] = tc
            for col in get_center_columns:
                try:
                    new_row[col] = row[col]
                except:
                    print("WARNING column", col, "was not in original data")

            # loop over the columns we want to interpolate
            for ycol, yerrcol in zip(y, yerr):
                # get the y's
                y1 = before[ycol]
                y2 = after[ycol]
                yerr1 = before[yerrcol]
                yerr2 = after[yerrcol]
                # interpolate and update row
                new_y, new_yerr = interpolate_values(y1, yerr1, y2, yerr2, t1, t2, tc, non_linear_uncertainty_per_second=non_linear_uncertainty_per_second)
                #print(t1, tc, t2, y1, y2, new_y)
                if newycol is None:
                    new_row[ycol] = new_y
                    new_row[yerrcol] = new_yerr
                else:
                    new_row[newycol] = new_y
                    new_row['d'+newycol] = new_yerr

            
            # add the rows to the new dataset
            if keeprow: 
                results_df = results_df.append(before, ignore_index=True)
            else:
                results_df = results_df.append(row, ignore_index=True)
            results_df = results_df.append(new_row, ignore_index=True)

    # fix the time column if there was one:
    if timecol:
        results_df[x] = pd.to_datetime(results_df[float_x], unit = 's')

    return results_df

def estimate_non_linearity_factor(dset, y=["nu_c"], yerr=["dnu_c"], x="time_p", groupbys=["mcycle"], identifier="cycle"):
    """ Takes a dataset and interpolates every first and third value to the second. Then calculates residuals between interpolated and actual second value and calculates a mean for the whole dataset
    """
    dset["oddity"] = np.where(dset[identifier] % 2 != 0, 'odd', 'even') # this is needed for the interpolation function as an identifier

    interpolated_data = interpolate(dset, y=y, yerr=yerr, groupbys=["mcycle"], x=x,
                identifier="oddity", id_reference="even", id_interpolate="odd",
                non_linear_uncertainty_per_second=0)
    
    time_diffs = dset.epoch.to_numpy()[:-1] - dset.epoch.to_numpy()[1:]
    time_per_cycle = np.median(time_diffs)
    #print("time steps median/mean:", time_per_cycle, np.mean(time_diffs))

    residuals = []
    for gname, grp in interpolated_data.groupby([identifier]):
        try:
            odd = float(grp[ grp['oddity']=='odd'][y].iloc[0])
            even = float(grp[ grp['oddity']=='even'][y].iloc[0])
            diff = odd - even
            #print(odd, even, diff)
        except:
            print(gname, "not complete, skipped")
            continue
        residuals.append(diff)

    residuals = np.asarray(residuals)
    #print(residuals)

    rms_residuals = (np.sqrt(np.sum(residuals**2))/len(residuals))
    std_residuals = np.std(residuals)
    
    #print(rms_residuals, std_residuals, time_per_cycle)

    rms_residuals_per_second = rms_residuals/time_per_cycle
    std_residuals_per_second = std_residuals/time_per_cycle
    return rms_residuals_per_second, std_residuals_per_second, time_per_cycle

def naive_cancellation_ratio(dset, y='ratio_nu_c', yerr='dratio_nu_c', groupbys=['cycle'], identifier="position", ident_types=["position_2","position_1"],
                             keep_columns=['mcycle', 'time', 'ion_numer', 'ion_denom']):
    
    # prepare the results dataframe
    columns = keep_columns.copy()
    columns.append(groupbys)
    columns.append(["R", "dR"])
    columns.append([identifier+"_A", identifier+"_B"])
    df_results = pd.DataFrame()

    for grpname, grp in df_results.groupby(group_by):

        try:
            masked = grp["masked"].any()
            if masked:
                continue
        except:
            print('something went wrong looking for masked data tags')
            raise

        yA = float( grp[ grp[identifier]==ident_types[0] ][y] )
        dyA = float( grp[ grp[identifier]==ident_types[0] ][yerr] )
        yB = float( grp[ grp[identifier]==ident_types[1] ][y] )
        dyB = float( grp[ grp[identifier]==ident_types[1] ][yerr] )

        R = np.sqrt(yA*yB)
        dR = R/2*np.sqrt( (dyA/yA)**2 + (dyB/yB)**2 )

        data = []
        for kc in keep_columns:
            val = grp[kc].unique()
            if len(val) > 1:
                print("to many different values for column", kc, val, 'taking the first...')
                val = val.iloc[0]
            data.append(val)
        data.append([grpname[0], R, dR, ident_types[0], ident_types[1]])

        series_data = pd.Series(data=data, index=columns)
        df_results = df_results.append(series_data, ignore_index=True)

    return df_results
