import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import plotly.graph_objs as go

"""

This contains functions to calculate the ratio of two values. Either time-independent ("naive") or time dependent (interpolation if needed). The two values
can be e.g. cyclotron frequencies (single trap analysis) or already ratios of frequencies from different traps (cancellation method).

"""

np.set_printoptions(precision=12)

def calc_trap_ratios(dset, y='nu_c', yerr='dnu_c', groupbys=['cycle', 'position'], identifier="trap", ident_types=[2,3], keep_columns=['mcycle', 'time', 'ion'],
                     additional_identifiers=['ion']):
    """Calcultes the ratio of nu_c frequencies (or the ratio of any other column) for the traps given
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

    # prepare the results dataframe
    columns = keep_columns.copy()
    columns.append(groupbys)
    columns.append(["ratio_"+y, "dratio_"+y])
    columns.append([identifier+"_numer", identifier+"_denom"])
    for addident in additional_identifiers:
        columns.append([addident+"_numer", addident+"_denom"])
    df_results = pd.DataFrame()

    for grpname, grp in df_results.groupby(group_by):

        try:
            masked = grp["masked"].any()
            if masked:
                continue
        except:
            print('something went wrong looking for masked data tags')
            raise

        ynum = float( grp[ grp[identifier]==ident_types[0] ][y] )
        dynum = float( grp[ grp[identifier]==ident_types[0] ][yerr] )
        yden = float( grp[ grp[identifier]==ident_types[1] ][y] )
        dyden = float( grp[ grp[identifier]==ident_types[1] ][yerr] )

        ratio = ynum/yden
        dratio = ratio*np.sqrt( (dynum/ynum)**2 + (dyden/yden)**2 )

        data = []
        for kc in keep_columns:
            val = grp[kc].unique()
            if len(val) > 1:
                print("to many different values for column", kc, val, 'taking the first...')
                val = val.iloc[0]
            data.append(val)
        data.append([grpname[0], grpname[1], ratio, dratio, ident_types[0], ident_types[1]])
        for ai in additional_identifiers:
            ainum = grp[ grp[identifier]==ident_types[0] ][ai]
            aiden = grp[ grp[identifier]==ident_types[1] ][ai]
            data.append(ainum, aiden)

        series_data = pd.Series(data=data, index=columns)
        df_results = df_results.append(series_data, ignore_index=True)

    return df_results

def interpolate(dset, y='nu_c', yerr='dnu_c', ident_column="position", x_fixed="position_1", x_interpolate="position_2",
                keep_columns=['mcycle', 'time', 'ion'], x="time"):
    
    dset.sort_values(x)
    for idx, row in dset.iterrows():
        if idx == 0: continue
        fixed_ident = row.x_fixed
        be_ident = row.x_fixed
    

def naive_ratio(dset, y='nu_c', yerr='dnu_c', groupbys=['cycle'], identifier="position", ident_types=["position_2","position_1"],
                keep_columns=['mcycle', 'time', 'ion_numer', 'ion_denom'], half_group=False ):
    
    
    # prepare the results dataframe
    columns = keep_columns.copy()
    columns.append(groupbys)
    columns.append(["R", "dR"])
    columns.append([identifier+"_numer", identifier+"_denom"])
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

