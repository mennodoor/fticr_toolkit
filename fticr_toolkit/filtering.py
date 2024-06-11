import numpy as np
from scipy import stats
import pandas as pd
from fticr_toolkit import statistics
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

def minmax_value(df, val, min_val = 1, max_val = 1e9):
    """
    filters by min max boundaries
    """
    pd.options.mode.chained_assignment = None  # default='warn'

    df = df.reset_index(drop=True)
    
    counter = 0

    for i in range(len(df)):

        this_value = df.iloc[i][val]
        if (this_value < min_val) or (this_value > max_val) and not df.loc[i, 'masked']:
            df.loc[i, 'masked'] = True
            counter += 1

    if counter: print("filtered min max", val, counter)
    return df

def mode_filter(df, val, round_decimals = 1, value_multiplicator = 0.1):
    """
    Calculates the mode of an array (*factor, rounded) and filters all values not the
    same (*factor, rounded) as the mode.
    """
    pd.options.mode.chained_assignment = None  # default='warn'

    vals = df[val].to_numpy() * float(value_multiplicator)
    vals = np.round(vals, round_decimals)
    mode = stats.mode(vals)[0]
    #print(mode)

    counter = 0

    for i, value in enumerate(vals):
        if value != mode:
            df.loc[i, 'masked'] = True
            counter += 1

    if counter: print("filtered mode", val, counter)
    return df

def nan_filter(df, columns):
    """
    Maskes rows where one of the given columns contains a nan value.
    """
    pd.options.mode.chained_assignment = None  # default='warn'

    masked = np.zeros(len(df), dtype=bool)
    for col in columns:
        masked = np.logical_or( masked, df[col].isnull().to_numpy() )

    df['masked'] = np.logical_or( masked, df['masked'].to_numpy() )
    counter = np.sum(masked)
    if counter: print("filtered nans", columns, counter)
    return df

def three_sigma(df, val, err=None, times_sigma=3, undrift_xcolumn=None, manual_std=None, max_std=None, around="mean", undrift_order=1, show=False):
    """
    Calculates the mean and sigma of the column val and marks all values 3 sigma away from the mean
    as masked True.

    undrift_xcolumn:
    if this column name is given (as a string), the name column will be used as x-values for an attempt 
    to do a linear fit and remove the drift from the data in order to calculate a proper 3 sigma deviation 
    from the expected value.
    """
    pd.options.mode.chained_assignment = None  # default='warn'

    #df = df.reset_index(drop=True)
    if len(df.index) <= 2:
        print("warning, dataset too small", len(df.index))
        return df

    val_array = df[val].to_numpy()
    if err is None:
        err_array = np.zeros(len(val_array))
    else:
        err_array = df[err].to_numpy()
    if undrift_xcolumn is not None:

        xSeries = df[undrift_xcolumn].copy()

        if "time" in undrift_xcolumn:
            first_date = xSeries.iloc[0]
            xSeries -= first_date
            x = xSeries.dt.total_seconds().to_numpy()
        else:
            x = xSeries.to_numpy()

        x -= x[0] # 'normalize'
        y = val_array

        nan_idx = np.argwhere(np.isnan(y))
        #print(nan_idx)
        xfit = np.delete(x, nan_idx)
        yfit = np.delete(y, nan_idx)
        coef = np.polyfit(xfit,yfit,undrift_order)
        #print(coef)
        poly1d_fn = np.poly1d(coef) 

        val_array = val_array - poly1d_fn(x) # remove drift 
        #val_array += poly1d_fn(x[0]) # add offset again

    #print(val_array)

    #mean, err_in, err_out, chi2red = statistics.complete_mean_and_error(val_array, err_array)
    #mean, err_in = statistics.mean(val_array)
    if isinstance(around, float):
        mean = around
        err_in == np.nanstd(val_array)
    elif around == "mean":
        mean, err_in = np.nanmean(val_array), np.nanstd(val_array)
    elif around == "median":
        mean, err_in = np.nanmedian(val_array), np.nanstd(val_array)

    #print(mean, err_in, manual_std)
    if manual_std is not None:
        err_in = manual_std

    if max_std is not None and err_in > max_std:
        err_in = max_std

    #print(mean, err_in, err_out)
    counter = 0
    idxe = []

    #for i in range(len(df)):
    for dfidx, i in zip(df.index.to_numpy(), range(len(df))):

        #this_value = df.iloc[i][val]
        this_value = val_array[i]
        #print(this_value)
        diff = abs(this_value-mean)
        #print(val, i, diff, mean)
        #if (diff > 3*float(err_in) and diff > 3*float(err_out)):
        if (diff > times_sigma*float(err_in)) and not df.loc[dfidx, 'masked']:
            #print(3*float(err_in), 3*float(err_out), diff)
            #print(3*float(err_in), diff)
            df.at[dfidx, 'masked'] = True
            print("filtered 1")
            idxe.append(dfidx)
            counter += 1
    
    if show:
        length = len(val_array)
        x = list(range(1, length+1))
        y_upper = [mean + times_sigma*err_in]*length
        y_lower = [mean - times_sigma*err_in]*length
        fig = go.Figure([
            go.Scatter(
                x=x,
                y=val_array,
                error_y=dict(
                    type='data', # value of error bar given in data coordinates
                    array=err_array,
                    visible=True),
                line=dict(color='rgb(0,100,80)'),
                mode='lines'
            ),
            go.Scatter(
                x=x+x[::-1], # x, then x reversed
                y=y_upper+y_lower[::-1], # upper, then lower reversed
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True 
            )
        ])
        fig.show()

    if counter: 
        print("filtered 3sigma", val, counter)

    return df

def sigma_size(df, err, manual_std=None, show=False):
    """
    Calculates the mean and sigma of the column val and marks all values 3 sigma away from the mean
    as masked True.
    """
    pd.options.mode.chained_assignment = None  # default='warn'
    if len(df.index) <= 2:
        print("warning, dataset too small", len(df.index))
        return df
        
    # TODO: rather not drop the index if possible
    df = df.reset_index(drop=True)

    err_array = df[err].to_numpy()

    mean, err_in, err_out, chi2red = statistics.complete_mean_and_error(err_array)
    mean = err_array.mean()
    mean = np.nanmedian(err_array)
    if manual_std is not None:
        mean = manual_std
    print(mean, err_in, err_out)
    counter = 0

    for i in range(len(df)):

        this_value = df.iloc[i][err]
        if (this_value > 3*float(mean)) and not df.loc[i, 'masked']:
            print("filtered 1")
            df.loc[i, 'masked'] = True
            counter += 1

    if show:
        val_array = df[err]
        length = len(val_array)
        x = list(range(1, length+1))
        y_upper = [3*mean]*length
        fig = go.Figure([
            go.Scatter(
                x=x,
                y=val_array,
                line=dict(color='rgb(0,100,80)'),
                mode='lines'
            ),
            go.Scatter(
                x=x+x[::-1], # x, then x reversed
                y=y_upper+[0]*length, # upper, then lower reversed
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True 
            )
        ])
        fig.show()

    if counter: print("filtered bigsigma", err, counter)
    return df

def ratio_filter(df, column="R", dcolumn="dR", minmax=None):
    grp = df.copy()

    if not column in grp:
        return grp
    # filter nan values (sometimes happens when an axial spectrum is missing)
    grp = nan_filter(grp, columns=[column])
    
    if not dcolumn in grp:
        return grp
    # apply autofilter 3-sigma condition: calc mean of values and std
    # if value is outside of mean+-3*std, it is masked
    grp = three_sigma(grp, val=column, err=dcolumn, show=False)
    #grp = mode_filter(grp, val=column)

    # apply autofilter sigma-size: if sigma of value is 3 time bigger
    # then mean sigma, it is masked
    grp = sigma_size(grp, err=dcolumn)

    if minmax is not None:
        grp = minmax_value(grp, column, min_val = minmax[0], max_val = minmax[1])

    return grp
