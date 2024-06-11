from enum import auto
from tokenize import group
import plotly.graph_objs as go
import plotly.offline as po
import plotly.express as px

from plotly.subplots import make_subplots
import qgrid
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib as mpl

import pandas as pd
import numpy as np
import allantools

from ipywidgets import interactive, HBox, VBox
import fticr_toolkit.statistics as statistics

def compare_dset_columns(df, x=["time_p", "time_p"], y=["nu_c", "nu_c_sb"], yerr=["dnu_c", "dnu_c_sb"], facet_col="trap", facet_row="position", title=""):

    row_values = df[facet_row].unique()
    col_values = df[facet_col].unique()

    titles = []
    for ridx, rval in enumerate(row_values):
        for cidx, cval in enumerate(col_values):
            string = facet_row+" : "+str(rval)+", "+facet_col+" : "+str(cval)
            titles.append(string)

    fig = make_subplots(rows=len(row_values), cols=len(col_values), subplot_titles=titles)
    
    for ridx, rval in enumerate(row_values):

        for cidx, cval in enumerate(col_values):

            for xcol, ycol, yerrcol in zip(x, y, yerr):

                subset = df[ (df[facet_row]==rval) & (df[facet_col]==cval) ]
                try:
                    error_data = dict(
                        type='data', # value of error bar given in data coordinates
                        array=subset[yerrcol],
                        visible=True)
                except:
                    error_data = {}
                fig.add_trace(
                    go.Scatter(x=subset[xcol], y=subset[ycol], error_y=error_data, name=ycol, mode='markers'), row=ridx+1, col=cidx+1,
                )

    fig.update_layout(title_text=title)
    fig.show()
    return fig


def filter_plot(df, groupby=["mcycle", "trap", "position"], ydata = ["nu_z", "nu_p"], xdata=["cycle"]):

    last_kwargs = {}
    xaxis = xdata[0]
    yaxis = ydata[0]
    
    # how to get a subset easy :)
    def get_subset(**kwargs):
        subset = df
        for key, value in kwargs.items():
            subset = subset[subset[key]==value]
            last_kwargs[key] = value
            
        return subset, kwargs
    
    # just supply an init data plot
    firsts = []
    for grp in groupby:
        firsts.append(df[grp].unique()[0])

    subset = df
    for first, grp in zip(firsts, groupby):
        subset = subset[subset[groupby] == first]
        last_kwargs[grp] = first

    x = subset[xdata[0]]
    y = subset[ydata[0]]
    try:
        yerr = subset['d'+ydata[0]]
    except:
        yerr = np.zeros(len(y))
    error_y = dict(
            type='data', # value of error bar given in data coordinates
            array=yerr,
            visible=True)
    
    f = go.FigureWidget([go.Scatter(x=x, y=y, error_y=error_y, mode = 'markers')])
    f.layout.hovermode = 'closest'

    def update_layout(masked, xaxis="", yaxis=""):
        scatter = f.data[0]
    
        colors = np.asarray(['#a3a7e4'] * len(masked))
        sizes = np.asarray([12] * len(masked))
        opacity = np.asarray([1] * len(masked))
        np.place(colors, masked, '#bae2be')
        np.place(sizes, masked, 8)

        with f.batch_update():
            if xaxis!="": f.layout.xaxis.title = xaxis
            if yaxis!="": f.layout.yaxis.title = yaxis
            scatter.marker.color = list(colors)
            scatter.marker.size = list(sizes)
            scatter.marker.opacity = list(opacity)

    # init data
    scatter = f.data[0]
    scatter.x = x
    scatter.y = y
    scatter.error_y = dict(
        type='data', # value of error bar given in data coordinates
        array=yerr,
        visible=True)
    
    masked = subset["masked"].to_numpy()
    update_layout(masked, xdata[0], ydata[0])
    
    def update_group(**kwargs):
        global last_kwargs, xaxis, yaxis
        
        # first get and remove the column/axis info
        xaxis = kwargs.pop("xdata")
        yaxis = kwargs.pop("ydata")
        subset, subkwargs = get_subset(**kwargs)
        last_kwargs = subkwargs
        
        scatter = f.data[0]

        xdata = subset[xaxis]
        scatter.x = xdata
        scatter.y = subset[yaxis]
        try:
            yerr = subset['d'+yaxis]
        except:
            yerr = np.zeros(len(y))
        scatter.error_y = dict(
            type='data', # value of error bar given in data coordinates
            array=yerr,
            visible=True)

        masked = subset["masked"].to_numpy()
        update_layout(masked, xaxis, yaxis)
    
    # interactive controls for...
    kwargs = {}
    for grp in groupby:
        kwargs[grp] = df[grp].unique()
    kwargs["ydata"] = ydata   
    kwargs["xdata"] = xdata   

    axis_dropdowns = interactive(update_group, **kwargs)

    # interactive click to mask / unmask
    def update_mask_click(trace, points, selector):
        
        # get the points position in the original df        
        subset, kw = get_subset(**last_kwargs)
        index = subset[subset[xaxis] == points.xs[0]].index[0]
        
        # get the current mask value and invert
        mask_bool = not df.at[index, "masked"]
        #print(index, mask_bool)
        df.at[index, "masked"] = mask_bool
        
        subset, kw = get_subset(**last_kwargs)
        masked = subset["masked"].to_numpy()
        update_layout(masked)
                
    scatter.on_click(update_mask_click)
    
    # interactive select to mask / unmask
    def update_mask_select(trace, points, selector):

        # get the points position in the original df        
        subset, kw = get_subset(**last_kwargs)
        index_list = list(subset[subset[xaxis].isin(points.xs[:])].index[:])
        
        # get the current mask value and invert
        for index in index_list:    
            mask_bool = not df.at[index, "masked"]
            #print(index, mask_bool)
            df.at[index, "masked"] = mask_bool
        
        subset, kw = get_subset(**last_kwargs)
        masked = subset["masked"].to_numpy()
        update_layout(masked)
                
    scatter.on_selection(update_mask_select)

    # Put everything together
    return VBox((HBox(axis_dropdowns.children),f))


def manual_grouping_plot(df, groupby=["mcycle", "trap"], sets="position", y="nu_c", yerr="dnu_c", x="time"):
    """
    Interactive manual grouping plot.

    IMPORTANT!!! The input df has to be sorted (with index reset!!!) by parameters given above [groupby, x] !!!
    e.g.
    df.sort(['mcycle', 'trap', 'time'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    Args:
        df (pandas DataFrame): Input DataFrame, HAS TO BE SORTED!!! (see above)
        groupby (list, optional): Choosing what to look at with these descrimators. Defaults to ["mcycle", "trap"].
        sets (str, optional): Choose what is plotted together with this column name. Defaults to "position".
        y (str, optional): Defaults to "nu_c".
        yerr (str, optional): Defaults to "dnu_c".
        x (str, optional): Defaults to "time".

    Returns:
        ipywidget VBox
    """

    last_kwargs = {}
    # This is needed because plotly truncates datetime to full milliseconds and
    # then the x value of a clicked on point is not the same as in the dataset.
    # To prevent this, we truncate the dateset first to seconds, ideally obviously
    # to milliseconds, but I did not find the appropriate command and seconds is fine
    # for this application here :)
    try:
        df[x] = df[x].dt.floor('S')
    except:
        pass

    # This function is used to get the data for the drop down menu, choosing the parameters of groupby
    def get_subset(**kwargs):
        nonlocal last_kwargs

        subset = df # take the original dataframe
        for key, value in kwargs.items():
            subset = subset[subset[key]==value] # pick the right parts
            last_kwargs[key] = value # update the last kwargs list
            
        return subset

    # init the figure with one line per data set (sets variable)
    lines = []
    set_idents = df[sets].unique().tolist()
    for ident in set_idents:
        #print(ident)
        lines.append( go.Scatter(x=[], y=[], error_y={}, mode='markers', name=ident) )

    f = go.FigureWidget(lines)
    f.layout.title = 'Manual Grouping - Choose data via drop down, y values are shifted to match (roughly). Change border points by clicking (going lower in doubt) or use selection and the group move drop down.'
    f.layout.hovermode = 'x'

    # this modifies the points color and size to show what group they belong, its used
    # either when the data is change to mark the groups initially or when the groups are
    # changed (so called inside update_group or update_mask_click)
    def update_layout(ident, groups, cycles, y_real):
        idx = set_idents.index(ident)
        #print('update', ident)
        scatter = f.data[idx]

        hovers = []
        for grp, cyc, yval in zip(groups, cycles, y_real):
            hovers.append('cycle : '+str(cyc)+'<br>group : '+str(int(grp))+'<br>real y = '+str(yval))
    
        mask = (groups%2 == 0)
        colors = np.asarray(['#054479'] * len(mask)) # blue
        np.place(colors, mask, '#cc5f24') # red

        if idx%2==0:
            sizes = np.asarray([20] * len(mask)) # big
            #np.place(sizes, mask, 8) # small
        else:
            sizes = np.asarray([12] * len(mask)) # small

        with f.batch_update():
            scatter.marker.color = list(colors)
            scatter.marker.size = list(sizes)
            scatter.hovertext = hovers

    move = 1 # for group selection feature
    # this function updates the data in the plot (its the callback for the dropdown widgets)
    def update_group(**kwargs):
        nonlocal move
        try:
            move = kwargs.pop("group_move")
        except KeyError:
            pass

        subset = get_subset(**kwargs)
        first_mean = None

        for idx, ident in enumerate(set_idents):
            scatter = f.data[idx]

            subset2 = subset[ subset[sets] == ident ]

            # adjusting y to have all matched
            mean = subset2[y].mean()
            if idx == 0:
                first_mean = mean
            multiply = first_mean/mean

            scatter.x = subset2[x]
            scatter.y = subset2[y]*multiply
            scatter.error_y = dict(
                type='data', # value of error bar given in data coordinates
                array=subset2[yerr].to_numpy(),
                visible=True)
                
            groups = subset2["group"].to_numpy()
            cycles = subset2["cycle"].to_numpy()
            yhere = subset2[y].to_numpy()
            update_layout(ident, groups, cycles, yhere)
    
    # interactive controls for choosnig groupy columns, e.g. mcycle and trap
    kwargs = {}
    for grp in groupby:
        kwargs[grp] = df[grp].unique()
    kwargs["group_move"] = [-1, +1] # needed for selection group change
    axis_dropdowns = interactive(update_group, **kwargs)
    #print(kwargs)

    # interactive click to move group up down on edge point
    def update_mask_click(trace, points, selector):
        this_ident = trace.name
        
        # get the subset and subident of this trace from the original df
        subset = get_subset(**last_kwargs)
        subset = subset[ subset[sets] == this_ident ]

        try:
            # get the current group value and the ones before and after
            index = subset[subset[x] == points.xs[0]].index[0]
            group_num = df.at[index, "group"]
            cycle_num = df.at[index, "cycle"]
            #print(index, group_num, cycle_num)
        except:
            return 0

        updown = 0
        try:
            lower = df.at[index-1, "group"] - group_num # group goes - 1
            if lower < 0:
                updown = -1
            #print('higher', higher, updown)
        except:
            pass
        try: # NOTE: LOWER ALWAYS WINS!!
            if updown==0:
                higher = df.at[index+1, "group"] - group_num # group goes + 1
                if higher > 0:
                    updown = 1
                #print('lower', lower, updown)
        except:
            pass

        df.at[index, "group"] = group_num + updown
        
        subset = get_subset(**last_kwargs)
        subset = subset[ subset[sets] == this_ident ]
        groups = subset["group"].to_numpy()
        cycles = subset["cycle"].to_numpy()
        #print(index, this_ident, cycle_num, group_num, adder)
        yhere = subset[y].to_numpy()
        update_layout(this_ident, groups, cycles, yhere)
    
    for scatter in f.data:
        scatter.on_click(update_mask_click)
    
    # interactive select to mask / unmask
    def update_group_select(trace, points, selector):
        this_ident = trace.name
        #print(this_ident)

        # get the subset and subident of this trace from the original df
        subset = get_subset(**last_kwargs)
        subset = subset[ subset[sets] == this_ident ]

        try:
            # get the current group value and the ones before and after
            #print(subset[x].isin(points.xs[:]))
            index_list = list(subset[subset[x].isin(points.xs[:])].index[:])
            #print(index_list)
            df.loc[index_list, "group"] += move
        except:
            #raise
            return 0

        subset = get_subset(**last_kwargs)
        subset = subset[ subset[sets] == this_ident ]
        groups = subset["group"].to_numpy()
        cycles = subset["cycle"].to_numpy()
        #print(index, this_ident, cycle_num, group_num, adder)
        yhere = subset[y].to_numpy()
        update_layout(this_ident, groups, cycles, yhere)

    
    for scatter in f.data:
        scatter.on_selection(update_group_select)


    # Put everything together
    return VBox((HBox(axis_dropdowns.children),f))



def filter_grid(df, col_opts={}, col_defs={}, grid_options={}):
    """
    Returns a qgrid object for visualization of datasets and manipulation of the column "masked".
    The only thing thats happening here is basically a little bit of default formating.
    """

    lcol_opts = {"editable": False, "toolTip": "dont mess with data!"}
    lcol_opts.update(col_opts)

    lcol_defs = { 
        'masked': { 'editable': True , 'toolTip': 'editable'}, 
        'position': { 'width': 80 },
        'trap': { 'width': 40 },
        'nu_z': { 'width': 100 },
        'nu_c': { 'width': 100 },
        'nu_p': { 'width': 100 },
        'ion': { 'width': 100 },
        'time': { 'width': 150 },
        'time_p': { 'width': 150 }
    }
    lcol_defs.update(col_defs)

    lgrid_options = {'forceFitColumns': False, 'defaultColumnWidth': 50}
    lgrid_options.update(grid_options)

    return qgrid.show_grid(df, column_options=lcol_opts, column_definitions=lcol_defs, grid_options=lgrid_options)


def ratio_plot(df, title, start = None, stop = None, student=False, showlegend=False, remove_masked=True, time_column="time_p"):

    if remove_masked:
        df = df[ df["masked"]==False ]

    if time_column not in df.columns:
        if "time" in df.columns:
            time_column = "time"
        elif "time_p" in df.columns:
            time_column = "time_p"

    if start is not None:
        df = df[df["time"]>=start]

    if stop is not None:
        df = df[df["time"]<=stop]

    # Build figure
    fig = go.FigureWidget()

    scatter_plots = {}

    def calc_mean(student=False):

        y = df["R"].to_numpy()
        yerr = df["dR"].to_numpy()
        masked = df["masked"].to_numpy()
        R4mean = y[~masked]
        dR4mean = yerr[~masked]
        
        R, err_in, err_out, chi2red = statistics.complete_mean_and_error(R4mean, dvalue = dR4mean, student = student)
        return R, err_in, err_out, chi2red
            
    for grp_name, grp in df.groupby(["measurement"]):
        x = grp["time"].to_numpy()
        y = grp["R"].to_numpy()
        yerr = grp["dR"].to_numpy()
        if grp["masked"].dtypes.name != 'bool':
            grp['masked'] = grp['masked'].astype('bool')
        masked = grp["masked"].to_numpy()

        opacities = np.ones(len(x))
        opacities[masked] = 0.2

        grp_string = grp_name+"\nmasked="
        hoverstrings = [grp_string+str(val) for val in masked]

        scatter = go.Scatter(
                mode='markers',
                x=x,
                y=y,
                error_y=dict(
                    type='data', # value of error bar given in data coordinates
                    array=yerr,
                    thickness=0.4,
                    visible=True
                ),
                marker=dict(
                    #color='LightSkyBlue',
                    #size=20,
                    opacity=opacities,
                    line=dict(
                        width=1.5
                    )
                ),
                name=grp_name,
                showlegend=True,
                hovertext=masked
            )
        
        scatter_plots[grp_name] = scatter
        #scatter_plots[grp_name].on_selection(selection_fn)

        # Add scatter 
        fig.add_trace(scatter_plots[grp_name])
            
    R, err_in, err_out, chi2red = calc_mean(student=student)
    label = "\t 1-R = %.12e +- %.3e (inner) +- %.3e (outer), chi2red %.2f" % (1-R, err_in, err_out, chi2red)
    print(title+label)

    # mean
    x = [df["time"].min(), df["time"].max()]
    x_rev = x[::-1]
    fig.add_trace(go.Scatter(
                        x=x,
                        y=[R, R],
                        error_y=dict(
                            type='data', # value of error bar given in data coordinates
                            array=[err_in, err_in],
                            thickness=0.5,
                            visible=True
                        ),
                        mode="lines+text",
                        name="mean R",
                        text=[label, ""],
                        textposition="bottom right"
                    ))

    # error in
    fig.add_trace(go.Scatter(
        x=x+x_rev,
        y=[R+err_in, R+err_in, R-err_in, R-err_in],
        fill='tozerox',
        fillcolor='rgba(0,176,246,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='error in',
        showlegend=False,
        ))

    # error out
    fig.add_trace(go.Scatter(
        x=x+x_rev,
        y=[R+err_out, R+err_out, R-err_out, R-err_out],
        fill='tozerox',
        fillcolor='rgba(0,176,246,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='error in',
        showlegend=False,
        ))

    fig.update_layout(
        title=title+label,
        xaxis_title="time",
        yaxis_title="R",
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#7f7f7f"
        ),
        showlegend=showlegend
    )
        
    fig.layout.hovermode = 'closest'

    return fig


def ratio_filter_plot(df, title, start = None, stop = None, student=False, time_column="time_p", remove_masked=True):

    if remove_masked:
        df = df[ df["masked"]==False ]

    if time_column not in df.columns:
        if "time" in df.columns:
            time_column = "time"
        elif "time_p" in df.columns:
            time_column = "time_p"

    if start is not None:
        df = df[df[time_column]>=start]

    if stop is not None:
        df = df[df[time_column]<=stop]

    # Build figure
    fig = go.FigureWidget()

    scatter_plots = {}

    def calc_mean(student=False):

        y = df["R"].to_numpy()
        yerr = df["dR"].to_numpy()
        masked = df["masked"].to_numpy()
        R4mean = y[~masked]
        dR4mean = yerr[~masked]
        
        #print(R4mean, dR4mean)
        R, err_in, err_out, chi2red = statistics.complete_mean_and_error(R4mean, dvalue = dR4mean, student = student)
        #print( R, err_in, err_out, chi2red)
        return R, err_in, err_out, chi2red

    def selection_fn(trace, points, selector):
        meas_name = trace.name
        #print(meas_name)
            
        # get the points position in the original df        
        subset = df[df["measurement"]==meas_name]
        index_list = list(subset[subset[time_column].isin(points.xs[:])].index[:])
        #print(index_list, points.xs[:])
        # get the current mask value and invert
        for index in index_list:    
            mask_bool = not df.at[index, "masked"]
            df.at[index, "masked"] = mask_bool

        masked = subset["masked"].to_numpy()
        opacities = np.ones(len(masked))
        opacities[masked] = 0.2

        R, err_in, err_out, chi2red = calc_mean(student=student)
        
        with fig.batch_update():
            scatter_plots[meas_name].marker.opacity = list(opacities)
            scatter_plots[meas_name].hovertext = masked
            fig.data[-1].y=[R, R]
            fig.data[-1].error_y=dict(
                            type='data', # value of error bar given in data coordinates
                            array=[err_in, err_in],
                            thickness=0.5,
                            visible=True
                        )
            label = "\t 1-R = %.12e +- %.3e (inner) +- %.3e (outer), chi2red %.2f" % (1-R, err_in, err_out, chi2red)
            #print(title+label)

            fig.data[-1].text=[label, ""]
            fig.update_layout(title=title+label)

            
    for grp_name, grp in df.groupby(["measurement"]):
        x = grp[time_column].to_numpy()
        y = grp["R"].to_numpy()
        yerr = grp["dR"].to_numpy()
        if grp["masked"].dtypes.name != 'bool':
            grp['masked'] = grp['masked'].astype('bool')
        masked = grp["masked"].to_numpy()

        opacities = np.ones(len(x))
        opacities[masked] = 0.2

        grp_string = grp_name+"\nmasked="
        hoverstrings = [grp_string+str(val) for val in masked]

        scatter = go.Scatter(
                mode='markers',
                x=x,
                y=y,
                error_y=dict(
                    type='data', # value of error bar given in data coordinates
                    array=yerr,
                    thickness=0.4,
                    visible=True
                ),
                marker=dict(
                    #color='LightSkyBlue',
                    #size=20,
                    opacity=opacities,
                    line=dict(
                        width=1.5
                    )
                ),
                name=grp_name,
                showlegend=True,
                hovertext=hoverstrings # masked
            )
        
        scatter_plots[grp_name] = scatter
        #scatter_plots[grp_name].on_selection(selection_fn)

        # Add scatter 
        fig.add_trace(scatter_plots[grp_name])
            
    R, err_in, err_out, chi2red = calc_mean(student=student)
    label = "\t 1-R = %.12e +- %.3e (inner) +- %.3e (outer), chi2red %.2f" % (1-R, err_in, err_out, chi2red)
    print(title+label)

    # mean
    fig.add_trace(go.Scatter(
                        x=[df[time_column].min(), df[time_column].max()],
                        y=[R, R],
                        error_y=dict(
                            type='data', # value of error bar given in data coordinates
                            array=[err_in, err_in],
                            thickness=0.5,
                            visible=True
                        ),
                        mode="lines+text",
                        name="mean R",
                        text=[label, ""],
                        textposition="bottom right"
                    ))

    fig.update_layout(
        title=title+label,
        xaxis_title=time_column,
        yaxis_title="R",
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#7f7f7f"
        )
    )

    for data in fig.data:
        #print(data.name)
        if data.name == "mean R":
            continue
        data.on_selection(selection_fn)
        
    fig.layout.hovermode = 'closest'

    return fig

####
## Part 2 stuff
####

def allanvariance(y, t, plot=False, relative=False, factor=1):
    #t = np.logspace(0, 3, 50)  # tau values from 1 to 1000
    #y = allantools.noise.white(10000)  # Generate some frequency data
    tdiff = (t[1:] - t[:-1]).mean()
    r = 1/tdiff  # sample rate in Hz of the input data
    (t2, ad, ade, adn) = allantools.oadev(y, rate=r, data_type="freq", taus=t)  # Compute the overlapping ADEV
    if relative:
        ad = ad/np.mean(y)
        ade = ade/np.mean(y)
    ad *= factor
    ade *= factor

    if plot:
        #fig = plt.loglog(t2, ad) # Plot the results
        fig = plt.errorbar(t2, ad, ade) # Plot the results
        #ax = plt.gca()
        plt.yscale("log")
        plt.xscale("log")
        plt.grid(which="both")
        plt.show()
    return t2, ad, ade, adn



def allancompare(df_dict, time='time_p', ycol='R', normalize = False, groupby=None):

    fig, ax = plt.subplots(figsize=(25,10))
    ax.set_xscale("log")
    ax.set_yscale("log")

    def add_line(subset, label):
    
        if len(subset) < 4:
            return 
        if ycol not in subset.columns:
            return
        if time not in subset.columns:
            subset[time] = subset['time_p']
        if "trap" not in subset.columns:
            subset["trap"] = 0

        subset = subset.sort_values(time)
        subset['epoch'] = subset[time].astype("int64")//1e9
        subset['seconds'] = subset['epoch'] - subset['epoch'].min()
        t = subset['seconds'].to_numpy()
        R = subset[ycol].to_numpy()
        if normalize:
            R = R/R.mean()
        t2, ad, ade, adn = allanvariance(R, t, plot=False)

        ax.errorbar(t2, ad, ade, label=label)
        #axes[1].loglog(t2, ad)

    for tag, df in sorted(df_dict.items()):
        try:
            if groupby is None:
                add_line(df, tag)
            else:
                for gname, grp in df.groupby(groupby):
                    add_line(grp, tag+" "+gname)

        except:
            raise
            print(tag, 'failed')

    #fig.legend(loc='lower left', ncol=4)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          ncol=4, fancybox=True, shadow=True)
    #fig.tight_layout()
    ax.grid()
    fig.show()
    return fig

def pptprint(*args):
    for arg in args:
        try:
            print( "%.13f" % arg, end="\t")
        except:
            print( arg, end="\t")
    print("")



"""def compare_groupby(df_dict, groupby=["measurement", "mcycle", "trap"] figsize=(8, 30), sort = False, time='time_p', trap_split=False, auto_flip=True):

    for key, df in df_dict.items():
        
        print(key)
        ts = []
        Rs = []
        dRs = []
        for gname, grp in df.groupby(groupby):
            pass
"""


def compare(df_dict, figsize=(8, 30), sort = False, timecol='time_p', groupby=["trap", "mcycle"], auto_flip=True, hover_data=[]):
    
    fig, ax = plt.subplots()
    if groupby is None:
        groupby = "None"

    tags = []
    idx = []
    i = 0
    Rs = []
    drs = []
    pptprint("analysis type ", "R-1     ", "err_in   ", "err_out   ", "chi2red   ", "(out/in) ", "1+(0.477/sqrt(N))", "# of R    ", "mean single err", "R std", "1/R")
    firstR = None
    firsttag = None
    if sort:
        iterable = sorted(df_dict.items())
    else:
        iterable = df_dict.items()

    all_in_one = pd.DataFrame()

    for tag, df in iterable:
        try:
            #print(tag, df.columns)
            if "R" not in df.columns:
                continue
            df['analysis'] = tag
            if timecol not in df.columns:
                df[timecol] = df['time_p']
            if "trap" not in df.columns:
                df["trap"] = 0

            keys = ['mcycle', 'Rminus', 'dR', timecol, 'trap', 'analysis']
            keys.extend(hover_data)
            all_in_one = all_in_one.append(df[keys], ignore_index=True)

            if tag.startswith("step"):
                tag = tag.split("_", 1)[1]
            
            if groupby == "None":
                df[groupby] = ""

            for subtag, thisdf in df.groupby(groupby):
                thistag = tag + " " + str(subtag)

                r_data = thisdf.R.to_numpy()
                dr_data = thisdf.dR.to_numpy()

                if auto_flip and np.mean(r_data)<1:
                    r_data = r_data**(-1)
                    dr_data = r_data**2 * dr_data

                #print(dr_data)
                R, err_in, err_out, chi2red = statistics.complete_mean_and_error(r_data, dvalue = dr_data, student = False)
                if firstR is None:
                    firstR = R
                    firsttag = thistag

                Rflip = 1/R
                dRflip = Rflip**2*max([err_in, err_out])

                pptprint(thistag, R-1, err_in, err_out, chi2red, (err_out/err_in), 1+(0.477/np.sqrt(len(thisdf))), '\t'+str(int(len(thisdf)))+'\t', np.mean(dr_data), np.std(r_data), Rflip, dRflip*1e12)
                #print(R, err_in)
                tags.append(thistag)
                Rs.append(R-1)
                i+=1
                idx.append(i)
                big_err = np.nanmax([err_in, err_out])
                drs.append(big_err)

                ax.errorbar([thistag], [R - firstR], yerr=[big_err], marker="o")
                annotate_string = '     {:0.1e}'.format(R - firstR)
                annotate_string += '\n'
                annotate_string += '  +-{:0.1e}'.format(big_err)
                ax.annotate(annotate_string, (thistag, R - firstR))
        except:
            #raise
            print(tag, 'was not possible')



    fig.set_figheight(figsize[0])
    fig.set_figwidth(figsize[1])

    ax.set_xlabel("analysis type & trap")
    ax.set_ylabel("R - R_"+str(firsttag))

    plt.title('Results comparison')
    plt.show()

    fig = px.line(all_in_one, x=timecol, y="Rminus", error_y="dR", color='analysis', facet_row='trap', hover_data=hover_data)
    fig.show()

    #fig = px.line(all_in_one, x=timecol, y="Rminus", error_y="dR", color='analysis', hover_data=hover_data)
    #fig.show()

    return all_in_one

def compare2(df, measurement):

    ratios = df.R.to_numpy()
    dratios = df.dR.to_numpy()
    R, err_in, err_out, chi2red = statistics.complete_mean_and_error(ratios, dvalue = dratios)
    Rminus = R - 1
    max_err = max(err_in, err_out)
    #print(measurement, "both traps", Rminus, max_err)
    
    for trap in df.trap.unique():
        df3 = df[df["trap"]==trap]
        if trap == 2:
            pass#continue

        #df4 = filtering.ratio_filter(df3, column="R", dcolumn="dR")
        mindex = measurement.split("_")[-1]
        #print(mindex, trap, type(mindex), type(trap))
        # bad_batch = [("3",2), ("4",3), ("6B",3)]
        skip = False
        for bb in bad_batch:
            if mindex == bb[0] and trap == bb[1]:
                skip = True
        if skip:
            #pass
            continue
        
        ratios = df3.R.to_numpy()
        dratios = df3.dR.to_numpy()
        R, err_in, err_out, chi2red = statistics.complete_mean_and_error(ratios, dvalue = dratios)
        Rminus = R - 1
        max_err = max(err_in, err_out)

        print(measurement, "trap", trap, Rminus, max_err)
        plt.errorbar(measurement, Rminus, max_err, label = str(measurement)+ "    trap_" + str(trap), fmt="o" )


def heatmap(data, column_labels, row_labels, title='', cmap=plt.cm.Reds_r, xlabel='poly degree', ylabel='group size'):

    column_labels = column_labels.astype('int')
    row_labels = row_labels.astype('int')
    data = np.asarray(data)

    if np.nanmin(data) < 0:
        data = data - (np.nanmin(data)-1)
        print(np.nanmin(data))

    fig, ax = plt.subplots()

    heatmap = ax.pcolor(data, cmap=cmap, norm=colors.LogNorm(vmin=np.nanmin(data), vmax=np.nanmax(data)),
                        vmin=np.nanmin(data), vmax=np.nanmax(data))
    heatmap.cmap.set_under('black')

    bar = fig.colorbar(heatmap, extend='both')

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


    
def resultsplotA(data, color=(240, 0, 142), colorB=(255, 0, 38), save=None, rcparams={}):

    plt.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams.update(rcparams)

    labelsize=14
    
    if sum(color) > 3:
        color = (color[0]/255, color[1]/255, color[2]/255)
    if sum(colorB) > 3:
        colorB = (colorB[0]/255, colorB[1]/255, colorB[2]/255)
        
    Rmean, din, dout, chi2 = statistics.complete_mean_and_error(data.R, dvalue = data.dR)
    print("Rmean", np.around(Rmean,13), din, dout)
    data["Roff"] = data["R"] - Rmean
    
    f, axises = plt.subplots(2, 3, gridspec_kw={'width_ratios': [4, 4, 1]})
    ax1, ax2, ax13 = axises[0]
    ax12, ax22, ax23 = axises[1]
    
    ax12.get_shared_x_axes().join(ax1, ax12)
    ax22.get_shared_x_axes().join(ax2, ax22)
    
    ax2.get_shared_y_axes().join(ax1, ax2)
    ax22.get_shared_y_axes().join(ax12, ax22)
    ax13.get_shared_y_axes().join(ax1, ax13)
    ax23.get_shared_y_axes().join(ax12, ax23)

    ax1.xaxis.set_ticklabels([])
    ax2.xaxis.set_ticklabels([])
    ax1.xaxis.set_ticklabels([])
    ax2.yaxis.set_ticklabels([])
    ax22.yaxis.set_ticklabels([])
    ax13.yaxis.set_ticklabels([])
    ax23.yaxis.set_ticklabels([])
    
    #ax1 = plt.subplot(231)
    #ax2 = plt.subplot(232, sharey=ax1)
    #plt.tick_params('y', labelleft=False)

    #ax12 = plt.subplot(233, sharex=ax1)
    #ax22 = plt.subplot(234, sharey=ax12, sharex=ax2)
    #plt.tick_params('y', labelleft=False)
    ymin = (data.Roff.min() - data.dR.max())*1.1
    ymax = (data.Roff.max() + data.dR.max())*1.1
    maxy = 0
    miny = 0
    
    for t, tdata in data.groupby("trap"):
        tdata.sort_values("time", inplace=False)
        
        ax = ax1 if t == 2 else ax2
        axs = ax12 if t == 2 else ax22
        tcolor = color if t == 2 else colorB
        
        rmean = []
        rerrs = []
        rsidx = np.arange(len(tdata))
        for i in rsidx:
            sub = tdata[:i]
            rm, rs = statistics.mean_and_error(sub.Roff, dvalue = sub.dR)
            rmean.append(rm)
            rerrs.append(rs)
        
        axs.errorbar(rsidx, rmean, rerrs, alpha=0.3, c=colorB)
        
        idx = 0
        offset = 0
        lastoffset = 0
        for m, mdata in tdata.groupby(["measurement", "mcycle"]):
            #if m[1] == 2: continue
            print(m)
        
            Rs = mdata.Roff.to_numpy()
            dRs = mdata.dR.to_numpy()
            idxes = np.arange(len(Rs)) 
            #idxes = mdata.cycle.to_numpy()
            idxes -= (min(idxes) - 1)
            meandx = np.mean(idxes) + offset
            idxes += offset
            offset = max(idxes)
                                       
            ax.errorbar(idxes, Rs, dRs, ls="None", c=color)
                                       
            R, dR = statistics.mean_and_error(Rs, dvalue = dRs)
            
            miny = min([R-dR, miny])
            maxy = max([R+dR, maxy])
            axs.errorbar(meandx, R, dR, marker=".", ls="None", c=color, capsize= 3)
            #print(m, t, R, dR)
            
            #ax.vlines(offset, ymin, ymax, colors='k', linestyles='solid')
            
        R, dR = statistics.mean_and_error(tdata.Roff, dvalue = tdata.dR)
        print(t, R, dR)
        ax23.errorbar(t, R, dR, marker=".", c=tcolor, capsize= 3)
        ax13.hist(tdata.Roff, bins=30, orientation="horizontal", alpha=0.5, color=tcolor)
        ax13.xaxis.tick_top()
    
    ax23.set_xlim((1.4, 3.6))
    ax12.set_ylim((miny*1.1, maxy*1.1))
    ax.set_ylim((ymin, ymax))

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.03, hspace=0.15)
    fig = plt.gcf()
    trap2 = fig.text(0.27, 0.91, 'trap 2', ha='center', size=labelsize)
    trap3 = fig.text(0.665, 0.91, 'trap 3', ha='center', size=labelsize)
    #trap2.set_bbox(dict(facecolor="w", alpha=0.5, edgecolor=color, boxstyle='round'))
    #trap3.set_bbox(dict(facecolor="w", alpha=0.5, edgecolor=color, boxstyle='round'))
    fig.text(0.005, 0.5, r'$\textrm{R}_{\textrm{cf}} - \overline{\textrm{R}_{\textrm{cf}}}$', va='center', rotation='vertical', size=labelsize)
    fig.text(0.46, 0.015, 'measurement cycle', ha='center', size=labelsize)
    fig.text(0.925, 0.015, 'trap', ha='center', size=labelsize)
    #hint1 = fig.text(0.465, 0.555, 'single frequency ratios', ha='center', size=labelsize-4)
    hint1 = fig.text(0.465, 0.835, 'single frequency ratios', ha='center', size=labelsize-4)
    hint1.set_bbox(dict(facecolor="w", alpha=0.8, edgecolor=color, boxstyle='round'))
    hint1 = fig.text(0.465, 0.425, 'averaged per ion set / N det.', ha='center', size=labelsize-4)
    hint1.set_bbox(dict(facecolor="w", alpha=0.8, edgecolor=color, boxstyle='round'))
    
    if save is not None:
        plt.savefig(save, format='pdf', pad_inches=1)
    plt.show()
    