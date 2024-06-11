import os
from datetime import datetime, timedelta
from pathlib import Path
from importlib import import_module

import h5py
import json
import numpy as np
import pandas as pd
from pprint import pprint

from fticr_toolkit import data_conversion

def average_data(data_group, average_indexes = [0,1,0,1,0,1,0,1,0,1], data_idx_attr = "avg_num", only_attr = {"type": "amplitude"}, time_format = '%Y%m%d_%H%M%S.%f', min_data_size=40000):
    """
    Takes a data group (hdf5 or fhds) and averages all dsets in this group according to the grouping
    given in the average_group_lengths variable.

    returns a list of list: [ [first_dset_attributes, averaged_dset_data], [ ... ], ... ]
    """
    keys = []
    dset_idxs = []
    for name, dset in data_group.items():
        use = False
        for attr, val in only_attr.items():
            if dset.attrs[attr] == val:
                use = True
        if use:
            dset_idxs.append( dset.attrs[data_idx_attr] ) 
            keys.append(name)

    length_data_group = len(keys)
    length_averaging_idx = len(average_indexes)
    #if length_data_group != length_averaging_idx:
    #    raise IndexError("The length of average indexes and number of datasets must be the same.")
    grp_indexes = list(set(average_indexes)) # basically average_indexes.unique()

    avg_data = {} # storage to give away later
    for idx in grp_indexes:
        avg_data[idx] = {"array": None, "time_list": [], "attr": None, "shape": None, "time": None}

    # loop over datasets and average
    for idx, name in zip(dset_idxs, keys):
        dset = data_group[name]

        # NOTE: this here is sadly very important: the dset[:] hhas to come before
        #       getting the attributes, due to some automated update in fhds
        dset_data = np.asarray(dset[:], dtype=np.float64)
        dset_attrs = data_conversion.attrs_to_dict(dset)
        #print(data_conversion.attrs_to_dict(dset))

        wierd_data = False

        if dset_data.size < min_data_size:
            # axial spec typical resolution 0.1 Hz, span 4000 Hz 
            print('spectrum to small, probably a phase spec')
            wierd_data = True

        avg_idx = average_indexes[idx]

        if avg_data[avg_idx]["shape"] is None and not wierd_data:
            avg_data[avg_idx]["shape"] = dset_data.shape
        elif not wierd_data:
            if avg_data[avg_idx]["shape"] != dset_data.shape:
                print("warning, wierd data shape!", name)
                wierd_data = True

        #print(idx, avg_idx, avg_data[avg_idx]["array"])
        if avg_data[avg_idx]["array"] is None and not wierd_data:
            avg_data[avg_idx]["array"] = dset_data
            avg_data[avg_idx]["attrs"] = dset_attrs
            avg_counter = 1
        elif not wierd_data:
            avg_data[avg_idx]["array"] += dset_data
            avg_data[avg_idx]["attrs"].update(dset_attrs)
            avg_counter += 1
        else:
            pass

        avg_data[avg_idx]["time_list"].append( datetime.strptime(dset.attrs['time'], time_format) )

    avg_data_list = []
    for avg_idx, data in avg_data.items():

        time_list = np.asarray(data["time_list"])
        #print(time_list)
        m = time_list.min()
        mean_time = (m + (time_list - m).mean())
        avg_data[avg_idx]["time"] = mean_time.strftime(time_format)
        avg_data[avg_idx]["attrs"]["time"] = mean_time.strftime(time_format)

        avg_data[avg_idx]["array"] /= len(time_list)

        avg_data_list.append( (avg_data[avg_idx]["array"], avg_data[avg_idx]["attrs"] ) )
        
    return avg_data