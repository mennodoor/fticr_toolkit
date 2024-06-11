import os
import copy
from datetime import datetime, timedelta
from pathlib import Path
from importlib import import_module, reload

import h5py
import json
import numpy as np
import pandas as pd
from pprint import pprint

from sqlalchemy.ext.hybrid import ExprComparator

### Data import #####################################################################################################

def load_data(measurement_folder="./example_data", input_data="part1_data", output_folder="part2_data", measurement_script = "unwrap_n_measure_repeat"):
    """
    Move current working directory to measurement folder, loads the input data and measurement 
    config and creates the output folder if neccessary. 
    
    If you  want to load raw measurement data stored in the fhds package format, please do input_data="<root_folder_name>.fhds".
    """
    # change cwd
    try:
        os.chdir(Path(measurement_folder))
    except:
        print("\n >>> WARNING!!!! Changing directoy to measurement folder did not work. You maybe used a relative path and are already in the correct path. Absolut paths are much safer for that matter...")
        print("\n >>> --> current directory", os.getcwd())

    # NOTE: return this dict later!
    try:
        mod = import_module(measurement_script)
        mod = reload(mod)
        meas_config = getattr(mod, "config")
    except Exception as e:
        print(e)
        meas_config = None

    # creates output_folder directory:
    if output_folder is not None:
        try:
            os.makedirs(Path(output_folder))
            print(os.getcwd())
        except:
            print("\n >>> WARNING!!! output_folder already exists! files will maybe be overwritten! <<< \n")

    # load measurement data:
    data = {}
    if input_data.endswith(".hdf5"):
        print("\n >>> I hope you are in PART1, loading raw measurement data now... <<< \n")
        h5obj = h5py.File(Path(input_data), "r")
        data["meas_data"] = h5obj
    elif input_data.endswith(".fhds"):
        from fhds import fhds, interface
        print("\n >>> I hope you are in PART1, loading raw measurement data now... <<< \n")
        fhds_root = interface.load_measurement(Path(input_data[:-5]), "r")
        data["meas_data"] = fhds_root
    else:
        print("\n >>> I hope you are in PART2 or 3, loading pre analysed data from folder... <<< \n")
        data_folder = Path(input_data)
        filenames = data_folder.glob("*.csv")
        for filename in filenames:
            df = pd.read_csv(filename)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # removes unneccessary index column
            if not "masked" in df:
                df["masked"] = False
            try:
                df["time"] = pd.to_datetime(df['time'])
                #df["time"] = df['time'].str.split('.').str[0]
            except KeyError:
                pass
            except:
                print("\n >>> WARNING!!! Time data conversion to datetime format failed for "+str(filename)+" <<< \n")
            if 'time' in df:
                df = df[df['time'].dt.year > 2010 ]
            if "R" in df.columns and "Rminus" not in df.columns:
                df["Rminus"] = df["R"] - 1
            if "R" in df.columns and df.R.mean() < 1:
                df["R"] = 1/df.R.to_numpy()
                df["Rminus"] = df["R"] - 1
                df["dR"] = 1/np.power(df.R.to_numpy(), 2)*df.dR
                try:
                    ion_num = df["ion_numer"].to_numpy()
                    df["ion_numer"] = df["ion_denom"].to_numpy()
                    df["ion_denom"] = ion_num
                except KeyError:
                    print("\n >>> WARNING!!! Flipped ratio for "+str(filename)+" but could not flip ion numerator and ion denominator colums! <<< \n")

            data[filename.stem] = df

    return data, meas_config, output_folder

def load_data_multi_folder(folderlist=[], input_data='results', measurement_script = "unwrap_n_measure_repeat"):
    """Imports data from multiple measurement folders.
    
    Returns a dict with same structure as load_data, just with an additional first level with key of the measurement
    folder and value of data (measurement config also inside data).

    Args:
        folderlist (list, optional): [description]. Defaults to [].
        input_data (str, optional): [description]. Defaults to 'results'.
        measurement_script (str, optional): [description]. Defaults to "unwrap_n_measure_repeat".
    """

    all_data = {}
    for submeas in folderlist:
        print(submeas)
        all_data[submeas.stem] = {}
        data, config, _ = load_data(measurement_folder=submeas, input_data=input_data, output_folder=None, measurement_script = measurement_script)
        data["measurement_config"] = copy.deepcopy(config)
        #print(config['position_1']['configuration']['traps'][2]['excitation_amplitude'])
        all_data[submeas.stem] = data
    
    return all_data

def input_filter(data_dict, settings = None, UnwrapFilter=False):
    """
    Uses an the filter_settings.csv data to filter input data right from the beginning...
    """
    if isinstance(settings, pd.DataFrame):
        pass
    elif settings is not None and settings == False:
        return data_dict
    elif settings is None:
        settings = pd.read_csv("filter_settings.csv")
    print("filter settings:")
    pprint(settings)

    for key, dset in data_dict.items():
        new_dset = pd.DataFrame()

        if key.find("unwrap") != -1 and not UnwrapFilter: #or "Ndet" in key: # dont apply the filter for N determination measurements!!
            print("skip", key)
            continue
        elif key.find("unwrap") != -1:
            cyclefilter = False
        else:
            cyclefilter = True

        try:
            for i in range(0, len(settings.index)):
                row = settings.iloc[i]
                mc = int(row["mcycle"])
                trap = row["trap"]
                position = row["position"]  
                if row["max_cycle"] == 0:
                    continue
                if row["min_cycle"] > 0 and not cyclefilter:
                    continue
                subset = dset[ (dset["mcycle"] == mc) & (dset["trap"] == trap) & (dset["position"] == position) ]
                if cyclefilter:
                    subset = subset[ (subset["cycle"] >= row["min_cycle"]) & (subset["cycle"] <= row["max_cycle"]) ]

                new_dset = new_dset.append( subset , ignore_index=True )
        except KeyError as e:
            print("not filterable, missing key for data file", key)
            print(e)
            new_dset = dset

        if 'time' in new_dset:
            new_dset = new_dset[new_dset['time'].dt.year > 2010 ]

        data_dict[key] = new_dset

    data_dict["filter_settings"] = settings
    return data_dict

def load_settings(folder, settings = {}):
    data_folder = Path(folder)
    # settings are stored in json files
    filenames = data_folder.glob("settings.json")
    for filename in filenames:
        with open(filename, "r") as json_file:
            jdict = json.load(json_file)
            settings.update( jdict )
    return settings

### Pandas helper #####################################################################################################

def update_pdset(target, source, target_columns, source_columns, use_index=False):
    """This function returns the target DataFrame, where all target_columns have
    been overwritten by the source_columns data from the source Dataframe. If the target
    column does not yet exist, a new column will be created.
    The index of the source DataFrame will only be used for merging if use_index == True.
    If the index is not used, the DataFrames have to be of same length!

    Args:
        target (pd DataFrame): [description]
        source (pd DataFrame): [description]
        target_columns (list): [description]
        source_columns (list): [description]
        use_index (bool): [description]

    Raises:
        ValueError: [description]

    Returns:
        pd DataFrame: [description]
    """
    pd.options.mode.chained_assignment = None  # default='warn'

    if not target_columns:
        target_columns == source_columns
    if len(source_columns) != len(target_columns):
        raise ValueError("The two column arrays have to have the same size!")

    for tcol, scol in zip(target_columns, source_columns):
        try: # first try overwrite
            if use_index:
                target[tcol] = source[scol].copy()
            else:
                target[tcol][:] = source[scol].to_numpy()
        except KeyError:
            try: # if not existing, try inserting
                if use_index:
                    target.insert(-1, tcol, source[scol].copy() )
                else:
                    target.insert(-1, tcol, source[scol].to_numpy() )
            except KeyError:
                print('source column', scol, 'does not exist')
            except:
                print('source column', scol, 'could not be tranfered to target column', tcol)
                raise
    
    pd.options.mode.chained_assignment = 'warn'  # default='warn'
    return target

def fix_column_dtypes(df, more_dtypes = {}):

    dtypes = {
        "mcycle": 'int64',
        "trap": 'int64',
        "cycle": 'int64',
        "subcycle": 'int64',
        "masked": 'bool',
        "average_idx": 'int64',
        "average_idx_p": 'int64'
    }
    dtypes.update(more_dtypes)
    for col in df.columns:
        for key, dty in dtypes.items():
            if key in col:
                try:
                    df[col] = df[col].astype(dty)
                except:
                    pass

    return df

def sort_cycles(list_of_cycle_names):
    x = list_of_cycle_names
    y = [ int(t[5:]) for t in x]
    z = sorted(y)
    return ['cycle'+str(t) for t in z]

def group_len2index(len_list):
    idx_list = []
    idx = 0
    for length in len_list:
        idx_list.extend([idx]*length)
    return idx_list

### Spectrum averaging #############################################################################################################################

def average_data(data_group, average_group_lengths=[10], only_attr = {"type": "amplitude"}, time_format = '%Y%m%d_%H%M%S.%f', min_data_size=40000):
    """
    Takes a data group (hdf5 or fhds) and averages all dsets in this group according to the grouping
    given in the average_group_lengths variable.

    returns a list of list: [ [first_dset_attributes, averaged_dset_data], [ ... ], ... ]
    """
    keys = []
    for name, dset in data_group.items():
        use = False
        for attr, val in only_attr.items():
            if dset.attrs[attr] == val:
                use = True
        if use:
            keys.append(name)

    length_data_group = len(keys)
    #print(length_data_group)
    length_sum = sum(average_group_lengths)
    counter = 0
    
    #print((data_group.name))
    grp_index = 0
    grp_len = average_group_lengths[grp_index]
    data_index = 0
    
    avg_data = [ ] # tuple of attrs, averaged data
    data_shape = None
    time_list = None
    
    # loop over datasets and average
    #for name, dset in data_group.items():
    for name in keys:
        dset = data_group[name]
        # check if this is the correct type of data to average
        skip = False
        for attr, val in only_attr.items():
            if dset.attrs[attr] != val:
                skip = True
        if skip:
            continue

        if 'time' in dset.attrs.keys() and time_list is None:
            time_list = []

        # NOTE: this here is sadly very important: the dset[:] hhas to come before
        #       getting the attributes, due to some automated update in fhds
        dset_data = dset[:]
        dset_attrs = attrs_to_dict(dset)
        #print(attrs_to_dict(dset))

        wierd_data = False

        if dset_data.size < min_data_size:
            # axial spec typical resolution 0.1 Hz, span 4000 Hz 
            print('spectrum to small, probably a phase spec')
            wierd_data = True

        if data_shape is None and not wierd_data:
            data_shape = dset_data.shape
        elif not wierd_data:
            if data_shape != dset_data.shape:
                print("warning, wierd data shape!", name)
                wierd_data = True

        counter += 1

        #print(grp_index, data_index, avg_data)
        if data_index == 0 and not wierd_data:
            avg_data.append( [dset_attrs, dset_data] )
            avg_counter = 1
        elif not wierd_data:
            avg_data[grp_index][1] += dset_data
            avg_data[grp_index][0].update(dset_attrs)
            avg_counter += 1
        else:
            pass
        
        #if time_list is not None and not wierd_data: # 20200714_022108.073367
        if time_list is not None: # rather take all the time stamps to get the mean, just more practical in the end.
            #print(dset.attrs['time'])
            time_list.append( datetime.strptime(dset.attrs['time'], time_format) )
        
        data_index += 1
        
        # that was the last index to average
        if data_index >= grp_len or counter >= length_sum or counter >= length_data_group:
            # so divide through number of datasets and get mean time
            avg_data[grp_index][1] /= avg_counter
            #print("normalized", avg_counter)
            if time_list is not None:
                time_list = np.asarray(time_list)
                #print(time_list)
                m = time_list.min()
                mean_time = (m + (time_list - m).mean())
                #mean_time = time_list.mean()
                #print(' >>> mean time: ', mean_time)
                avg_data[grp_index][0]['time'] = mean_time.strftime(time_format)
                time_list = None # important! re-None it
                
            try:
                grp_index += 1
                grp_len = average_group_lengths[grp_index]
                data_index = 0
            except:
                #if counter == length_sum:
                if counter == length_data_group:
                    break
                else:
                    print("WARNING! NOT ALL DATA WAS USED? counter =", counter)
        
    return avg_data

### HDF5 Helper ########################################################################################################

def clear_h5py_file(filename):
    with h5py.File(filename, 'w') as f1:    # open the file
        print("file cleared")

def write_save(filename, path, data, attrs={}, check=True):
    with h5py.File(filename, 'r+') as f1:   # open the file
        try:
            dset = f1[path]                  # load the data 
            dset[...] = data                 # assign new values to data
        except KeyError:                    # create a new dset if neccessary
            dset =  f1.create_dataset(path, data=data)

        for key, item in attrs.items():     # write attributes
            #print(key, item)
            dset.attrs.create(key, item)
            if item != dset.attrs[key]:
                raise IOError("Attribute not correctly written "+str(key)+" path "+str(path))

    with h5py.File(filename, 'r') as f1:
        ret = np.allclose(f1[path][()], data) # checks if data in file and data given are the same
        #for key, val in f1[path].attrs.items():
        #    print(key, val)

    if check and not ret:
        raise ValueError("The data was not written correctly to the file!")
    return True

def get_group(root, *args):
    """
    This function can take unlimited strings as args (after the root group of a hdf5 or fhds file) and loops
    through these group names, basically going down the path by attaching the group name one after the other.

    returns the group object
    """
    for grp in args:
        root = root[grp]
    return root

def get_data_paths( h5py_datagroup, attr_type=("phase", "amplitude"), name_includes=None, name_startswith=None):
    datakeys = []
    try:
        root_path = h5py_datagroup.path
    except:
        root_path = ""
        
    def collect(name, item):
        if name.startswith(root_path):
            #print(root_path)
            name = name[len(root_path):]

        ending = name.split('/')[-1]
        #print(ending, name_includes)
        if name_startswith is not None:
            if ( ending.startswith( name_startswith ) ):
                datakeys.append(name)
                #print(name)
        elif name_includes is not None:
            if any(substring in ending for substring in name_includes):
                datakeys.append(name)
                #print(name)
        elif attr_type is not None:
            if any(a_type == item.attrs.get("type", "") for a_type in attr_type):
                datakeys.append(name)
                #print(name)

    h5py_datagroup.visititems(collect)
    return datakeys

def attrs_to_pdSeries(dset):
    try:
        path = dset.path
    except:
        path = dset.name

    columns = ["path"]
    data = [path]
    for key, item in dset.attrs.items():
        columns.append(key)
        data.append(data)
        
    return pd.Series( data, index=columns )

def attrs_to_dict(dset):
    ret_dict = {}
    for key, val in dset.attrs.items():
        ret_dict[key] = val
    return ret_dict

def data_time_filter( h5py_datagroup, time_min_max = [None, None], startswith=("amplitude", "phase"), stringfrmt = '%Y%m%d_%H%M%S.%f'):

    datakeys = get_data_paths(h5py_datagroup, startswith)
    if time_min_max[0] is not None:
        time_min_max[0] = datetime.strptime(time_min_max[0], stringfrmt)
    if time_min_max[1] is not None:
        time_min_max[1] = datetime.strptime(time_min_max[1], stringfrmt)
    
    masked_keys = []
    for key in datakeys:
        try:
            time = datetime.strptime(h5py_datagroup[key].attrs["time"], stringfrmt)
            h5py_datagroup[key].attrs["masked"] = False

            if time_min_max[0] is not None and time < time_min_max[0]:
                h5py_datagroup[key].attrs["masked"] = True
                masked_keys.append(key)
            if time_min_max[1] is not None and time > time_min_max[1]:
                h5py_datagroup[key].attrs["masked"] = True
                masked_keys.append(key)
        except:
            print("error for time query of datakey", key)

    return masked_keys

### Spectrum ########################################################################################################

class Spectrum():
    def __init__(self, dataset, start = None, stop = None):
        """
        This is a class for Spectra
        The dataset could be a hdf5 dataset, a pandas dataset or a numpy array, it just has to be addressable
        by dataset[0][:] for amplitudes and dataset[1][:] for phases and in case of pandas or numpy start and stop
        of the frequency span have to be given using the init function and in a hdf5 dataset using attrs
        """
        self.dataset = dataset
        self.name = dataset.name
        self.attrs = dataset.attrs
        self.start = start
        self.stop = stop
        #print(dataset.attrs.keys())
        self.reset()
        
    def reset(self):
        data = self.dataset[:]
        if len(data) == 2:
            self.amps = data[0]
            self.phases = data[1]
            self.steps = len(self.amps)
            try:
                self.start = self.dataset.attrs["start"]
                self.stop = self.dataset.attrs["stop"] 
                self.steps = int(self.dataset.attrs["binnum"])
            except NameError:
                print("no attrs assigned to dataset...")
                if None in (self.start, self.binnum):
                    print("please define frequency space on Spectrum-init")
                    raise
            
            self.freqs = np.linspace(self.start, self.stop, self.steps)

        elif len(data) == 3:
            self.freqs = data[0]
            self.amps = data[1]
            self.phases = data[2]
            self.steps = len(self.amps)
            self.start = self.freqs[0]
            self.stop = self.freqs[-1] 
        else:
            print("not possible to create SPectrum object. Frequencz data missing? (start, stop missing?)")
            raise KeyError

        if "span" not in self.dataset.attrs.keys():
            self.span = self.stop - self.start
        if "center" not in self.dataset.attrs.keys():
            self.center = self.start + self.span/2
            
    def cut(self, center, span):
        new_freqs = self.freqs[np.where(abs(self.freqs-center) < span/2)]
        self.amps = self.amps[np.where(abs(self.freqs-center) < span/2)]
        self.phases = self.phases[np.where(abs(self.freqs-center) < span/2)]
        self.freqs = new_freqs
        self.center = center
        self.span = span
        return 
    
    def downsample(self, factor=3, inplace=True):
        if factor%2 != 1:
            raise ValueError("needs an odd value")
        
        cutlen = int(len(self.amps)/factor)-1
        subsets = None
        for i in range(factor):
            sub = self.amps[i::factor][:cutlen]
            #print(len(sub))
            if subsets is None:
                subsets = np.reshape(sub, (1,-1))
            else:
                subsets = np.append(subsets, [sub], axis=0)
        
        #print(subsets)
        
        amps = np.sum(subsets, axis=0)/factor
        freqs = self.freqs[int(factor/2)::factor][:cutlen]

        if inplace:
            self.amps = amps
            self.freqs = freqs
            return
        else:
            return freqs, amps

    def get_phase(self, frequency):
        index = np.argmin(np.abs(np.array(self.freqs)-frequency))
        return self.phases[index]

    def get_amplitude(self, frequency):
        index = np.argmin(np.abs(np.array(self.freqs)-frequency))
        return self.amplitudes[index]

    def get_attrs(self):
        return self.dataset.attrs

    def get_meta(self):
        self.meta = {}
        for key, val in self.dataset.attrs.items():
            self.meta[key] = val
        return self.meta