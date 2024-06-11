#filesystem based hirachical data storage

import json
import os
from pathlib import Path
import numpy as np

class Metadict(object):
    def __init__(self, path, filename, mode):
        self.path = path
        self.filename =  filename + "_meta"
        self.mode = mode

        # assign meta data file, if not existing create empty one
        if not self.path.joinpath(self.filename + '.json').exists():
            self.write_meta({})

    def write_meta(self, meta):
        if self.mode.startswith('r'):
            print('object loaded in read mode, no writing possible')
            return 0
        filename = self.path.joinpath(self.filename + '.json')
        with filename.open('w') as outfile:
            json.dump(meta, outfile, indent=2, sort_keys=False)

    def read_meta(self):
        filename = self.path.joinpath(self.filename + '.json')
        with filename.open('r') as f:
            data = json.load(f)
        return data

    def __setitem__(self, key, item):
        meta = self.read_meta()
        meta[key] = item
        self.write_meta(meta)

    def __getitem__(self, key):
        meta = self.read_meta()
        return meta[key]

    def get(self, key, default = None):
        meta = self.read_meta()
        try:
            value = meta[key]
        except KeyError:
            if default is not None:
                value = default
            else:
                raise
        return value

    def __repr__(self):
        return repr(self.read_meta())

    def __len__(self):
        return len(self.read_meta())

    def __delitem__(self, key):
        meta = self.read_meta()
        del meta[key]
        self.write_meta(meta)

    def clear(self):
        meta = self.read_meta()
        ret = meta.clear()
        self.write_meta(meta)
        return ret

    def copy(self):
        return self.read_meta().copy()

    def has_key(self, k):
        return k in self.read_meta()

    def update(self, *args, **kwargs):
        meta = self.read_meta()
        ret = meta.update(*args, **kwargs)
        self.write_meta(meta)
        return ret

    def keys(self):
        return self.read_meta().keys()

    def values(self):
        return self.read_meta().values()

    def items(self):
        return self.read_meta().items()

class VolatileAttrs():
    def __init__(self, metadict_obj):
        self.meta = metadict_obj
        self.mode = self.meta.mode
        self.attrs = {}

    def read_attrs(self):
        meta = self.meta.read_meta()
        meta.update(self.attrs)
        return meta

    def __setitem__(self, key, item):
        attrs = self.read_attrs()
        attrs[key] = item
        self.attrs = attrs
        if self.mode != "r":
            self.meta.write_meta(attrs)

    def __getitem__(self, key):
        attrs = self.read_attrs()
        return attrs[key]

    def get(self, key, default = None):
        attrs = self.read_attrs()
        try:
            value = attrs[key]
        except KeyError:
            if default is not None:
                value = default
            else:
                raise
        return value

    def has_key(self, k):
        return k in self.read_attrs()

    def update(self, *args, **kwargs):
        attrs = self.read_attrs()
        ret = attrs.update(*args, **kwargs)
        if self.mode != "r":
            self.meta.write_meta(attrs)
        self.attrs = attrs
        return ret

    def keys(self):
        return self.read_attrs().keys()

    def values(self):
        return self.read_attrs().values()

    def items(self):
        return self.read_attrs().items()

class Metadata(object):
    def __init__(self, path, filename, mode, mother_obj_path=''):
        self.path = path
        self.mode = mode
        self.filename = filename
        if mother_obj_path == '':
            self.obj_path = '/'
        elif mother_obj_path == '/':
            self.obj_path = mother_obj_path + self.path.name
        else:
            self.obj_path = mother_obj_path + '/' + self.path.name
        self.__meta__ = Metadict(path, filename, mode)
        self.__attrs__ = VolatileAttrs(self.__meta__) 

    @property
    def meta(self):
        return self.__meta__

    @meta.setter
    def meta(self, value):
        if self.mode.startswith('r'):
            print('object loaded in read mode, no writing possible')
            return 0
        
        self.__meta__.write_meta(value)

    @property
    def attrs(self):
        return self.__attrs__


class Dataset(Metadata):
    def __init__(self, path, name, mode, mother_obj_path=''):
        super().__init__(path, name, mode, mother_obj_path=mother_obj_path)
        self.name = name
        self.file_path = self.path
        self.path = mother_obj_path + "/" + self.name

    def open(self, mode=None):
        filename = self.file_path.joinpath(self.name)

        if mode is None:
            raise ValueError('Mode needs to be defined while opening: ' + str(filename))

        return filename.open(mode)

    def __getitem__(self, key):
        if isinstance(key, slice):
            file_obj = self.open(mode="r")
            data = np.fromfile(file_obj, dtype=np.float32) 
            start, stop, num = data[0:3]
            self.attrs["start"] = float(start)
            self.attrs["stop"] = float(stop)
            self.attrs["binnum"] = int(num)
            data = data[3:].reshape((2, int(data[3:].size/2)))
            #freq = np.linspace(float(start),float(stop),int(num))
            #data = np.asarray([freq, data[0], data[1]])
            return data[key]
        else:
            print("Dataset, only slicing posibile!")
            raise KeyError

    def __str__(self): 
        return str(self.file_path.joinpath(self.name))

    def visit(self, callback):
        callback(self.path)

    def visititems(self, callback):
        callback(self.path, self)

class Group(Metadata):
    '''
    Attributes:
        _groups: a list which comprises all folders under current directory, each element is a Group instance
        datasets: a list which comprises all data files under current directory, each element is a Dataset instance
    '''
    def __init__(self, path, mode, mother_obj_path=''):
        super().__init__(path, str(path.name), mode, mother_obj_path=mother_obj_path)

        #self.groups = [Group(x, mode) for x in self.path.iterdir() if x.is_dir()]

        self._groups = []
        self.datasets = [Dataset(self.path, x.name, self.mode, self.obj_path) 
                        for x in self.path.iterdir() 
                        if x.is_file() and not x.name.endswith("_meta.json")]
        self.datasets.sort(key=lambda x: x.name, reverse=False)
        
    ## TODO: Check if dataset or group is already present in groups or datasets
    ## Maybe there is already an error because the folder already exists

    def __len__(self):
        return len(self.datasets)

    @property
    def groups(self):
        if not self._groups: # _groups is empty => initalise
            self._groups = [Group(x, self.mode, self.obj_path) for x in self.path.iterdir() if x.is_dir()]
            self._groups.sort(key=lambda x: x.path.name, reverse=False)

        return self._groups

    @groups.setter
    def groups(self, value):
        self._groups = value

    def update(self): # very reduantend code in this function...
        self._groups = [Group(x, self.mode, self.obj_path) for x in self.path.iterdir() if x.is_dir()]
        self._groups.sort(key=lambda x: x.path.name, reverse=False) 
        self.datasets = [Dataset(self.path, x.name, self.mode, self.obj_path) for x in self.path.iterdir() if x.is_file() and not x.name.endswith("_meta.json")]
        self.datasets.sort(key=lambda x: x.name, reverse=False)
        for group in self._groups:
            group.update()

    def create_dataset(self, name):
        dataset = Dataset(self.path, name, self.mode, self.obj_path)
        self.datasets.append(dataset)
        return dataset

    def create_group(self, name):
        path = self.path.joinpath(name)
        path.mkdir()
        group = Group(path, self.mode, self.obj_path)
        self.groups.append(group)
        return group

    def __str__(self): 
       return str(self.path.name)

def load(pathname, mode="r"):
    path = Path(pathname)
    if mode == "w":
        path.mkdir()

    if mode == "a":
        if not path.exists():
            path.mkdir()
        mode = "w"

    return Group(path, mode)
