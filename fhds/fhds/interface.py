from . import fhds
import numpy as np

class Interface:
    def __init__(self, group):
        self.group = group

    def save_spectrum(self, name, spectrum, header = {}, binary = False):

        filename = name+'.csv'
        if binary:
            filename = name+'.spec'
        
        data = self.group.create_dataset(filename)
        data_header =  {**header, **spectrum.header}
        data.meta = data_header
        filehandle = data.open(mode="wb")
        spectrum.save(filehandle, binary = binary)
        filehandle.close()
        
        
    def save_image_plt(self, name, plt_object, header = {}, format = 'png'):
        filename = name+'.'+format
        data = self.group.create_dataset(filename)
        filehandle = data.open()
        plt_object.savefig(filehandle, format=format)
        data.meta = header

    def save_image_raw(self, name, imagedata, header = {}, format = 'png'):

        import imageio

        filename = name+'.'+format
        data = self.group.create_dataset(filename)
        filehandle = data.open()
        imageio.imwrite(filehandle, imagedata[:, :, :], format=format)
        data.meta = header

    def __iter__(self):
        # add iteration capability such that files are given back
        # I rather return the data...
        # return the data would needed the knowlege how the spectrum or what other data is saved...
        self.iter = iter(self.group.datasets)
        return self

    def __next__(self):
        return next(self.iter)

    #def __getitem__(self, item):
    #    return next((x for x in self.group.datasets if x.name == item), None)


class MeasurementData:
    def __init__(self, group):
        self.group = group
        self.name = self.group.path.name
        self.path = self.group.obj_path
        self.data = Interface(group)
        self._sub_measurements = []

    @property
    def sub_measurements(self):
        if not self._sub_measurements: #is it empty?  mainly for average_axial
            self._sub_measurements = [MeasurementData(x) for x in self.group.groups]
        return self._sub_measurements

    @sub_measurements.setter
    def sub_measurements(self, value):
        self._sub_measurements = value

    @property
    def meta(self):
        return self.group.meta

    @meta.setter
    def meta(self, value):
        self.group.meta = value

    @property
    def attrs(self):
        return self.group.attrs

    def create_sub_measurement(self, name, open_if_exist=False):
        #existing
        # Return MeasurementData instance
        
        if self[name] is not None and open_if_exist: 
            return self[name]

        if self[name] is not None:
            raise ValueError("Sub measurement already exist: ", name)

        group = self.group.create_group(name)
        new_meas = MeasurementData(group)
        self.sub_measurements.append(new_meas)
        return new_meas

    def create_cycle(self): #is cycle a subset of a step function

        if not self.meta.has_key("cycle_count"):
            self.meta["cycle_count"] = 0

        self.meta["cycle_count"] = self.meta["cycle_count"] + 1

        cycle = self.create_sub_measurement("cycle" + str(self.meta["cycle_count"]))
        cycle.meta["cycle"] = self.meta["cycle_count"]
        return cycle

    # add iteration capability such that sub_measurements are given back
    def __iter__(self):
        self.iter = iter(self.sub_measurements)
        return self

    def __next__(self):
        return next(self.iter)

    def __getitem__(self, item):
        try:
            #print(item)
            #print(self.keys())
            if item.startswith('/'):
                item = item[1:]
            if item.endswith('/'):
                item = item[:-1]
            if "/" in item:
                first_group, rest_of_path = item.split('/', 1)
                if first_group in self.keys():
                    return self[first_group][rest_of_path]
                else:
                    raise KeyError
            else:
                return next((x for x in self.values() if x.name == item), None)
        except:
            raise KeyError

    def keys(self):
        return [x.name for x in self.sub_measurements] + [x.name for x in self.group.datasets] 

    def values(self):
        return [x for x in self.sub_measurements] + [x for x in self.group.datasets] 

    def items(self):
        return [(x.name, x) for x in self.sub_measurements] + [(x.name, x) for x in self.group.datasets] 

    def visit(self, callback):
        callback(self.path)
        for obj in self.values():
            obj.visit(callback)

    def visititems(self, callback):
        callback(self.path, self)
        for obj in self.values():
            obj.visititems(callback)

def load_measurement(name, mode="r"):
    root = fhds.load(name, mode)
    measurement = MeasurementData(root)
    return measurement