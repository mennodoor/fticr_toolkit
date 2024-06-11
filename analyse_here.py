from pathlib import Path
from pprint import pprint
import numpy as np
import papermill as pm
import os, time
from datetime import datetime
import fticr_toolkit
module_path = Path(fticr_toolkit.__file__).parents[1].absolute()
measurement_folder = Path.cwd()
os.chdir(module_path)

"""
This is the analysis in a 'copy me to the measurement folder' kind of way.
Just move it to the measurement folder, execute it there and it will excecute the notebooks
of the fticr toolkit and store them as html when they are done.
Part1 including averaging and fitting will take some time (~20 Minutes, depending on the length of the measureent)

"""

###    U S E R    S E T T I N G S    BEGIN #############################################################

measurement_script = 'unwrap_n_measure_repeat.py'

settings = {
    "grouping": [10],
    "reuse_averages": False,
    "reuse_fitdata": False,
    "fixed_phase_readout": False,
    # The filter settings are used to remove some data, which is nice if e.g. the ion was lost at some point and you dont want to see crappy random data. Either directly supply a pandas dset (more usefull for papermill batches) or rather
    # create a csv file as in the examples and just assign None to this variable, it will use the csv file then. 
    "filter_settings": None, # pandas dset with mc, trap, position, min_cycle, max_cycle; None: try to get filter settings from loaded data; False: no filter applied
    "post_unwrap": False, # if this is True, the analysis will automatically choose the post-unwrap (the pre-unwrap of the next main cycle) for nu_p frequency determination. Maybe the whole post-unwrap part is commented out in step 5, please check.
    "phase_filter": True, # filter measured phases by 3 sigma filter inside subcycle
    "averaging": True, # averaging subcycle data to match average_idx from part 1
    "sideband": False, # use sideband relation to calculate nu_c
    "polydegrees": 'auto', # number of degree of the polynom fit
    "polygrouping": 'auto', # group sizes for the polynom fit
    "poly_mode": "curvefit_AICc", # routine for polynom fitting, _ criterion for best model/poly-degree
}

part1_folder = "./part1_data/" # the measurements subfolder where the pre-analysed data (axial frequencies and nu_p phases) is
part2_folder = "./results/" # subfolder where the results of this analysis should go
use_settings_file = False # uses the settings file to overwrite this stuff here:

###    U S E R    S E T T I N G S    END ###############################################################



paras = {
    "settings": settings
}

# P A R T  1
print("running PART 1")

# the measurement folder is needed for the notebook and saving the final notebook
paras["measurement_folder"] = str(measurement_folder)
paras["input_file"] = 'pnp_dip_unwrap.fhds'
paras["output_folder"] = part1_folder
try:
    os.mkdir(str(measurement_folder) + paras["output_folder"])
except:
    pass
## Execute notebook
output_notebook = str(measurement_folder)+"/part1_data/part1.ipynb"
pm.execute_notebook(
    str(module_path) + '/Analysis_PART1.ipynb',
    output_notebook,
    parameters = paras,
    nest_asyncio=True
)

print("convert to HTML ... ")
os.system("jupyter nbconvert --to HTML "+ output_notebook)

# P A R T  2
print("running PART 2", str(measurement_folder))

paras["input_folder"] = part1_folder
paras["output_folder"] = part2_folder
try:
    os.mkdir(str(measurement_folder) + paras["output_folder"])
except:
    pass
## Execute notebook
output_notebook = str(measurement_folder)+"/results/part2.ipynb"
#print(output_notebook)
pm.execute_notebook(
    str(module_path) + '/Analysis_PART2.ipynb',
    output_notebook,
    parameters = paras,
    nest_asyncio=True
)

print("convert to HTML ... ")
os.system("jupyter nbconvert --to HTML "+ output_notebook)
