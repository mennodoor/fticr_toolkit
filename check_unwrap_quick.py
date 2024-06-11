from pathlib import Path
from pprint import pprint
import numpy as np
import papermill as pm
import os, time
from datetime import datetime
import fticr_toolkit
measurement_folder = Path.cwd()
module_path = Path(fticr_toolkit.__file__).parents[1].absolute()
os.chdir(module_path) # TODO: needed?

"""
This is the analysis in a 'copy me to the measurement folder' kind of way.
Just move it to the measurement folder, execute it there and it will excecute the notebooks
of the fticr toolkit and store them as html when they are done.
Part1 including averaging and fitting will take some time (~20 Minutes, depending on the length of the measureent)

"""

###    U S E R    S E T T I N G S    BEGIN #############################################################

measurement_script = 'unwrap_n_measure_repeat.py'

paras = {
    "input_file": 'pnp_dip_unwrap.fhds',  # the measurements subfolder where the pre-analysed data (axial frequencies and nu_p phases) is
    "output_folder": "./part1_data/", # subfolder where the results of this analysis should go
    "measurement_folder": str(measurement_folder),
    "settings": {
        "use_settings_file": False # uses the settings file to overwrite this stuff here:
    }
}

###    U S E R    S E T T I N G S    END ###############################################################

# the measurement folder is needed for the results and saving the final notebook

## Execute notebook
try:
    os.mkdir(str(measurement_folder) + paras["output_folder"])
except:
    pass

output_notebook = str(measurement_folder)+"/part1_data/part1.ipynb"
pm.execute_notebook(
    str(module_path) + '/JupyterNotebooks/check_unwrap_quick.ipynb',
    output_notebook,
    parameters = paras,
    nest_asyncio=True
)

print("convert to HTML ... ")
os.system("jupyter nbconvert --to HTML "+ output_notebook)
