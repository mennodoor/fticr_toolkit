from setuptools import setup, find_packages

setup(
    name='fticr_toolkit',
    version='0.1',
    packages=find_packages(),
    license='MIT',
    description='Analysis Framework for FT-ICR datasets, Pentatrap experiment',
    long_description=open('README.md').read(),
    install_requires=["pandas", "h5py", "scipy", "matplotlib", "numpy", "jupyter", "ipywidgets", "qgrid", "plotly", "papermill", "statsmodels"],
    url='https://git.mpi-hd.mpg.de/Pentatrap/FT-ICR_Toolkit',
    author='Menno Door',
    author_email='door@mpi-k.de'
)

import os

os.system("jupyter nbextension enable --py widgetsnbextension")
os.system("jupyter nbextension enable --py --sys-prefix qgrid")
os.system("jupyter nbextension enable --py --sys-prefix plotlywidget")

# TODO: somehow install node.js without conda... (needed only for jupyter lab)

# TODO: call the configuration stuff for ipywidgets:
# jupyter nbextension enable --py widgetsnbextension
# jupyter labextension install @jupyter-widgets/jupyterlab-manager
