# FT-ICR Toolkit

Summary of tools needed for FT-ICR data analysis and calculation of systematic shifts.

## Installation

Install the package via pip, but for best use as a in development mode.

```
pip install --user -e .
```

The requirements will be installed mostly automatically. 

If some of the visualization do not work, try to enable the respective features:

```
jupyter nbextension enable --py widgetsnbextension
jupyter nbextension enable --py qgrid
jupyter nbextension enable --py plotlywidget
```

If you get the following error when enabling qgrid and plotly:
- Validating: problems found:
  - require?  X qgrid/extension
The automatic installation failed but this can be easily fixed with these lines: 

```
jupyter nbextension install --py qgrid
jupyter nbextension enable --py qgrid
jupyter nbextension install --py plotlywidget
jupyter nbextension enable --py plotlywidget
```

**Overview**:
- **PART1** : converts the raw data of the measurement (spectra, phase spectra, time domain data?) and converts it to usable phase and frequency information for the next part of the analysis. This part takes a while since its doing averages of numpy arrays, fits and fft conversions on bigger datasets.
- **PART2** : takes the phase and frequency information and calculates the free cyclotron frequency and eventually mass ratios (maybe also more, if we add analysis of systematic measurements here as well...).
- **PART3** : only needed for multiple measurements of the same type, like multiple ratio measurements for the same two ion types. This will summarize all the data and creates means. Ideally it will also calculate some estimates for systematics (?)
- **PART4** : This is for batch processing multiple measurement runs, it just runs PART1  and PART2 automatically for all measurement folders in one parent folder.
