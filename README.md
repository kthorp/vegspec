# vegspec
A compilation of spectral vegetation indices and transformations in Python

The goal of the vegspec package is to encapsulate computations of 1) more than 145 published spectral vegetation indices and 2) several published pretreatment transformations for vegetative spectral reflectance data.

Scientific background material for the methods implemented in the vegspec package is given [here](http://github.com/kthorp/vegspec/tree/main/docs/document.pdf).

## Source Code
The spectral methodologies are contained in one Python module (i.e., [vegspec.py](http://github.com/kthorp/vegspec/tree/main/src/vegspec/vegspec.py)). The module defines a class (i.e., `VegSpec`) to encapsulate all the computations for one vegetative reflectance spectrum. To analyze sets of multiple spectral measurements, users can develop a list of unique `VegSpec` instances for each measurement. At minimum, each `VegSpec` instance requires inputting two 1-dimensional lists that specify the spectral reflecance factors and associated wavelengths.

The source code is available [here](http://github.com/kthorp/vegspec/tree/main/src/vegspec/vegspec.py).

## Install
`pip install vegspec`

## Quickstart

### Import the package
`import vegspec as vs`

### Instantiate a VegSpec class
`spectrum = vs.VegSpec(wl,rf)`

where required inputs are defined as follows:

* wl : list, Wavelengths (nm) corresponding to the data in rf
* rf : list, Reflectance factors (0 to 1) for a single spectral scan

Other optional arguments for the `VegSpec` constructor are as follows:

* fs1 : int, optional, Filter size for Savitsky-Golay first derivatives, (default = 7)
* po1 : int, optional, Polynomial order for Savitsky-Golay first derivatives, (default = 2)
* fs2 : int, optional, Filter size for Savitsky-Golay second derivatives, (default = 15)
* po2 : int, optional, Polynomial order for Savitsky-Golay second derivatives, (default = 2)
* wlblu : float, optional, Primary wavelength of visible blue light (nm), (default = 480.)
* wlgrn : float, optional, Primary wavelength of visible green light (nm), (default = 550.)
* wlred : float, optional, Primary wavelength of visible red light (nm), (default = 670.)
* wlnir : float, optional, Primary wavelength of near-infrared light (nm), (default = 800.)
* solslp : float, optional, Slope of the soil line, (default = 1.166)
* solicpt : float, optional, Intercept of the soil line, (default = 0.042)
* getlirf : boolean, optional, If True, compute log inverse reflectance, (default = True)
* getcrrf : boolean, optional, If True, compute continuum removed spectra, (default = True)

The `VegSpec` class constructor will automatically compute all spectral vegetation indices and spectral data pretreatment transformations unless optional arguments (i.e., `getlirf` and `getcrrf`) are changed to `False`. If increased computational efficiency is needed, consider changing `getcrrf` to `False`.

### Access the results
```
for key in spectrum.indices.keys():
    print(key+',{:f}'.format(spectrum.indices[key]))
```

`print(spectrum.rfd1)`

The VegSpec class stores the computational results in the following attribute variables:
* rfd1 : numpy.ndarray, A 1D array of the Savitsky-Golay first derivative of rf
* rfd2 : numpy.ndarray, A 1D array of the Savitsky-Golay second derivative of rf
* lirf : numpy.ndarray, A 1D array of the logarithm of inverse rf (log10(1/rf))
* lirfd1 : numpy.ndarray, A 1D array of the Savitsky-Golay first derivative of lirf
* lirfd2 : numpy.ndarry, A 1D array of the Savitsky-Golay second derivative of lirf
* crrf : numpy.ndarray, A 1D array of the continuum removal of rf
* indices : dict, A dictionary of 145+ published spectral vegetation indices

For further information on the keys of the `indices` attribute, see the documentation [here](http://github.com/kthorp/vegspec/tree/main/docs/document.pdf).

## Test suite and further examples
A test suite and further example scripts for using the vegspec package are [here](https://github.com/kthorp/vegspec/tree/main/tests).

[test0](https://github.com/kthorp/vegspec/tree/main/tests/test0) - Demonstrates the computation of spectral transformations for one example spectrum, collected from a cotton leaf at Maricopa, Arizona.

[test1](https://github.com/kthorp/vegspec/tree/main/tests/test1) - Demonstrates computation of spectral vegetation indices for cotton leaf spectra collected at Maricopa, Arizona in 2021 and 2022 and correlates the indices with leaf chlorophyll.

[test2](https://github.com/kthorp/vegspec/tree/main/tests/test2) - Demonstrates computation of spectral vegetation indices for cotton leaf spectra collected at Maricopa, Arizona in 2019 and 2020 and correlates the indices with leaf chlorophyll and leaf nitrogen.

[test3](https://github.com/kthorp/vegspec/tree/main/tests/test3) - Demonstrates computation of spectral vegetation indices for wheat canopy spectra collected at Maricopa, Arizona in 2011 and 2012 and correlates the indices with leaf area index, canopy weight, and plant nitrogen.

[test4](https://github.com/kthorp/vegspec/tree/main/tests/test4) - Demonstrates computation of spectral vegetation indices for cotton canopy spectra collected at Maricopa, Arizona in 2010, 2011, and 2012 and correlates the indices with leaf chlorophyll, leaf water thickness, specific leaf weight, and leaf area index.

[test5](https://github.com/kthorp/vegspec/tree/main/tests/test5) - Demonstrates computation of spectral vegetation indices for lesquerella canopy spectra collected at Maricopa, Arizona in 2008 and 2009 and correlates the indices with canopy weight, flower count, silique count, and silique dry weight.

[test6](https://github.com/kthorp/vegspec/tree/main/tests/test6) - Demonstrates computation of spectral vegetation indices for maize canopy spectra collected at Ames, Iowa in 2004 and correlates the indices with plant stand density.

[test7](https://github.com/kthorp/vegspec/tree/main/tests/test7) - Demonstrates computation of spectral vegetation indices for soybean canopy spectra collected at Urbana, Illinois in 2001 and correlates the indices with vegetation and weed cover.

## Further information
The vegspec package is further described in the following article:

Thorp, K. R., 2024. vegspec: A compilation of spectral vegetation indices and transformations in Python. SoftwareX. In review.


Also, the vegspec package was used to conduct the following research:

Thorp, K. R., Thompson, A. L., Herritt, M. T., 2024. Phenotyping cotton leaf chlorophyll via proximal hyperspectral reflectance sensing, spectral vegetation indices, and machine learning. Frontiers in Plant Science. In prep.
