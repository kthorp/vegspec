"""
########################################################################
The test1.py module contains a function to analyze the vegetative
reflectance spectrum in sample_spectrum.csv.

The test1.py module contains the following:
    run - function to run vegspec for a sample cotton leaf spectrum, collected in a field study at Maricopa, Arizona

01/24/2024 Script developed for running vegspec for sample_spectrum.csv
########################################################################
"""

import vegspec as vs
import os

def run():
    """Analyze sample vegetative reflectance spectrum"""

    #Get the module directory
    module_dir = os.path.dirname(os.path.abspath(__file__))

    #Load data
    f = open(os.path.join(module_dir,'sample_spectrum.csv'))
    lines = f.readlines()
    f.close()
    wl = []
    rf = []
    for line in lines[1:]:
        line = line.strip().split(',')
        wl.append(float(line[0]))
        rf.append(float(line[1]))

    #Analyze spectra with vegspec
    spectrum = vs.VegSpec(wl,rf)

    #Output vegetation index results
    f = open('indices.csv','w')
    for key in spectrum.indices.keys():
        f.write(key+',{:f}\n'.format(spectrum.indices[key]))
    f.close()

    #Output derivatives
    f = open('derivatives.csv','w')
    for i in range(len(spectrum.rfd1)):
        f.write('{:f},{:f}\n'.format(spectrum.rfd1[i],spectrum.rfd2[i]))
    f.close()

if __name__ == '__main__':
    run()
