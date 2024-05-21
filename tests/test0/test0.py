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
import matplotlib.pyplot as plt

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

    #Output transformations
    f = open('transformations.csv','w')
    for i in range(len(spectrum.rfd1)):
        f.write('{:f},{:f},{:f},{:f},{:f},{:f}\n'.format(
                spectrum.rfd1[i],spectrum.rfd2[i],spectrum.lirf[i],
                spectrum.lirfd1[i],spectrum.lirfd2[i],spectrum.crrf[i]))
    f.close()

    #Plot data
    font=10
    fig = plt.figure(figsize=(6.25,9.00),facecolor='w')
    plt.subplots_adjust(left=0.12, right=0.99, bottom=0.05, top=0.99,
                        hspace=0.00, wspace=0.00)
    ax1 = fig.add_subplot(7,1,1)
    ax1.plot(wl,rf,'-',color='k')
    ax1.set_xticks([300,500,700,900,1100,1300,1500,1700,1900,2100,2300,2500])
    ax1.set_xticklabels('', rotation=0, fontsize=font)
    ax1.set_ylabel(r'$\rho$', labelpad=1, fontsize=font)
    ax1.get_yaxis().set_label_coords(-0.09,0.5)
    ax1.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5])
    ax1.set_yticklabels(['0.0','0.1','0.2','0.3','0.4','0.5'], rotation=0, fontsize=font)
    ax1.annotate('(a)',(260.,0.48),size=font)

    ax2 = fig.add_subplot(7,1,2)
    ax2.plot(wl,spectrum.rfd1,color='k')
    ax2.set_xticks([300,500,700,900,1100,1300,1500,1700,1900,2100,2300,2500])
    ax2.set_xticklabels('', rotation=0, fontsize=font)
    ax2.set_ylabel(r'$\rho^{\prime}$', labelpad=1, fontsize=font)
    ax2.get_yaxis().set_label_coords(-0.09,0.5)
    ax2.set_yticks([-0.005,0.000,0.005,0.010])
    ax2.set_yticklabels(['-5e-3','0','5e-3','10e-3'], rotation=0, fontsize=font)
    ax2.annotate('(b)',(260.,0.0095),size=font)

    ax3 = fig.add_subplot(7,1,3)
    ax3.plot(wl,spectrum.rfd2,color='k')
    ax3.set_xticks([300,500,700,900,1100,1300,1500,1700,1900,2100,2300,2500])
    ax3.set_xticklabels('', rotation=0, fontsize=font)
    ax3.set_ylabel(r'$\rho^{\prime\prime}$', labelpad=1, fontsize=font)
    ax3.get_yaxis().set_label_coords(-0.09,0.5)
    ax3.set_yticks([-0.0002,0.0000,0.0002,0.0004,0.0006])
    ax3.set_yticklabels(['-2e-4','0','2e-4','4e-4','6e-4'], rotation=0, fontsize=font)
    ax3.annotate('(c)',(260.,0.00058),size=font)

    ax4 = fig.add_subplot(7,1,4)
    ax4.plot(wl,spectrum.lirf,color='k')
    ax4.set_xticks([300,500,700,900,1100,1300,1500,1700,1900,2100,2300,2500])
    ax4.set_xticklabels('', rotation=0, fontsize=font)
    ax4.set_ylabel(r'$\log_{10} \rho^{-1}$', labelpad=1, fontsize=font)
    ax4.get_yaxis().set_label_coords(-0.09,0.5)
    ax4.set_yticks([0.4,0.6,0.8,1.0,1.2,1.4])
    ax4.set_yticklabels(['0.4','0.6','0.8','1.0','1.2','1.4'], rotation=0, fontsize=font)
    ax4.annotate('(d)',(260.,1.31),size=font)

    ax5 = fig.add_subplot(7,1,5)
    ax5.plot(wl,spectrum.lirfd1,color='k')
    ax5.set_xticks([300,500,700,900,1100,1300,1500,1700,1900,2100,2300,2500])
    ax5.set_xticklabels('', rotation=0, fontsize=font)
    ax5.set_ylabel(r'$(\log_{10} \rho^{-1})^{\prime}$', labelpad=1, fontsize=font)
    ax5.get_yaxis().set_label_coords(-0.09,0.5)
    ax5.set_yticks([-0.03,-0.02,-0.01,0.00,0.01,0.02])
    ax5.set_yticklabels(['-3e-2','-2e-2','-1e-2','0','1e-2','2e-2'], rotation=0, fontsize=font)
    ax5.annotate('(e)',(260.,0.018),size=font)

    ax6 = fig.add_subplot(7,1,6)
    ax6.plot(wl,spectrum.rfd2,color='k')
    ax6.set_xticks([300,500,700,900,1100,1300,1500,1700,1900,2100,2300,2500])
    ax6.set_xticklabels('', rotation=0, fontsize=font)
    ax6.set_ylabel(r'$(\log_{10} \rho^{-1})^{\prime\prime}$', labelpad=1, fontsize=font)
    ax6.get_yaxis().set_label_coords(-0.09,0.5)
    ax6.set_yticks([-0.0002,0.0000,0.0002,0.0004,0.0006])
    ax6.set_yticklabels(['-2e-4','0','2e-4','4e-4','6e-4'], rotation=0, fontsize=font)
    ax6.annotate('(f)',(260.,0.00057),size=font)

    ax7 = fig.add_subplot(7,1,7)
    ax7.plot(wl,spectrum.crrf,color='k')
    ax7.set_xticks([300,500,700,900,1100,1300,1500,1700,1900,2100,2300,2500])
    ax7.set_xticklabels(['300','500','700','900','1100','1300','1500','1700','1900','2100','2300','2500'], rotation=0, fontsize=font)
    ax7.set_xlabel('Wavelength (nm)', labelpad=1, fontsize=font)
    ax7.set_ylabel(r'$\rho_{CR}$', labelpad=1, fontsize=font)
    ax7.get_yaxis().set_label_coords(-0.09,0.5)
    ax7.set_yticks([0.2,0.4,0.6,0.8,1.0])
    ax7.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'], rotation=0, fontsize=font)
    ax7.annotate('(g)',(260.,0.93),size=font)

    plt.show()
    fig.savefig('transformations.pdf',dpi=300)
    plt.close(fig)

if __name__ == '__main__':
    run()
