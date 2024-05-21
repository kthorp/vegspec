import vegspec as vs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

filepath = './2021_2022_Cotton_Chlorophyll_LeafReflectance_PlotAvg.csv'
data = pd.read_csv(filepath,na_values='NA')
print(data)
print(data.shape)
print(data.iloc[0].values[-2151:])

wvln = list(range(350,2501))
spec = []

#Load spectra and compute all transformations and indices
for i in list(range(data.shape[0])):
    spec.append(vs.VegSpec(wvln,data.iloc[i].values[-2151:]))
    print(i)

#Analyze vegetation indices
fout = open('spectralanalysis.csv','w')

xlab = ['Chl $a$ ($\mu$g cm$^{-2}$)','Chl $b$ ($\mu$g cm$^{-2}$)',
        'Chl $a+b$ ($\mu$g cm$^{-2}$)','Chl $a$ (mg g$^{-1}$)',
        'Chl $b$ (mg g$^{-1}$)','Chl $a+b$ (mg g$^{-1}$)']
xval = [18.0,3.0,21.0,4.5,0.5,5.5]

for j, iname in enumerate(spec[0].indices.keys()):
    index = list()
    for s in spec:
        index.append(s.indices[iname])

    fig = plt.figure(figsize=(6.25,9.00),facecolor='w')
    plt.subplots_adjust(left=0.12, right=0.98, bottom=0.05, top=0.99,
                        hspace=0.30, wspace=0.30)

    string=iname + ','
    for i, x in enumerate(['ChlAugcm2','ChlBugcm2','ChlABugcm2',
                           'ChlAmgg','ChlBmgg','ChlABmgg']):
        xy = np.stack((data[x],index))
        xy = xy[:,~np.isnan(xy).any(axis=0)]

        #Compute correlation between measured data and indices
        slope, intercept, r, p, se = stats.linregress(xy[0],xy[1])
        string+='%f,'%(r*r) #Rsquared
        y1 = slope * min(xy[0]) + intercept
        y2 = slope * max(xy[0]) + intercept

        #Develop linear regression model and compute RMSE
        slope2, intercept2, r2, p2, se2 = stats.linregress(xy[1],xy[0])
        estimate = slope2 * xy[1] + intercept2
        rmse1 = np.sqrt(np.mean((xy[0]-estimate)*(xy[0]-estimate)))
        rmse2 = rmse1/np.mean(xy[0])*100.
        string+='%f,'%rmse1
        string+='%f,'%rmse2

        #print(xy[0])
        #print(xy[1])
        #print(slope2)
        #print(intercept2)
        #print(estimate)
        #print(rmse1)
        #print(rmse2)

        lbx = min(xy[0])
        ubx = max(xy[0])
        lby = min(xy[1])
        uby = max(xy[1])
        font= 10
        pad = 0.05
        lbx = lbx-(ubx-lbx)*pad
        ubx = ubx+(ubx-lbx)*pad
        lby = lby-(uby-lby)*pad
        uby = uby+(uby-lby)*pad
        ax1 = fig.add_subplot(3,2,i+1)
        ax1.scatter(xy[0],xy[1], s=1, color='k')
        ax1.plot([min(xy[0]),max(xy[0])],[y1,y2],'-',color='k')
        ax1.set_xlabel(xlab[i], labelpad=1, fontsize=font)
        ax1.set_ylabel(iname, labelpad=1, fontsize=font)
        ax1.set_xlim([lbx,ubx])
        ax1.set_ylim([lby,uby])
        ax1.annotate('y = {:6.4f} + {:7.4f}'.format(slope,intercept),
                     (xval[i],lby+(uby-lby)*.93),size=font)
        ax1.annotate('r$^2$={:6.4f}'.format(r*r),(xval[i],lby+(uby-lby)*.85),
                     size=font)

    plt.show()
    fig.savefig('SpectralIndex%03d_%s.pdf'%(j+1,iname),dpi=300)
    plt.close(fig)
    fout.write(string+'\n')

fout.close()
