"""
########################################################################
The vegspec.py module contains the VegSpec class, which defines
functions for analysis of one vegetative reflectance spectrum, including
calculations of spectral derivatives, log inverse reflectance, continuum
removal, and more than 145 published spectral vegetation indices.

04/06/2022 Initial Python functions developed by Kelly Thorp
12/02/2022 Finalized code for release in the vegspec Python package
01/23/2023 Added more spectral vegetation indices
05/09/2024 Finalized code for first SoftwareX published release
########################################################################
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
from scipy.optimize import curve_fit
import math
import time

class VegSpec:
    """A class for analyzing vegetative spectral reflectance data.

    Manages computations of spectral derivatives, log inverse
    reflectance, continuum removal, and more than 145 published spectral
    vegetation indices.

    Attributes
    ----------
    wl : numpy.ndarray
        A 1D array of wavelengths (nm) corresponding to the data in rf
    rf : numpy.ndarray
        A 1D array of reflectance factors for a single spectral scan
    bndwdth : numpy.ndarray
        A 1D array of bandwidths (nm) corresponding to the data in rf
    NaN : float
        Simplified representation of float('NaN')
    wlblu : float
        Primary wavelength of visible blue light (nm) (default = 480.)
    wlgrn : float
        Primary wavelength of visible green light (nm) (default = 550.)
    wlred : float
        Primary wavelength of visible red light (nm) (default = 670.)
    wlnir : float
        Primary wavelength of near-infrared light (nm) (default = 800.)
    solslp : float
        Slope of the soil line (default = 1.166, Huete et al. (1984))
    solicpt : float
        Intercept of soil line (default = 0.042, Huete et al. (1984))
    rfd1 : numpy.ndarray
        A 1D array of the Savitsky-Golay first derivative of rf
    rfd2 : numpy.ndarray
        A 1D array of the Savitsky-Golay second derivative of rf
    lirf : numpy.ndarray
        A 1D array of the logarithm of inverse rf (log10(1/R))
    lirfd1 : numpy.ndarray
        A 1D array of the Savitsky-Golay first derivative of lirf
    lirfd2 : numpy.ndarry
        A 1D array of the Savitsky-Golay second derivative of lirf
    crrf : numpy.ndarray
        A 1D array of the continuum removal of rf
    indices : dict
        A dictionary of 145+ published spectral vegetation indices
    duration : float
        Wall-clock time (seconds) required for computation
    """

    def __init__(self,wl,rf,fs1=7,po1=2,fs2=15,po2=2,
                 wlblu=480.,wlgrn=550.,wlred=670.,wlnir=800.,
                 solslp=1.166,solicpt=0.042,
                 getlirf=True,getcrrf=True):
        """Initialize and compute the VegSpec class attributes.

        Parameters
        ----------
        wl : list
            Wavelengths (nm) corresponding to the data in rf
        rf : list
            Reflectance factors (0 to 1) for a single spectral scan
        fs1 : int, optional
            Filter size for Savitsky-Golay first derivatives
            (default = 7)
        po1 : int, optional
            Polynomial order for Savitsky-Golay first derivatives
            (default = 2)
        fs2 : int, optional
            Filter size for Savitsky-Golay second derivatives
            (default = 15)
        po2 : int, optional
            Polynomial order for Savitsky-Golay second derivatives
            (default = 2)
        wlblu : float, optional
            Primary wavelength of visible blue light (nm)
            (default = 480.)
        wlgrn : float, optional
            Primary wavelength of visible green light (nm)
            (default = 550.)
        wlred : float, optional
            Primary wavelength of visible red light (nm)
            (default = 670.)
        wlnir : float, optional
            Primary wavelength of near-infrared light (nm)
            (default = 800.)
        solslp : float, optional
            Slope of the soil line
            (default = 1.166, Huete et al. (1984))
        solicpt : float, optional
            Intercept of the soilline
            (default = 0.042, Huete et al. (1984))
        getlirf : boolean, optional
            If True, compute log inverse reflectance
            (default = True)
        getcrrf : boolean, optional
            If True, compute continuum removed spectra
            (default = True)
        """

        start = time.time()
        if len(wl) != len(rf):
            raise ValueError('Input lists must have same length.')
        else:
            self.wl = np.array(wl).astype(float)
            self.rf = np.array(rf).astype(float)
        if min(self.wl)<250. or max(self.wl)>2500.:
            raise ValueError('wl data must range from 250 to 2500 nm.')
        if min(self.rf)<0. or max(self.rf)>1.:
            raise ValueError('rf data must range from 0 to 1.')
        bndwdth = []
        for i in range(len(wl)):
            if i in [0]:
                bndwdth.append(wl[1]-wl[0])
            elif i in [len(wl)-1]:
                bndwdth.append(wl[-1]-wl[-2])
            else:
                bndwdth.append((wl[i]-wl[i-1])/2.+(wl[i+1]-wl[i])/2.)
        self.bndwdth = np.array(bndwdth)
        self.NaN = float('NaN')
        self.wlblu = wlblu
        self.wlgrn = wlgrn
        self.wlred = wlred
        self.wlnir = wlnir
        self.solslp = solslp
        self.solicpt = solicpt
        self.rfd1 = self._derivative1(fs1,po1)
        self.rfd2 = self._derivative2(fs2,po2)
        self.lirf = self._loginvrf()
        self.lirfd1 = self._loginvrfd1(fs1,po1)
        self.lirfd2 = self._loginvrfd2(fs2,po2)
        self.crrf = self._contremov()
        self.indices = {}
        self.indices.update({'BRSR':   self._BRSR()   })
        self.indices.update({'JSR':    self._JSR()    })
        self.indices.update({'NDVI':   self._NDVI()   })
        self.indices.update({'PVI':    self._PVI()    })
        self.indices.update({'WLREIP': self._WLREIP() })
        self.indices.update({'DVI':    self._DVI()    })
        self.indices.update({'NDVI2':  self._NDVI2()  })
        self.indices.update({'WLREIP2':self._WLREIP2()})
        self.indices.update({'SAVI':   self._SAVI()   })
        self.indices.update({'TSAVI':  self._TSAVI()  })
        self.indices.update({'WDVI':   self._WDVI()   })
        self.indices.update({'MSI':    self._MSI()    })
        self.indices.update({'BD':     self._BD()     })
        self.indices.update({'BDR':    self._BDR()    })
        self.indices.update({'SAVI2':  self._SAVI2()  })
        self.indices.update({'WLREIPG':self._WLREIPG()})
        self.indices.update({'WLCWMRG':self._WLCWMRG()})
        self.indices.update({'TSAVI2': self._TSAVI2() })
        self.indices.update({'CPSR1':  self._CPSR1()  })
        self.indices.update({'CPSR2':  self._CPSR2()  })
        self.indices.update({'CPSR3':  self._CPSR3()  })
        self.indices.update({'PRI':    self._PRI()    })
        self.indices.update({'GEMI':   self._GEMI()   })
        self.indices.update({'BMSR':   self._BMSR()   })
        self.indices.update({'BMLSR':  self._BMLSR()  })
        self.indices.update({'BMDVI':  self._BMDVI()  })
        self.indices.update({'PSR':    self._PSR()    })
        self.indices.update({'PD':     self._PD()     })
        self.indices.update({'WLPD':   self._WLPD()   })
        self.indices.update({'VSR':    self._VSR()    })
        self.indices.update({'VDR':    self._VDR()    })
        self.indices.update({'CRSR1':  self._CRSR1()  })
        self.indices.update({'CRSR2':  self._CRSR2()  })
        self.indices.update({'CRSR3':  self._CRSR3()  })
        self.indices.update({'CRSR4':  self._CRSR4()  })
        self.indices.update({'CRSR5':  self._CRSR5()  })
        self.indices.update({'FSUM':   self._FSUM()   })
        self.indices.update({'DREIP':  self._DREIP()  })
        self.indices.update({'NDVI3':  self._NDVI3()  })
        self.indices.update({'GSUM1':  self._GSUM1()  })
        self.indices.update({'GSUM2':  self._GSUM2()  })
        self.indices.update({'NLI':    self._NLI()    })
        self.indices.update({'CAR':    self._CAR()    })
        self.indices.update({'CARI':   self._CARI()   })
        self.indices.update({'NPCI':   self._NPCI()   })
        self.indices.update({'EGFN':   self._EGFN()   })
        self.indices.update({'MSAVI1': self._MSAVI1() })
        self.indices.update({'MSAVI2': self._MSAVI2() })
        self.indices.update({'ESUM1':  self._ESUM1()  })
        self.indices.update({'ESUM2':  self._ESUM2()  })
        self.indices.update({'NDPI':   self._NDPI()   })
        self.indices.update({'SIPI':   self._SIPI()   })
        self.indices.update({'SRPI':   self._SRPI()   })
        self.indices.update({'NPQI':   self._NPQI()   })
        self.indices.update({'RDVI':   self._RDVI()   })
        self.indices.update({'MSR':    self._MSR()    })
        self.indices.update({'PRI2':   self._PRI2()   })
        self.indices.update({'NDWI':   self._NDWI()   })
        self.indices.update({'GTSR1':  self._GTSR1()  })
        self.indices.update({'GTSR2':  self._GTSR2()  })
        self.indices.update({'GNDVI':  self._GNDVI()  })
        self.indices.update({'OSAVI':  self._OSAVI()  })
        self.indices.update({'WI':     self._WI()     })
        self.indices.update({'WNR':    self._WNR()    })
        self.indices.update({'PSSRA':  self._PSSRA()  })
        self.indices.update({'PSSRB':  self._PSSRB()  })
        self.indices.update({'PSSRC':  self._PSSRC()  })
        self.indices.update({'PSNDA':  self._PSNDA()  })
        self.indices.update({'PSNDB':  self._PSNDB()  })
        self.indices.update({'PSNDC':  self._PSNDC()  })
        self.indices.update({'DSR1':   self._DSR1()   })
        self.indices.update({'DSR2':   self._DSR2()   })
        self.indices.update({'DNDR':   self._DNDR()   })
        self.indices.update({'DDR1':   self._DDR1()   })
        self.indices.update({'DDR2':   self._DDR2()   })
        self.indices.update({'GMSR':   self._GMSR()   })
        self.indices.update({'PSRI':   self._PSRI()   })
        self.indices.update({'TVI':    self._TVI()    })
        self.indices.update({'MCARI':  self._MCARI()  })
        self.indices.update({'MOR':    self._MOR()    })
        self.indices.update({'ZTSR1':  self._ZTSR1()  })
        self.indices.update({'CI':     self._CI()     })
        self.indices.update({'ZTDR1':  self._ZTDR1()  })
        self.indices.update({'ZTSR2':  self._ZTSR2()  })
        self.indices.update({'CAI':    self._CAI()    })
        self.indices.update({'ARI':    self._ARI()    })
        self.indices.update({'MND1':   self._MND1()   })
        self.indices.update({'MND2':   self._MND2()   })
        self.indices.update({'MND3':   self._MND3()   })
        self.indices.update({'MND4':   self._MND4()   })
        self.indices.update({'CAINT':  self._CAINT()  })
        self.indices.update({'ZTSUM':  self._ZTSUM()  })
        self.indices.update({'PRI3':   self._PRI3()   })
        self.indices.update({'ZTDPR1': self._ZTDPR1() })
        self.indices.update({'ZTDPR2': self._ZTDPR2() })
        self.indices.update({'ZTDP21': self._ZTDP21() })
        self.indices.update({'ZTDP22': self._ZTDP22() })
        self.indices.update({'GI':     self._GI()     })
        self.indices.update({'ZTSR3':  self._ZTSR3()  })
        self.indices.update({'ZTSR4':  self._ZTSR4()  })
        self.indices.update({'ZTSR5':  self._ZTSR5()  })
        self.indices.update({'ZTSR6':  self._ZTSR6()  })
        self.indices.update({'VARI':   self._VARI()   })
        self.indices.update({'CRI500': self._CRI500() })
        self.indices.update({'CRI700': self._CRI700() })
        self.indices.update({'TCARI':  self._TCARI()  })
        self.indices.update({'TOR':    self._TOR()    })
        self.indices.update({'EVI':    self._EVI()    })
        self.indices.update({'NDNI':   self._NDNI()   })
        self.indices.update({'NDLI':   self._NDLI()   })
        self.indices.update({'MSR2':   self._MSR2()   })
        self.indices.update({'SMNDVI': self._SMNDVI() })
        self.indices.update({'GRRGM':  self._GRRGM()  })
        self.indices.update({'GRRREM': self._GRRREM() })
        self.indices.update({'DPI':    self._DPI()    })
        self.indices.update({'SRWI':   self._SRWI()   })
        self.indices.update({'MTCI':   self._MTCI()   })
        self.indices.update({'WDRVI':  self._WDRVI()  })
        self.indices.update({'MCARI1': self._MCARI1() })
        self.indices.update({'MCARI2': self._MCARI2() })
        self.indices.update({'MTVI1':  self._MTVI1()  })
        self.indices.update({'MTVI2':  self._MTVI2()  })
        self.indices.update({'DD':     self._DD()     })
        self.indices.update({'LCA':    self._LCA()    })
        self.indices.update({'RGI':    self._RGI()    })
        self.indices.update({'BGI1':   self._BGI1()   })
        self.indices.update({'BGI2':   self._BGI2()   })
        self.indices.update({'BRI1':   self._BRI1()   })
        self.indices.update({'BRI2':   self._BRI2()   })
        self.indices.update({'WLREIPE':self._WLREIPE()})
        self.indices.update({'RVIOPT': self._RVIOPT() })
        self.indices.update({'SPVI':   self._SPVI()   })
        self.indices.update({'MMR':    self._MMR()    })
        self.indices.update({'TCI':    self._TCI()    })
        self.indices.update({'EVI2':   self._EVI2()   })
        self.indices.update({'DDN':    self._DDN()    })
        self.indices.update({'CVI':    self._CVI()    })
        self.indices.update({'WUTCARI':self._WUTCARI()})
        self.indices.update({'WUOSAVI':self._WUOSAVI()})
        self.indices.update({'WUMCARI':self._WUMCARI()})
        self.indices.update({'WUMSR':  self._WUMSR()  })
        self.indices.update({'WUTOR':  self._WUTOR()  })
        self.indices.update({'WUMOR':  self._WUMOR()  })
        self.indices.update({'DCNI':   self._DCNI()   })
        self.indices.update({'TGI':    self._TGI()    })
        self.indices.update({'WDRVI2': self._WDRVI2() })
        self.indices.update({'AIVI':   self._AIVI()   })
        self.indices.update({'DND':    self._DND()    })
        end = time.time()
        self.duration = end - start

    #Spectral transformations
    def _derivative1(self,fs1,po1):
        """Compute first derivative using a Savitsky-Golay filter"""
        return savgol_filter(self.rf,fs1,po1,1)

    def _derivative2(self,fs2,po2):
        """Compute second derivative using a Savitsky-Golay filter"""
        return savgol_filter(self.rf,fs2,po2,2)

    def _loginvrf(self):
        """Compute log10(1/R)
        (Yoder & Pettigrew-Crosby, 1995; Blackburn, 1998))"""
        clip = np.clip(self.rf,0.0000000000001,100000.)
        return np.log10(np.reciprocal(clip))

    def _loginvrfd1(self,fs1,po1):
        """Compute the first derivative of log10(1/R)
        (Yoder & Pettigrew-Crosby, 1995; Blackburn, 1998)"""
        return savgol_filter(self.lirf,fs1,po1,1)

    def _loginvrfd2(self,fs2,po2):
        """Compute the second derivative of log10(1/R)
        (Yoder & Pettigrew-Crosby, 1995; Blackburn, 1998)"""
        return savgol_filter(self.lirf,fs2,po2,2)

    def _contremov(self):
        """Compute the continuum removed spectra (Kokaly & Clark, 1999;
        Curran et al., 2001; Huang et al., 2004)"""
        points = np.transpose(np.stack((self.wl,self.rf),axis=0))
        hull = ConvexHull(points)
        continuum = list()
        whichhull = list()
        for i in list(range(len(self.wl))):
            whichhull[:] = list()
            for simplex in hull.simplices:
                if min(simplex) <= i <= max(simplex):
                    p1 = points[simplex[0]]
                    p2 = points[simplex[1]]
                    coef = np.polyfit([p1[0],p2[0]],[p1[1],p2[1]],1)
                    whichhull.append(coef[0]*self.wl[i]+coef[1])
            continuum.append(max(whichhull))
        #plot = plt.plot(self.wl,self.rf)
        #plt.plot(self.wl,np.array(continuum), 'r-')
        #plt.show()
        return np.divide(self.rf,np.array(continuum))

    #Spectral indices
    def _BRSR(self):
        """Compute Birth simple ratio (BRSR) (Birth & McVey, 1968)"""
        R675 = np.interp(675.,self.wl,self.rf,self.NaN,self.NaN)
        R745 = np.interp(745.,self.wl,self.rf,self.NaN,self.NaN)
        return R745/R675

    def _JSR(self):
        """Compute Jordan simple ratio (JSR) (Jordan, 1969)"""
        R675 = np.interp(675.,self.wl,self.rf,self.NaN,self.NaN)
        R800 = np.interp(800.,self.wl,self.rf,self.NaN,self.NaN)
        return R800/R675

    def _NDVI(self):
        """Compute the Normalized Difference Vegetation Index (NDVI)
        (Rouse et al., 1973)"""
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        return (Rnir-Rred)/(Rnir+Rred)

    def _PVI(self):
        """Compute the Perpendicular Vegetation Index (PVI)
        Original PVI publication (Richardson and Wiegand, 1977)
        Algebraic basis for PVI (Jackson et al., 1980)
        Soil line parameters (Heute et al., 1984)"""
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        a = self.solslp
        b = self.solicpt
        return (Rnir-a*Rred-b)/math.sqrt(1.+a*a)

    def _WLREIP(self):
        """Identify the wavelength (nm) of the red edge inflection point
        (Collins, 1978; Horler et al., 1983)"""
        widx = np.where(np.logical_and(self.wl>=680.,self.wl<=750.))
        if len(widx[0])==0: return self.NaN
        return self.wl[widx][np.argmax(self.rfd1[widx])]

    def _DVI(self):
        """Compute the Difference Vegetation Index (DVI)
        (Tucker, 1979)"""
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        return Rnir-Rred

    def _NDVI2(self):
        """Compute the Normalized Difference Vegetation Index 2 (NDVI2)
        (Tucker, 1979)"""
        Rgrn = np.interp(self.wlgrn,self.wl,self.rf,self.NaN,self.NaN)
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        return (Rgrn-Rred)/(Rgrn+Rred)

    def _WLREIP2(self):
        """Compute the wavelength (nm) of the red edge inflection point
        (Guyot & Baret, 1988; Cho & Skidmore, 2006)"""
        R670 = np.interp(670.,self.wl,self.rf,self.NaN,self.NaN)
        R700 = np.interp(700.,self.wl,self.rf,self.NaN,self.NaN)
        R740 = np.interp(740.,self.wl,self.rf,self.NaN,self.NaN)
        R780 = np.interp(780.,self.wl,self.rf,self.NaN,self.NaN)
        return 700. + 40.*(((R670+R780)/2.-R700)/(R740-R700))

    def _SAVI(self, L=0.50):
        """Compute the Soil-Adjusted Vegetation Index (SAVI)
        (Huete, 1988)"""
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        return (1+L)*(Rnir-Rred)/(Rnir+Rred+L)

    def _TSAVI(self):
        """Compute the Transformed Soil Adjusted Vegetation Index 
        (TSAVI)
        Original TSAVI publication (Baret et al., 1989)
        Soil line parameters (Heute et al., 1984)"""
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        a = self.solslp
        b = self.solicpt
        return a*(Rnir-a*Rred-b)/(Rred+a*Rnir-a*b)

    def _WDVI(self):
        """Compute the Weighted Difference Vegetation Index (WDVI)
        WDVI formulation (Clevers, 1989)
        Soil line slope parameter (Heute et al., 1984)"""
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        C = self.solslp
        return Rnir-C*Rred

    def _MSI(self):
        """Compute the Moisture Stress Index (MSI)
        (Hunt & Rock, 1989)"""
        R820 = np.interp(820.,self.wl,self.rf,self.NaN,self.NaN)
        R1600 = np.interp(1600.,self.wl,self.rf,self.NaN,self.NaN)
        return R1600/R820

    def _BD(self):
        """Compute the Boochs derivative (BD) (Boochs et al., 1990)"""
        return np.interp(703.,self.wl,self.rfd1,self.NaN,self.NaN)

    def _BDR(self):
        """Compute the Boochs derivative ratio (BDR)
        (Boochs et al., 1990)"""
        D703 = np.interp(703.,self.wl,self.rfd1,self.NaN,self.NaN)
        widx = np.where(np.logical_and(self.wl>=680.,self.wl<=750.))
        if len(widx[0])==0: return self.NaN
        Dmax = np.amax(self.rfd1[widx])
        return D703/Dmax

    def _SAVI2(self):
        """Compute the Soil-Adjusted Vegetation Index 2 (SAVI2)
        Original SAVI2 publication (Major et al., 1990)
        Soil line parameters (Huete et al., 1984)"""
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        a = self.solslp
        b = self.solicpt
        return Rnir/(Rred+b/a)

    def _WLREIPG(self):
        """Compute the wavelength (nm) of the red edge inflection point
        via Gaussian fit (WLREIPG) (Miller et al., 1990)"""
        class IGfunc:
            def __init__(self,R0):
                self.R0 = R0
            def IG(self,lmbda,lmbda0,sgma,Rs):
                R=list()
                for l in lmbda:
                    sqrterm = math.pow(lmbda0-l,2.)
                    expo = math.exp((-1.*sqrterm)/(2.*sgma*sgma))
                    R.append(Rs-(Rs-self.R0)*expo)
                return np.array(R)
        widx = np.where(np.logical_and(self.wl>=660.,self.wl<=810.))
        xdata = self.wl[widx]
        ydata = self.rf[widx]
        #Initial guesses as per Miller al. (1990) page 1758
        Rs = np.amax(ydata)
        lmb0 = xdata[np.argmin(ydata)]
        widx2 = np.where(np.logical_and(xdata>=lmb0-10.,xdata<=lmb0))
        R0 = np.mean(ydata[widx2])
        func = IGfunc(R0)
        p0 = np.array([lmb0,30.,Rs])
        popt, pcov = curve_fit(func.IG,xdata,ydata,p0)
        ymod = func.IG(xdata,popt[0],popt[1],popt[2])
        #print(popt[0],popt[1],popt[2],popt[0]+popt[1])
        #plt.plot(xdata,ydata,'-')
        #plt.plot(xdata,ymod,'--')
        #plt.show()
        return popt[0]+popt[1]

    def _WLCWMRG(self):
        """Compute the wavelength (nm) of the chlorophyll-well minimum
        reflectance via Gaussian fit (WLCWMRG) (Miller et al., 1990)"""
        class IGfunc:
            def __init__(self,R0):
                self.R0 = R0
            def IG(self,lmbda,lmbda0,sgma,Rs):
                R=list()
                for l in lmbda:
                    sqrterm = math.pow(lmbda0-l,2.)
                    expo = math.exp((-1.*sqrterm)/(2.*sgma*sgma))
                    R.append(Rs-(Rs-self.R0)*expo)
                return np.array(R)
        widx = np.where(np.logical_and(self.wl>=660.,self.wl<=810.))
        xdata = self.wl[widx]
        ydata = self.rf[widx]
        #Initial guesses as per Miller al. (1990) page 1758
        Rs = np.amax(ydata)
        lmb0 = xdata[np.argmin(ydata)]
        widx2 = np.where(np.logical_and(xdata>=lmb0-10.,xdata<=lmb0))
        R0 = np.mean(ydata[widx2])
        func = IGfunc(R0)
        p0 = np.array([lmb0,30.,Rs])
        popt, pcov = curve_fit(func.IG,xdata,ydata,p0)
        ymod = func.IG(xdata,popt[0],popt[1],popt[2])
        #print(popt[0],popt[1],popt[2])
        #plt.plot(xdata,ydata,'-')
        #plt.plot(xdata,ymod,'--')
        #plt.show()
        return popt[0]

    def _TSAVI2(self, X=0.08):
        """Compute the Transformed Soil Adjusted Vegetation Index 2
        (TSAVI2)
        Updated TSAVI publication (Baret & Guyot, 1991)
        Soil line parameters (Heute et al., 1984)"""
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        a = self.solslp
        b = self.solicpt
        return a*(Rnir-a*Rred-b)/(a*Rnir+Rred-a*b+X*(1.+a*a))

    def _CPSR1(self):
        """Compute the Chappelle simple ratio 1 (CPSR1)
        (Chappelle et al., 1992)"""
        R675 = np.interp(675.,self.wl,self.rf,self.NaN,self.NaN)
        R700 = np.interp(700.,self.wl,self.rf,self.NaN,self.NaN)
        return R675/R700

    def _CPSR2(self):
        """Compute the Chappelle simple ratio 2 (CPSR2)
        (Chappelle et al., 1992)"""
        R650 = np.interp(650.,self.wl,self.rf,self.NaN,self.NaN)
        R675 = np.interp(675.,self.wl,self.rf,self.NaN,self.NaN)
        R700 = np.interp(700.,self.wl,self.rf,self.NaN,self.NaN)
        return R675/(R650*R700)

    def _CPSR3(self):
        """Compute the Chappelle simple ratio 3 (CPSR3)
        (Chapelle et al., 1992)"""
        R500 = np.interp(500.,self.wl,self.rf,self.NaN,self.NaN)
        R760 = np.interp(760.,self.wl,self.rf,self.NaN,self.NaN)
        return R760/R500

    def _PRI(self):
        """Compute the Photochemical Reflectance Index (PRI)
        (Gamon et al., 1992)"""
        R531 = np.interp(531.,self.wl,self.rf,self.NaN,self.NaN)
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        return (R550-R531)/(R550+R531)

    def _GEMI(self):
        """Compute the Global Environment Monitoring Index (GEMI)
        (Pinty & Verstraete, 1992)"""
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        eta = 2.*(Rnir*Rnir-Rred*Rred)+1.5*Rnir+0.5*Rred
        eta = eta / (Rnir+Rred+0.5)
        return eta*(1.-0.25*eta)-(Rred-0.125)/(1.-Rred)

    def _BMSR(self):
        """Compute the Buschmann simple ratio (BMSR)
        (Buschmann & Nagel, 1993)"""
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        R800 = np.interp(800.,self.wl,self.rf,self.NaN,self.NaN)
        return R550/R800

    def _BMLSR(self):
        """Compute the Buschmann log simple ratio (BMLSR)
        (Buschmann & Nagel, 1993)"""
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        R800 = np.interp(800.,self.wl,self.rf,self.NaN,self.NaN)
        return math.log10(R800/R550)

    def _BMDVI(self):
        """Compute the Buchmann difference vegetation index (BMDVI)
        (Buschmann & Nagel, 1993)"""
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        R800 = np.interp(800.,self.wl,self.rf,self.NaN,self.NaN)
        return R800-R550

    def _PSR(self):
        """Compute the Penuelas simple ratio (PSR)
        (Penuelas et al., 1993)"""
        R970 = np.interp(970.,self.wl,self.rf,self.NaN,self.NaN)
        R900 = np.interp(900.,self.wl,self.rf,self.NaN,self.NaN)
        return R970/R900

    def _PD(self):
        """Compute the Penuelas derivative (PD)
        (Penuelas et al., 1993)"""
        widx = np.where(np.logical_and(self.wl>=900.,self.wl<=970.))
        if len(widx[0])==0: return self.NaN
        return np.amin(self.rfd1[widx])

    def _WLPD(self):
        """Identify the wavelength (nm) of the minimum first derivative
        in the near infrared region (WLPD) (Penuelas et al., 1993)"""
        widx = np.where(np.logical_and(self.wl>=900.,self.wl<=970.))
        if len(widx[0])==0: return self.NaN
        return self.wl[widx][np.argmin(self.rfd1[widx])]

    def _VSR(self):
        """Compute the Vogelmann simple ratio (VSR)
        (Vogelmann et al., 1993)"""
        R720 = np.interp(720.,self.wl,self.rf,self.NaN,self.NaN)
        R740 = np.interp(740.,self.wl,self.rf,self.NaN,self.NaN)
        return R740/R720

    def _VDR(self):
        """Compute the Vogelmann derivative ratio (VDR)
        (Vogelmann et al., 1993)"""
        D705 = np.interp(705.,self.wl,self.rfd1,self.NaN,self.NaN)
        D715 = np.interp(715.,self.wl,self.rfd1,self.NaN,self.NaN)
        return D715/D705

    def _CRSR1(self):
        """Compute the Carter simple ratio 1 (CRSR1) (Carter, 1994)"""
        R420 = np.interp(420.,self.wl,self.rf,self.NaN,self.NaN)
        R695 = np.interp(695.,self.wl,self.rf,self.NaN,self.NaN)
        return R695/R420

    def _CRSR2(self):
        """Compute the Carter simple ratio 2 (CRSR2) (Carter, 1994)"""
        R605 = np.interp(605.,self.wl,self.rf,self.NaN,self.NaN)
        R760 = np.interp(760.,self.wl,self.rf,self.NaN,self.NaN)
        return R605/R760

    def _CRSR3(self):
        """Compute the Carter simple ratio 3 (CRSR3) (Carter, 1994)"""
        R695 = np.interp(695.,self.wl,self.rf,self.NaN,self.NaN)
        R760 = np.interp(760.,self.wl,self.rf,self.NaN,self.NaN)
        return R695/R760

    def _CRSR4(self):
        """Compute the Carter simple ratio 4 (CRSR4) (Carter, 1994)"""
        R710 = np.interp(710.,self.wl,self.rf,self.NaN,self.NaN)
        R760 = np.interp(760.,self.wl,self.rf,self.NaN,self.NaN)
        return R710/R760

    def _CRSR5(self):
        """Compute the Carter simple ratio 5 (CRSR5) (Carter, 1994)"""
        R670 = np.interp(670.,self.wl,self.rf,self.NaN,self.NaN)
        R695 = np.interp(695.,self.wl,self.rf,self.NaN,self.NaN)
        return R695/R670

    def _FSUM(self):
        """Compute the area of the first derivative red edge peak from
        680 nm to 780 nm (FSUM) (Filella & Penuelas, 1994;
        Fillela et al., 1995)"""
        widx = np.where(np.logical_and(self.wl>=680.,self.wl<=780.))
        if len(widx[0])==0: return self.NaN
        return np.sum(self.rfd1[widx]*self.bndwdth[widx])

    def _DREIP(self):
        """Identify the amplitude of the first derivative at the red
        edge inflection point (DREIP) (Filella & Penuelas, 1994;
        Fillela et al., 1995)"""
        widx = np.where(np.logical_and(self.wl>=680.,self.wl<=780.))
        if len(widx[0])==0: return self.NaN
        return np.amax(self.rfd1[widx])

    def _NDVI3(self):
        """Compute the Normalized Difference Vegetation Index 3 (NDVI3)
        (Gitelson & Merzlyak, 1994)"""
        R705 = np.interp(705.,self.wl,self.rf,self.NaN,self.NaN)
        R750 = np.interp(750.,self.wl,self.rf,self.NaN,self.NaN)
        return (R750-R705)/(R750+R705)

    def _GSUM1(self):
        """Compute the sum of reflectance from 705 nm to 750 nm,
        normalized by reflectance at 705 nm (GSUM1)
        (Gitelson & Merzlyak, 1994)"""
        R705 = np.interp(705.,self.wl,self.rf,self.NaN,self.NaN)
        widx = np.where(np.logical_and(self.wl>=705.,self.wl<=750.))
        if len(widx[0])==0: return self.NaN
        return np.sum((self.rf[widx]/R705-1.)*self.bndwdth[widx])

    def _GSUM2(self):
        """Compute the sum of reflectance from 705 nm to 750 nm,
        normalized by reflectance at 555 nm (GSUM2)
        (Gitelson & Merzlyak, 1994)"""
        R555 = np.interp(555.,self.wl,self.rf,self.NaN,self.NaN)
        widx = np.where(np.logical_and(self.wl>=705.,self.wl<=750.))
        if len(widx[0])==0: return self.NaN
        return np.sum((self.rf[widx]/R555-1.)*self.bndwdth[widx])

    def _NLI(self):
        """Compute the Nonlinear Index (NLI) (Goel & Qin, 1994)"""
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        return (Rnir*Rnir-Rred)/(Rnir*Rnir+Rred)

    def _CAR(self):
        """Compute the Chlorophyll Absorption in Reflectance (CAR)
        (Kim et al., 1994)"""
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)*100.
        R670 = np.interp(670.,self.wl,self.rf,self.NaN,self.NaN)*100.
        R700 = np.interp(700.,self.wl,self.rf,self.NaN,self.NaN)*100.
        a = np.array([[700.-550.],[R700-R550]])
        b = np.array([[670.-550.],[R670-R550]])
        dot1 = np.dot(a.transpose(),a)[0][0]
        dot2 = np.dot(b.transpose(),b)[0][0]
        dot3 = np.dot(a.transpose(),b)[0][0]
        return math.sqrt((dot2*dot1-dot3*dot3)/dot1)

    def _CARI(self):
        """Compute the Chlorophyll Absorption Ratio Index (CARI)
        (Kim et al., 1994)"""
        R670 = np.interp(670.,self.wl,self.rf,self.NaN,self.NaN)*100.
        R700 = np.interp(700.,self.wl,self.rf,self.NaN,self.NaN)*100.
        CAR = self._CAR()
        return CAR*(R700/R670)

    def _NPCI(self):
        """Compute the Normalized Pigments Chlorophyll Ratio Index
        (NPCI) (Penuelas et al., 1994)"""
        R430 = np.interp(430.,self.wl,self.rf,self.NaN,self.NaN)
        R680 = np.interp(680.,self.wl,self.rf,self.NaN,self.NaN)
        return (R680-R430)/(R680+R430)

    def _EGFN(self):
        """Compute the Edge-Green First-derivative Normalized Difference
        Index (EGFN) (Penuelas et al., 1994)"""
        widx = np.where(np.logical_and(self.wl>=500.,self.wl<=600.))
        if len(widx[0])==0: return self.NaN
        dG = np.amax(self.rfd1[widx])
        widx = np.where(np.logical_and(self.wl>=680.,self.wl<=750.))
        if len(widx[0])==0: return self.NaN
        dRE = np.amax(self.rfd1[widx])
        return (dRE-dG)/(dRE+dG)

    def _MSAVI1(self):
        """Compute the Modified Soil Adjusted Vegetation Index 1 (MSAVI1)
        MSAVI formualtion (Qi et al., 1994)
        Soil line slope (Heute et al., 1984)"""
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        NDVI = self._NDVI()
        WDVI = self._WDVI()
        a = self.solslp
        L = 1.-2.*a*NDVI*WDVI
        return (1+L)*(Rnir-Rred)/(Rnir+Rred+L)

    def _MSAVI2(self):
        """Compute the Modified Soil Adjusted Vegetation Index 2 (MSAVI2)
        (Qi et al., 1994)"""
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        sqrterm = math.pow(2.*Rnir+1.,2.)
        return 0.5*(2.*Rnir+1.-math.sqrt(sqrterm-8.*(Rnir-Rred)))

    def _ESUM1(self):
        """Compute the area of the first derivative red edge peak from
        626 nm to 795 nm (ESUM1) (Elvidge & Chen, 1995)"""
        widx = np.where(np.logical_and(self.wl>=626.,self.wl<=795.))
        if len(widx[0])==0: return self.NaN
        return np.sum(np.absolute(self.rfd1[widx])*self.bndwdth[widx])

    def _ESUM2(self):
        """Compute the area of the second derivative red edge peaks from
        626 nm to 795 nm (ESUM2) (Elvidge & Chen, 1995)"""
        widx = np.where(np.logical_and(self.wl>=626.,self.wl<=795.))
        if len(widx[0])==0: return self.NaN
        return np.sum(np.absolute(self.rfd2[widx])*self.bndwdth[widx])

    def _NDPI(self):
        """Compute the Normalized Difference Pigment Index (NDPI)
        (Penuelas et al., 1995a) Photosynthetica"""
        R420 = np.interp(420.,self.wl,self.rf,self.NaN,self.NaN)
        R670 = np.interp(670.,self.wl,self.rf,self.NaN,self.NaN)
        return (R670-R420)/(R670+R420)

    def _SIPI(self):
        """Compute the Structure Independent Pigment Index (SIPI)
        (Penuelas et al., 1995a) Photosynthetica"""
        R445 = np.interp(445.,self.wl,self.rf,self.NaN,self.NaN)
        R680 = np.interp(680.,self.wl,self.rf,self.NaN,self.NaN)
        R800 = np.interp(800.,self.wl,self.rf,self.NaN,self.NaN)
        return (R800-R445)/(R800-R680)

    def _SRPI(self):
        """Compute the Simple Ratio Pigment Index (SRPI)
        (Penuelas et al., 1995b) IJRS"""
        R430 = np.interp(430.,self.wl,self.rf,self.NaN,self.NaN)
        R680 = np.interp(680.,self.wl,self.rf,self.NaN,self.NaN)
        return R430/R680

    def _NPQI(self):
        """Compute the Normalized Phaeophytinization Index (NPQI)
        (Penuelas et al., 1995b) IJRS"""
        R415 = np.interp(415.,self.wl,self.rf,self.NaN,self.NaN)
        R435 = np.interp(435.,self.wl,self.rf,self.NaN,self.NaN)
        return (R415-R435)/(R415+R435)

    def _RDVI(self):
        """Compute the Renormalized Difference Vegetation Index (RDVI)
        (Roujean & Breon, 1995)"""
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        return (Rnir-Rred)/math.sqrt(Rnir+Rred)

    def _MSR(self):
        """Compute the Modified Simple Ratio (MSR) (Chen, 1996)"""
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        return (Rnir/Rred-1.)/(math.sqrt(Rnir/Rred+1.))

    def _PRI2(self):
        """Compute the Photochemical Reflectance Index 2 (PRI2)
        (Fillela et al., 1996)"""
        R539 = np.interp(539.,self.wl,self.rf,self.NaN,self.NaN)
        R570 = np.interp(570.,self.wl,self.rf,self.NaN,self.NaN)
        return (R539-R570)/(R539+R570)

    def _NDWI(self):
        """Compute the Normalized Difference Water Index (NDWI)
        (Gao, 1996)"""
        R860 = np.interp(860.,self.wl,self.rf,self.NaN,self.NaN)
        R1240 = np.interp(1240.,self.wl,self.rf,self.NaN,self.NaN)
        return (R860-R1240)/(R860+R1240)

    def _GTSR1(self):
        """Compute the Gitelson simple ratio 1 (GTSR1)
        (Gitelson & Merzlyak, 1996)"""
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        R750 = np.interp(750.,self.wl,self.rf,self.NaN,self.NaN)
        return R750/R550

    def _GTSR2(self):
        """Compute the Gitelson simple ratio 2 (GTSR2)
        (Gitelson & Merzlyak, 1996)"""
        R700 = np.interp(700.,self.wl,self.rf,self.NaN,self.NaN)
        R750 = np.interp(750.,self.wl,self.rf,self.NaN,self.NaN)
        return R750/R700

    def _GNDVI(self):
        """Compute the Green Normalized Difference Vegetation Index
        (GNDVI) (Gitelson et al., 1996)"""
        Rgrn = np.interp(self.wlgrn,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        return (Rnir-Rgrn)/(Rnir+Rgrn)

    def _OSAVI(self):
        """Compute the Optimized Soil Adjusted Vegetation Index (OSAVI)
        (Rondeaux et al., 1996)"""
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        return (1.+0.16)*(Rnir-Rred)/(Rnir+Rred+0.16)

    def _WI(self):
        """Compute the Water Index (WI) (Penuelas et al., 1997)"""
        R900 = np.interp(900.,self.wl,self.rf,self.NaN,self.NaN)
        R970 = np.interp(970.,self.wl,self.rf,self.NaN,self.NaN)
        return R900/R970

    def _WNR(self):
        """Compute the WI NDVI ratio (Penuelas et al., 1997)"""
        WI = self._WI()
        NDVI = self._NDVI()
        return WI/NDVI

    def _PSSRA(self):
        """Compute the Pigment Specific Simple Ratio for chlA (PSSRA)
        (Blackburn, 1998a; 1998b)"""
        R680 = np.interp(680.,self.wl,self.rf,self.NaN,self.NaN)
        R800 = np.interp(800.,self.wl,self.rf,self.NaN,self.NaN)
        return R800/R680

    def _PSSRB(self):
        """Compute the Pigment Specific Simple Ratio for chlB (PSSRB)
        (Blackburn, 1998a; 1998b)"""
        R635 = np.interp(635.,self.wl,self.rf,self.NaN,self.NaN)
        R800 = np.interp(800.,self.wl,self.rf,self.NaN,self.NaN)
        return R800/R635

    def _PSSRC(self):
        """Compute the Pigment Specific Simple Ratio for carotenoid
        (PSSRC) (Blackburn, 1998a; 1998b)"""
        R470 = np.interp(470.,self.wl,self.rf,self.NaN,self.NaN)
        R800 = np.interp(800.,self.wl,self.rf,self.NaN,self.NaN)
        return R800/R470

    def _PSNDA(self):
        """Compute the Pigment Specific Normalized Difference for chlA
        (PSNDA) (Blackburn, 1998a; 1998b)"""
        R680 = np.interp(680.,self.wl,self.rf,self.NaN,self.NaN)
        R800 = np.interp(800.,self.wl,self.rf,self.NaN,self.NaN)
        return (R800-R680)/(R800+R680)

    def _PSNDB(self):
        """Compute the Pigment Specific Normalized Difference for chlB
        (PSNDB) (Blackburn, 1998a; 1998b)"""
        R635 = np.interp(635.,self.wl,self.rf,self.NaN,self.NaN)
        R800 = np.interp(800.,self.wl,self.rf,self.NaN,self.NaN)
        return (R800-R635)/(R800+R635)

    def _PSNDC(self):
        """Compute the Pigment Specific Normalized Difference for
        carotenoid (PSNDC) (Blackburn, 1998a; 1998b)"""
        R470 = np.interp(470.,self.wl,self.rf,self.NaN,self.NaN)
        R800 = np.interp(800.,self.wl,self.rf,self.NaN,self.NaN)
        return (R800-R470)/(R800+R470)

    def _DSR1(self):
        """Compute the Datt simple ratio 1 (DSR1) (Datt, 1998)"""
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        R672 = np.interp(672.,self.wl,self.rf,self.NaN,self.NaN)
        R708 = np.interp(708.,self.wl,self.rf,self.NaN,self.NaN)
        return R672/(R550*R708)

    def _DSR2(self):
        """Compute the Datt simple ratio 2 (DSR2) (Datt, 1998)"""
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        R672 = np.interp(672.,self.wl,self.rf,self.NaN,self.NaN)
        return R672/R550

    def _DNDR(self):
        """Compute the Datt normalized difference ratio (DNDR)
        (Datt, 1999a; 1999b)"""
        R680 = np.interp(680.,self.wl,self.rf,self.NaN,self.NaN)
        R710 = np.interp(710.,self.wl,self.rf,self.NaN,self.NaN)
        R850 = np.interp(850.,self.wl,self.rf,self.NaN,self.NaN)
        return (R850-R710)/(R850-R680)

    def _DDR1(self):
        """Compute the Datt first derivative ratio (DDR)
        (Datt, 1999b)"""
        D704 = np.interp(704.,self.wl,self.rfd1,self.NaN,self.NaN)
        D754 = np.interp(754.,self.wl,self.rfd1,self.NaN,self.NaN)
        return D754/D704

    def _DDR2(self):
        """Compute the Datt second derivative ratio (DDR)
        (Datt, 1999b)"""
        DD688 = np.interp(688.,self.wl,self.rfd2,self.NaN,self.NaN)
        DD712 = np.interp(712.,self.wl,self.rfd2,self.NaN,self.NaN)
        return DD712/DD688

    def _GMSR(self):
        """Compute the Gammon simple ratio (GMSR)
        (Gamon & Surfus, 1999)"""
        Rgrn = np.interp(self.wlgrn,self.wl,self.rf,self.NaN,self.NaN)
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        return Rred/Rgrn

    def _PSRI(self):
        """Compute the Plant Senesence Reflectance Index (PSRI)
        (Merzlyak et al., 1999)"""
        R500 = np.interp(500.,self.wl,self.rf,self.NaN,self.NaN)
        R678 = np.interp(678.,self.wl,self.rf,self.NaN,self.NaN)
        R750 = np.interp(750.,self.wl,self.rf,self.NaN,self.NaN)
        return (R678-R500)/R750

    def _TVI(self):
        """Compute the Triangular Vegetation Index (TVI)
        (Broge & Leblanc, 2000)"""
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        R670 = np.interp(670.,self.wl,self.rf,self.NaN,self.NaN)
        R750 = np.interp(750.,self.wl,self.rf,self.NaN,self.NaN)
        return 0.5*(120.*(R750-R550)-200.*(R670-R550))

    def _MCARI(self):
        """Compute the Modified Chlorophyll Absorption in Reflectance
        Index (MCARI) (Daughtry et al., 2000)"""
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        R670 = np.interp(670.,self.wl,self.rf,self.NaN,self.NaN)
        R700 = np.interp(700.,self.wl,self.rf,self.NaN,self.NaN)
        return ((R700-R670)-0.2*(R700-R550))*(R700/R670)

    def _MOR(self):
        """Compute the MCARI OSAVI ratio (MOR)
        (Daughtry et al., 2000)"""
        MCARI = self._MCARI()
        OSAVI = self._OSAVI()
        return MCARI/OSAVI

    def _ZTSR1(self):
        """Compute the Zarco-Tejada simple ratio 1 (ZTSR1)
        (Zarco-Tejada et al., 2000a; 2000b)"""
        R655 = np.interp(655.,self.wl,self.rf,self.NaN,self.NaN)
        R685 = np.interp(685.,self.wl,self.rf,self.NaN,self.NaN)
        return R685/R655

    def _CI(self):
        """Compute the Curvature Index (CI)
        (Zarco-Tejada et al., 2000a; 2000b; 2002; 2003; 2009)"""
        R683 = np.interp(683.,self.wl,self.rf,self.NaN,self.NaN)
        R675 = np.interp(675.,self.wl,self.rf,self.NaN,self.NaN)
        R691 = np.interp(691.,self.wl,self.rf,self.NaN,self.NaN)
        return (R683*R683)/(R675*R691)

    def _ZTDR1(self):
        """Compute the Zarco-Tejada derivative ratio 1 (ZTDR1)
        (Zarco-Tejada et al., 2000b; 2003)"""
        D706 = np.interp(706.,self.wl,self.rfd1,self.NaN,self.NaN)
        D730 = np.interp(730.,self.wl,self.rfd1,self.NaN,self.NaN)
        return D730/D706

    def _ZTSR2(self):
        """Compute the Zarco-Tejada simple ratio 2 (ZTSR2)
        (Zarco-Tejada et al., 2000b; 2009)"""
        R710 = np.interp(710.,self.wl,self.rf,self.NaN,self.NaN)
        R750 = np.interp(750.,self.wl,self.rf,self.NaN,self.NaN)
        return R750/R710

    def _CAI(self):
        """Compute the Cellulose Absorption Index (CAI)
        (Daughtry, 2001)"""
        R2019 = np.interp(2019.,self.wl,self.rf,self.NaN,self.NaN)
        R2109 = np.interp(2109.,self.wl,self.rf,self.NaN,self.NaN)
        R2206 = np.interp(2206.,self.wl,self.rf,self.NaN,self.NaN)
        return 0.5*(R2019+R2206)-R2109

    def _ARI(self):
        """Compute the Anthocyanin Reflectance Index (ARI)
        (Gitelson et al., 2001)"""
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        R700 = np.interp(700.,self.wl,self.rf,self.NaN,self.NaN)
        return (1./R550)-(1./R700)

    def _MND1(self):
        """Compute the Maccioni normalized difference 1 (MND1)
        (Maccioni et al., 2001)"""
        R680 = np.interp(680.,self.wl,self.rf,self.NaN,self.NaN)
        R710 = np.interp(710.,self.wl,self.rf,self.NaN,self.NaN)
        R780 = np.interp(780.,self.wl,self.rf,self.NaN,self.NaN)
        return (R780-R710)/(R780-R680)

    def _MND2(self):
        """Compute the Maccioni normalize difference 2 (MND2)
        (Maccioni et al., 2001)"""
        R542 = np.interp(542.,self.wl,self.rf,self.NaN,self.NaN)
        R750 = np.interp(750.,self.wl,self.rf,self.NaN,self.NaN)
        widx = np.where(np.logical_and(self.wl>=660.,self.wl<=680.))
        if len(widx[0])==0: return self.NaN
        Rredmin = np.amin(self.rf[widx])
        return (R542-Rredmin)/(R750-Rredmin)

    def _MND3(self):
        """Compute the Maccioni normalize difference 3 (MND3)
        (Maccioni et al., 2001)"""
        R706 = np.interp(706.,self.wl,self.rf,self.NaN,self.NaN)
        R750 = np.interp(750.,self.wl,self.rf,self.NaN,self.NaN)
        widx = np.where(np.logical_and(self.wl>=660.,self.wl<=680.))
        if len(widx[0])==0: return self.NaN
        Rredmin = np.amin(self.rf[widx])
        return (R706-Rredmin)/(R750-Rredmin)

    def _MND4(self):
        """Compute the Maccioni normalize difference 4 (MND4)
        (Maccioni et al., 2001)"""
        R556 = np.interp(556.,self.wl,self.rf,self.NaN,self.NaN)
        R750 = np.interp(750.,self.wl,self.rf,self.NaN,self.NaN)
        widx = np.where(np.logical_and(self.wl>=660.,self.wl<=680.))
        if len(widx[0])==0: return self.NaN
        Rredmin = np.amin(self.rf[widx])
        return (R556-Rredmin)/(R750-Rredmin)

    def _CAINT(self):
        """Compute the Chlorophyll Absorption Integral (CAINT)
        (Oppelt & Mauser, 2001)"""
        widx = np.where(np.logical_and(self.wl>=600.,self.wl<=735.))
        if len(widx[0])==0: return self.NaN
        R600 = np.interp(600.,self.wl,self.rf,self.NaN,self.NaN)*100.
        R735 = np.interp(735.,self.wl,self.rf,self.NaN,self.NaN)*100.
        x = [600.,735.]
        y = [R600,R735]
        coef = np.polyfit(x,y,1)
        re = coef[0]*self.wl+coef[1]
        req = self.rf[widx]/re[widx]*self.bndwdth[widx]
        return np.sum(req)

    def _ZTSUM(self):
        """Compute the area of the first derivative peak from 680 nm to
        760 nm (ZTSUM) (Zarco-Tejada et al., 2001b)"""
        widx = np.where(np.logical_and(self.wl>=680.,self.wl<=760.))
        if len(widx[0])==0: return self.NaN
        return np.sum(self.rfd1[widx]*self.bndwdth[widx])

    def _PRI3(self):
        """Compute the Photochemical Reflectance Index 3 (PRI3)
        (Zarco-Tejada et al., 2001b)"""
        R531 = np.interp(531.,self.wl,self.rf,self.NaN,self.NaN)
        R570 = np.interp(570.,self.wl,self.rf,self.NaN,self.NaN)
        return (R531-R570)/(R531+R570)

    def _ZTDPR1(self):
        """Compute the Zarco-Tejada derivative peak ratio 1 (ZTDPR1)
        (Zarco-Tejada et al., 2001b)"""
        pkwv = self._WLREIPG()
        Dp = np.interp(pkwv,self.wl,self.rfd1,self.NaN,self.NaN)
        Dp12 = np.interp(pkwv+12.,self.wl,self.rfd1,self.NaN,self.NaN)
        return Dp/Dp12

    def _ZTDPR2(self):
        """Compute the Zarco-Tejada derivative peak ratio 2 (ZTDPR2)
        (Zarco-Tejada et al., 2001b)"""
        pkwv = self._WLREIPG()
        Dp = np.interp(pkwv,self.wl,self.rfd1,self.NaN,self.NaN)
        Dp22 = np.interp(pkwv+22.,self.wl,self.rfd1,self.NaN,self.NaN)
        return Dp/Dp22

    def _ZTDP21(self):
        """Compute the Zarco-Tejada derivative peak ratio 21 (ZTDP21)
        (Zarco-Tejada et al., 2001b)"""
        pkwv = self._WLREIPG()
        Dp = np.interp(pkwv,self.wl,self.rfd1,self.NaN,self.NaN)
        D703 = np.interp(703.,self.wl,self.rfd1,self.NaN,self.NaN)
        return Dp/D703

    def _ZTDP22(self):
        """Compute the Zarco-Tejada derivative peak ratio 22 (ZTDP22)
        (Zarco-Tejada et al., 2001b)"""
        pkwv = self._WLREIPG()
        Dp = np.interp(pkwv,self.wl,self.rfd1,self.NaN,self.NaN)
        D720 = np.interp(720.,self.wl,self.rfd1,self.NaN,self.NaN)
        return Dp/D720

    def _GI(self):
        """Compute the Greeness Index (GI)
        (Zarco-Tejada et al., 2001b)"""
        R554 = np.interp(554.,self.wl,self.rf,self.NaN,self.NaN)
        R677 = np.interp(677.,self.wl,self.rf,self.NaN,self.NaN)
        return R554/R677

    def _ZTSR3(self):
        """Compute the Zarco-Tejada simple ratio 3 (ZTSR3)
        (Zarco-Tejada et al., 2001a)"""
        R630 = np.interp(630.,self.wl,self.rf,self.NaN,self.NaN)
        R680 = np.interp(680.,self.wl,self.rf,self.NaN,self.NaN)
        return R680/R630

    def _ZTSR4(self):
        """Compute the Zarco-Tejada simple ratio 4 (ZTSR4)
        (Zarco-Tejada et al., 2001a)"""
        R630 = np.interp(630.,self.wl,self.rf,self.NaN,self.NaN)
        R685 = np.interp(685.,self.wl,self.rf,self.NaN,self.NaN)
        return R685/R630

    def _ZTSR5(self):
        """Compute the Zarco-Tejada simple ratio 5 (ZTSR5)
        (Zarco-Tejada et al., 2001a)"""
        R630 = np.interp(630.,self.wl,self.rf,self.NaN,self.NaN)
        R687 = np.interp(687.,self.wl,self.rf,self.NaN,self.NaN)
        return R687/R630

    def _ZTSR6(self):
        """Compute the Zarco-Tejada simple ratio 6 (ZTSR6)
        (Zarco-Tejada et al., 2001a)"""
        R630 = np.interp(630.,self.wl,self.rf,self.NaN,self.NaN)
        R690 = np.interp(690.,self.wl,self.rf,self.NaN,self.NaN)
        return R690/R630

    def _VARI(self):
        """Compute the Visible Atmospherically Resistant Index (VARI)
        (Gitelson et al., 2002a)"""
        Rblu = np.interp(self.wlblu,self.wl,self.rf,self.NaN,self.NaN)
        Rgrn = np.interp(self.wlgrn,self.wl,self.rf,self.NaN,self.NaN)
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        return (Rgrn-Rred)/(Rgrn+Rred-Rblu)

    def _CRI500(self):
        """Compute the Carotenoid Reflectance Index (CRI500)
        (Gitelson et al., 2002b)"""
        R510 = np.interp(510.,self.wl,self.rf,self.NaN,self.NaN)
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        return (1./R510)-(1./R550)

    def _CRI700(self):
        """Compute the Carotenoid Reflectance Index (CRI700)
        (Gitelson et al., 2002b)"""
        R510 = np.interp(510.,self.wl,self.rf,self.NaN,self.NaN)
        R700 = np.interp(700.,self.wl,self.rf,self.NaN,self.NaN)
        return (1./R510)-(1./R700)

    def _TCARI(self):
        """Compute the Transformed Chlorophyll Absorption Ratio Index
        (TCARI) (Haboudane et al., 2002)"""
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        R670 = np.interp(670.,self.wl,self.rf,self.NaN,self.NaN)
        R700 = np.interp(700.,self.wl,self.rf,self.NaN,self.NaN)
        return 3.*((R700-R670)-0.2*(R700-R550)*(R700/R670))

    def _TOR(self):
        """Compute the TCARI OSAVI ratio (TOR)
        (Haboudane et al., 2002)"""
        TCARI = self._TCARI()
        OSAVI = self._OSAVI()
        return TCARI/OSAVI

    def _EVI(self):
        """Compute the Enhanced Vegetation Index (EVI)
        (Huete et al., 2002)"""
        Rblu = np.interp(self.wlblu,self.wl,self.rf,self.NaN,self.NaN)
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        return 2.5*((Rnir-Rred)/(Rnir+(6.*Rred)-(7.5*Rblu)+1.))

    def _NDNI(self):
        """Compute the Normalized Difference Nitrogen Index (NDNI)
        (Serrano et al., 2002)"""
        R1510 = np.interp(1510.,self.wl,self.rf,self.NaN,self.NaN)
        R1680 = np.interp(1680.,self.wl,self.rf,self.NaN,self.NaN)
        num=math.log10(1./R1510)-math.log10(1./R1680)
        den=math.log10(1./R1510)+math.log10(1./R1680)
        return num/den

    def _NDLI(self):
        """Compute the Normalized Difference Lignin Index (NDLI)
        (Serrano et al., 2002)"""
        R1754 = np.interp(1754.,self.wl,self.rf,self.NaN,self.NaN)
        R1680 = np.interp(1680.,self.wl,self.rf,self.NaN,self.NaN)
        num=math.log10(1./R1754)-math.log10(1./R1680)
        den=math.log10(1./R1754)+math.log10(1./R1680)
        return num/den

    def _MSR2(self):
        """Compute the Modified Simple Ratio 2 (MSR2)
        (Sims & Gamon, 2002)"""
        R445 = np.interp(445.,self.wl,self.rf,self.NaN,self.NaN)
        R705 = np.interp(705.,self.wl,self.rf,self.NaN,self.NaN)
        R750 = np.interp(750.,self.wl,self.rf,self.NaN,self.NaN)
        return (R750-R445)/(R705-R445)

    def _SMNDVI(self):
        """Compute the Sims Modified Normalized Difference Vegetation
        Index (SMNDVI) (Sims & Gamon, 2002)"""
        R445 = np.interp(445.,self.wl,self.rf,self.NaN,self.NaN)
        R705 = np.interp(705.,self.wl,self.rf,self.NaN,self.NaN)
        R750 = np.interp(750.,self.wl,self.rf,self.NaN,self.NaN)
        return (R750-R705)/(R750+R705-2.*R445)

    def _GRRGM(self):
        """Compute the Gitelson Reciprocal Reflectance Green Model
        (GRRGM) (Gitelson et al., 2003; 2005)"""
        Rgrn = np.interp(self.wlgrn,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        return Rnir/Rgrn-1.

    def _GRRREM(self):
        """Compute the Gitelson Reciprocal Reflectance Red Edge Model
        (GRRREM) (Gitelson et al., 2003; 2005)"""
        wlre = self._WLREIP()
        Rre = np.interp(wlre,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        return Rnir/Rre-1.

    def _DPI(self):
        """Compute the Double-Peak Index (DPI)
        (Zarco-Tejada et al., 2003a)"""
        D688 = np.interp(688.,self.wl,self.rfd1,self.NaN,self.NaN)
        D697 = np.interp(697.,self.wl,self.rfd1,self.NaN,self.NaN)
        D710 = np.interp(710.,self.wl,self.rfd1,self.NaN,self.NaN)
        return (D688*D710)/(D697*D697)

    def _SRWI(self):
        """Compute the Simple Ratio Water Index (SRWI)
        (Zarco-Tejada et al., 2003b)"""
        R860 = np.interp(860.,self.wl,self.rf,self.NaN,self.NaN)
        R1240 = np.interp(1240.,self.wl,self.rf,self.NaN,self.NaN)
        return R860/R1240

    def _MTCI(self):
        """Compute the MERIS Terrestrial Chlorophyll Index (MTCI)
        (Dash & Curran, 2004)"""
        R681 = np.interp(681.,self.wl,self.rf,self.NaN,self.NaN)
        R709 = np.interp(709.,self.wl,self.rf,self.NaN,self.NaN)
        R754 = np.interp(754.,self.wl,self.rf,self.NaN,self.NaN)
        return (R754-R709)/(R709-R681)

    def _WDRVI(self,a=0.15):
        """Compute the Wide Dynamic Range Vegetation Index (WDRVI)
        (Gitelson, 2004)"""
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        return (a*Rnir-Rred)/(a*Rnir+Rred)

    def _MCARI1(self):
        """Compute the Modified Chlorophyll Absorption in Reflectance
        Index 1 (MCARI1) (Haboudane et al., 2004)"""
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        R670 = np.interp(670.,self.wl,self.rf,self.NaN,self.NaN)
        R800 = np.interp(800.,self.wl,self.rf,self.NaN,self.NaN)
        return 1.2*(2.5*(R800-R670)-1.3*(R800-R550))

    def _MCARI2(self):
        """Compute the Modified Chlorophyll Absorption in Reflectance
        Index 2 (MCARI2) (Haboudane et al., 2004)"""
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        R670 = np.interp(670.,self.wl,self.rf,self.NaN,self.NaN)
        R800 = np.interp(800.,self.wl,self.rf,self.NaN,self.NaN)
        num = 1.5*(2.5*(R800-R670)-1.3*(R800-R550))
        den1 = (2.0*R800+1.)*(2.0*R800+1.)
        den2 = 6.*R800 - 5.*math.sqrt(R670)
        return num / math.sqrt(den1-den2-0.5)

    def _MTVI1(self):
        """Compute the Modified Triangular Vegetation Index 1 (MTVI1)
        (Haboudane et al., 2004)"""
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        R670 = np.interp(670.,self.wl,self.rf,self.NaN,self.NaN)
        R800 = np.interp(800.,self.wl,self.rf,self.NaN,self.NaN)
        return 1.2*(1.2*(R800-R550)-2.5*(R670-R550))

    def _MTVI2(self):
        """Compute the Modified Triangular Vegetation Index 2 (MTVI2)
        (Haboudane et al., 2004)"""
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        R670 = np.interp(670.,self.wl,self.rf,self.NaN,self.NaN)
        R800 = np.interp(800.,self.wl,self.rf,self.NaN,self.NaN)
        num = 1.5*(1.2*(R800-R550)-2.5*(R670-R550))
        den1 = (2.*R800+1.)*(2.*R800+1.)
        den2 = 6.*R800 - 5.*math.sqrt(R670)
        return num / math.sqrt(den1-den2-0.5)

    def _DD(self):
        """Compute the Double Difference Index (DD)
        (Le Maire et al., 2004)"""
        R672 = np.interp(672.,self.wl,self.rf,self.NaN,self.NaN)
        R701 = np.interp(701.,self.wl,self.rf,self.NaN,self.NaN)
        R720 = np.interp(720.,self.wl,self.rf,self.NaN,self.NaN)
        R749 = np.interp(749.,self.wl,self.rf,self.NaN,self.NaN)
        return (R749-R720)-(R701-R672)

    def _LCA(self):
        """Compute the Lignin Cellulose Absorption Index (LCA)
        (Daughtry et al., 2005)"""
        R2165 = np.interp(2165.,self.wl,self.rf,self.NaN,self.NaN)
        R2205 = np.interp(2205.,self.wl,self.rf,self.NaN,self.NaN)
        R2330 = np.interp(2330.,self.wl,self.rf,self.NaN,self.NaN)
        return 100.*((R2205-R2165)+(R2205-R2330))

    def _RGI(self):
        """Compute the Red Green Pigment Index (RGI)
        (Zarco-Tejada et al., 2005)"""
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        R690 = np.interp(690.,self.wl,self.rf,self.NaN,self.NaN)
        return R690/R550

    def _BGI1(self):
        """Compute the Blue Green Pigment Index 1 (BGI1)
        (Zarco-Tejada et al., 2005)"""
        R400 = np.interp(400.,self.wl,self.rf,self.NaN,self.NaN)
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        return R400/R550

    def _BGI2(self):
        """Compute the Blue Green Pigment Index 2 (BGI2)
        (Zarco-Tejada et al., 2005)"""
        R450 = np.interp(450.,self.wl,self.rf,self.NaN,self.NaN)
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        return R450/R550

    def _BRI1(self):
        """Compute the Blue Red Pigment Index 1 (BRI1)
        (Zarco-Tejada et al., 2005)"""
        R400 = np.interp(400.,self.wl,self.rf,self.NaN,self.NaN)
        R690 = np.interp(690.,self.wl,self.rf,self.NaN,self.NaN)
        return R400/R690

    def _BRI2(self):
        """Compute the Blue Red Pigment Index 2 (BRI2)
        (Zarco-Tejada et al., 2005)"""
        R450 = np.interp(450.,self.wl,self.rf,self.NaN,self.NaN)
        R690 = np.interp(690.,self.wl,self.rf,self.NaN,self.NaN)
        return R450/R690

    def _WLREIPE(self):
        """Compute the wavelength (nm) of the red edge inflection point
        via linear extrapolation (WLREIPE) (Cho & Skidmore, 2006)"""
        D680 = np.interp(680.,self.wl,self.rfd1,self.NaN,self.NaN)
        D700 = np.interp(700.,self.wl,self.rfd1,self.NaN,self.NaN)
        D725 = np.interp(725.,self.wl,self.rfd1,self.NaN,self.NaN)
        D760 = np.interp(760.,self.wl,self.rfd1,self.NaN,self.NaN)
        x1 = [680.,700.]
        y1 = [D680,D700]
        coef1 = np.polyfit(x1,y1,1)
        x2 = [725.,760.]
        y2 = [D725,D760]
        coef2 = np.polyfit(x2,y2,1)
        return -1.*(coef1[1]-coef2[1])/(coef1[0]-coef2[0])

    def _RVIOPT(self):
        """Compute the Reyniers VIopt (RVIOPT)
        (Reyniers et al., 2006)"""
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        return (1.+0.45)*(Rnir*Rnir+1.)/(Rred+0.45)

    def _SPVI(self):
        """Compute the Spectral Polygon Vegetation Index (SPVI)
        (Vincini et al., 2006)"""
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        R670 = np.interp(670.,self.wl,self.rf,self.NaN,self.NaN)
        R800 = np.interp(800.,self.wl,self.rf,self.NaN,self.NaN)
        return 0.4*(3.7*(R800-R670)-1.2*abs(R550-R670))

    def _MMR(self):
        """Compute the MCARI MTVI2 ratio (MMR)
        (Eitel et al., 2007; 2008)"""
        MCARI = self._MCARI()
        MTVI2 = self._MTVI2()
        return MCARI/MTVI2

    def _TCI(self):
        """Compute the Triangular Chlorophyll Index (TCI)
        (Haboudane et al., 2008)"""
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        R670 = np.interp(670.,self.wl,self.rf,self.NaN,self.NaN)
        R700 = np.interp(700.,self.wl,self.rf,self.NaN,self.NaN)
        return 1.2*(R700-R550)-1.5*(R670-R550)*math.sqrt(R700/R670)

    def _EVI2(self):
        """Compute the Enhanced Vegetation Index 2 (EVI2)
        (Jiang et al., 2008)"""
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        return 2.5*(Rnir-Rred)/(Rnir+2.4*Rred+1.)

    def _DDN(self):
        """Compute the New Double Difference Index (DDN)
        (Le Maire et al., 2008)"""
        R660 = np.interp(660.,self.wl,self.rf,self.NaN,self.NaN)
        R710 = np.interp(710.,self.wl,self.rf,self.NaN,self.NaN)
        R760 = np.interp(760.,self.wl,self.rf,self.NaN,self.NaN)
        return 2.*R710-R660-R760

    def _CVI(self):
        """Compute the Chlorophyll Vegetation Index (CVI)
        (Vincini et al., 2008)"""
        Rgrn = np.interp(self.wlgrn,self.wl,self.rf,self.NaN,self.NaN)
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        return (Rnir*Rred)/(Rgrn*Rgrn)

    def _WUTCARI(self):
        """Compute the Transformed Chlorophyll Absorption Ratio Index
        with Wu's modification (WUTCARI) (Wu et al., 2008)"""
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        R705 = np.interp(705.,self.wl,self.rf,self.NaN,self.NaN)
        R750 = np.interp(750.,self.wl,self.rf,self.NaN,self.NaN)
        return 3.*((R750-R705)-0.2*(R750-R550)*(R750/R705))

    def _WUOSAVI(self):
        """Compute the Optimized Soil-Adjusted Vegetation Index with
        Wu's modification (WUOSAVI) (Wu et al., 2008)"""
        R705 = np.interp(705.,self.wl,self.rf,self.NaN,self.NaN)
        R750 = np.interp(750.,self.wl,self.rf,self.NaN,self.NaN)
        return (1.+0.16)*(R750-R705)/(R750+R705+0.16)

    def _WUMCARI(self):
        """Compute the Modified Chlorophyll Absorption in Reflectance
        Index with Wu's modification (WUMCARI) (Wu et al., 2008)"""
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        R705 = np.interp(705.,self.wl,self.rf,self.NaN,self.NaN)
        R750 = np.interp(750.,self.wl,self.rf,self.NaN,self.NaN)
        return ((R750-R705)-0.2*(R750-R550))*(R750/R705)

    def _WUMSR(self):
        """Compute the Modified Simple Ratio with Wu's modification
        (WUMSR) (Wu et al., 2008)"""
        R705 = np.interp(705.,self.wl,self.rf,self.NaN,self.NaN)
        R750 = np.interp(750.,self.wl,self.rf,self.NaN,self.NaN)
        return (R750/R705-1.)/(math.sqrt((R750/R705)+1.))

    def _WUTOR(self):
        """Compute the TCARI OSAVI ratio with Wu's modification (WUTOR)
        (Wu et al., 2008)"""
        WUTCARI = self._WUTCARI()
        WUOSAVI = self._WUOSAVI()
        return WUTCARI/WUOSAVI

    def _WUMOR(self):
        """Compute the MCARI OSAVI ratio with Wu's modification (WUMOR)
        (Wu et al., 2008)"""
        WUMCARI = self._WUMCARI()
        WUOSAVI = self._WUOSAVI()
        return WUMCARI/WUOSAVI

    def _DCNI(self):
        """Compute the Double-peak Canopy Nitrogen Index (DCNI)
        (Chen et al., 2010)"""
        R670 = np.interp(670.,self.wl,self.rf,self.NaN,self.NaN)
        R700 = np.interp(700.,self.wl,self.rf,self.NaN,self.NaN)
        R720 = np.interp(720.,self.wl,self.rf,self.NaN,self.NaN)
        return (R720-R700)/(R700-R670)/(R720-R670+0.03)

    def _TGI(self):
        """Compute the Triangular Greeness Index (TGI)
        (Hunt et al., 2011)"""
        R480 = np.interp(480.,self.wl,self.rf,self.NaN,self.NaN)
        R550 = np.interp(550.,self.wl,self.rf,self.NaN,self.NaN)
        R670 = np.interp(670.,self.wl,self.rf,self.NaN,self.NaN)
        return -0.5*((670.-480.)*(R670-R550)-(670.-550.)*(R670-R480))

    def _WDRVI2(self,a=0.2):
        """Compute the Wide Dynamic Range Vegetation Index 2 (WDRVI2)
        (Peng & Gitelson, 2011)"""
        Rred = np.interp(self.wlred,self.wl,self.rf,self.NaN,self.NaN)
        Rnir = np.interp(self.wlnir,self.wl,self.rf,self.NaN,self.NaN)
        return (a*Rnir-Rred)/(a*Rnir+Rred)+(1.-a)/(1.+a)

    def _AIVI(self):
        """Compute the Angular Insensitivity Vegetation Index (AIVI)
        (He et al., 2016)"""
        R445 = np.interp(445.,self.wl,self.rf,self.NaN,self.NaN)
        R573 = np.interp(573.,self.wl,self.rf,self.NaN,self.NaN)
        R720 = np.interp(720.,self.wl,self.rf,self.NaN,self.NaN)
        R735 = np.interp(735.,self.wl,self.rf,self.NaN,self.NaN)
        return (R445*(R720+R735)-R573*(R720-R735))/(R720*(R573+R445))

    def _DND(self):
        """Compute the Derivative Normalized Difference (DND)
        (Sonobe & Wang, 2017)"""
        D522 = np.interp(522.,self.wl,self.rfd1,self.NaN,self.NaN)
        D728 = np.interp(728.,self.wl,self.rfd1,self.NaN,self.NaN)
        return (D522-D728)/(D522+D728)
