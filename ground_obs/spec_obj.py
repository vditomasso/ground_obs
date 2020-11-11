#!/bin/usr python3

import numpy as np
#import timeit
#from detecta import detect_peaks
#from PyAstronomy import pyasl
#from scipy.interpolate import interp1d
import astropy.units as u
#from astropy.coordinates import SkyCoord
#import astropy.coordinates as coord
#from astropy.time import Time
#from ground_obs import blend_calc_lib
#from scipy import interpolate
from spectres import spectres


class Spectrum():

    def __init__(self,wav,flux,wav_unit):
        '''
        Initialize a Spectrum object
        wav: wavelength (array)
        flux: flux (array)
        wav_unit: wavelength unit (string)
        '''
        self.wav = wav
        self.flux = flux
        self.R = self.R()
        self.wav_range = np.array([min(wav),max(wav)])
        self.wav_unit = u.Unit(wav_unit)
        # Could add check that wavelength array is increasing
        try:
            self.wav_unit = u.Unit(wav_unit)
        except ValueError:
            print('{} is not an astropy unit. \n Maybe you would like cm, micron, or nm?'.format(wav_unit))
            
    def R(self):
        '''Find the resolution of the Spectrum'''
        diffs = np.diff(self.wav)  # Calculates Delta lambdas
        diffs = np.append(diffs, diffs[-1])  # Keeps len(diffs) == len(wavs)
        return self.wav / diffs  # R = lambda / Delta lambda

    def to_unit(self,new_unit):
        '''Update the wavelength unit of the Spectrum instance'''
        self.wav_unit = u.Unit(new_unit)
 
 
 ### START WITH RESAMPLE -- it does not currently work ###
 
## From Ian's O2/utils.py
# def resample(wav, wav_band, R, flux):
#    wav_min, wav_max = wav_band
#    wav_central = (wav_min + wav_max) / 2
#    wav_delta = wav_central / R
#    wav_resampled = np.arange(wav_min, wav_max, wav_delta)
#    flux_resampled = spectres(wav_resampled, wav, flux)
#    return wav_resampled, flux_resampled
 
#    def resample(self, wav_min, wav_max, R):
#        '''Set the wavelength range and resolution of the Spectrum -- updates the wav, flux, wav_range and R of the Spectrum instance'''
#        wav_central = (wav_min + wav_max) / 2
#        wav_delta = wav_central / R
#        wav_resampled = np.arange(wav_min, wav_max, wav_delta)
#        flux_resampled = spectres(wav_resampled, self.wav, self.flux)
#        self.wav = wav_resampled
#        self.flux = flux_resampled
#        self.wav_range = np.array([min(wav),max(wav)])

    def change_wav_range(self, wav_min, wav_max):
        '''Change the wavelength range of the spectrum by updating the wav and flux attribute to only include the points that fall within the input wavelength range'''
        indices = np.where((self.wav>wav_min)&(self.wav<wav_max))
        if (min(self.wav)<wav_min) and (max(self.wav)>wav_max):
        # Check that the input wav_min and wav_max are within the spectrum's original wav_range
            self.wav_range = np.array([wav_min,wav_max])
            self.wav = self.wav[indices]
            self.flux = self.flux[indices]
        else:
            print('The input wavelength range ({}-{}) is not a subset of the initial wavelength range of this spectrum'.format(wav_min,wav_max))

    def change_R(self):
        '''Change the resolution of the spetrum -- updates the wav, flux and R attributes'''
        pass



### Testing ###
import data_io
telluric_spec_file = 'Atm_Transmission_Kurucz_2005.txt'
exo_spec_file = 'O2_1E6.txt'

tel_spec_df, exo_spec_df = data_io.load_data(data_io.get_data_file_path(telluric_spec_file), data_io.get_data_file_path(exo_spec_file))

wav = np.array(tel_spec_df['wavelength'])
flux = np.array(tel_spec_df['flux'])

test_spec = Spectrum(wav, flux, 'micron')
#print(test_spec.wav_unit)
#test_spec.to_unit('cm')
#print(test_spec.wav_unit)
#print(test_spec.R)
print(test_spec.wav_range)
#print(test_spec.R)
test_spec.change_wav_range(750,770)
#test_spec.resample(0.75,0.77,3e5)
print(test_spec.wav_range)
#print(test_spec.R)
