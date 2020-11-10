#!/bin/usr python3

import numpy as np
import timeit
from detecta import detect_peaks
from PyAstronomy import pyasl
from scipy.interpolate import interp1d
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from astropy.time import Time
#from ground_obs import blend_calc_lib
from scipy import interpolate

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
        self.R = self.get_R()
        self.wav_range = np.array([min(wav),max(wav)])
        self.wav_unit = u.Unit(wav_unit)
        try:
            self.wav_unit = u.Unit(wav_unit)
        except ValueError:
            print('{} is not an astropy unit. \n Maybe you would like cm, micron, or nm?'.format(wav_unit))
            
    def get_R(self):
        '''Find the resolution of the Spectrum'''
        diffs = np.diff(self.wav)  # Calculates Delta lambdas
        diffs = np.append(diffs, diffs[-1])  # Keeps len(diffs) == len(wavs)
        return self.wav / diffs  # R = lambda / Delta lambda
            
    def to_unit(self,new_unit):
        '''Update the wavelength unit of the Spectrum object'''
        self.wav_unit = u.Unit(new_unit)
        
#    def to_range(self, min_wav, max_wav):
#        '''Restrict the range of the wa'''

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
print(test_spec.R)
