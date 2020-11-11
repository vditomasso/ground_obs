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

    def __init__(self,wav,flux,wav_unit,medium='vac'):
        '''
        Initialize a Spectrum object
        wav: wavelength (array)
        flux: flux (array)
        wav_unit: wavelength unit (string)
        medium: either 'vac' or 'air' (string), default: 'vac'
        '''
        try:
            self.wav = wav
            self.flux = flux
            self.R = self._R()
            self.wav_range = np.array([min(wav),max(wav)])
            self.wav_unit = u.Unit(wav_unit)
            assert (medium =='vac') or (medium == 'air')
            self.medium = medium
        except ValueError:
            print('ValueError: {} is not an astropy unit. \n Maybe you would like cm, micron, or nm?'.format(wav_unit))
        except AssertionError:
            print('AssertionError: Medium needs to be either vac or air, not {}'.format(medium))

    def __str__(self):
        try:
            return('Spectrum object with {} wavelength range = {}-{} {} and avg R = {}'.format(self.medium,np.round(self.wav_range[0]),np.round(self.wav_range[1]),self.wav_unit,np.round(np.mean(self.R))))
        except AttributeError:
            return('Spectrum object was not initiated properly. \n Check that it has attributes: wav, flux, wav_range, R and medium.')
            
    def _R(self):
        '''Find the resolution of the Spectrum'''
        diffs = np.diff(self.wav)  # Calculates Delta lambdas
        diffs = np.append(diffs, diffs[-1])  # Keeps len(diffs) == len(wavs)
        return self.wav / diffs  # R = lambda / Delta lambda
        
    def to_air(self):
    # Following VALD3: http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
        R_before = self._R()
        if self.medium == 'air':
            print('Spectrum is already in air wavelengths')
        elif self.medium == 'vac':
            s = 10**4/self.wav
            n = 1 + 0.0000834254 + 0.02406147 / (130-s**2) + 0.00015998 / (38.9-s**2)
            self.wav = self.wav / n
            self.wav_range = np.array([min(self.wav),max(self.wav)])
            self.medium = 'air'
#            ### Maybe this should become a test in the testing module
#            if self._R().any()!=R_before.any():
#                print('Error: Resolution of the spectrum has changed')
        else:
            print('Error: self.medium is neither air nor vac')
            
    def to_vac(self):
    # Following VALD3: http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
        if self.medium=='vac':
            print('Spectrum is already in vacuum wavelengths')
        elif self.medium == 'air':
            s = 10**4/self.wav
            n =1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522-s**2) + 0.0001599740894897 / (38.92568793293-s**2)
            self.wav = self.wav*n
            self.wav_range = np.array([min(self.wav),max(self.wav)])
            self.medium = 'vac'
        else:
            print('Error: self.medium is neither air nor vac')

    def to_unit(self,new_unit):
        '''Update the wavelength unit of the Spectrum instance'''
        self.wav_unit = u.Unit(new_unit)

    def change_wav_range(self, wav_min, wav_max):
        '''Change the wavelength range of the spectrum -- updates the wav and flux attribute to only include the points that fall within the input wavelength range'''
        ### ADD IN A CHECK THAT THE RESULTING MIN(SELF.WAV) AND MAX(SELF.WAV) ACTUALLY MATCH THE INPUT WAV_MIN AND WAV_MAX
        indices = np.where((self.wav>wav_min)&(self.wav<wav_max))
        if (min(self.wav)<wav_min) and (max(self.wav)>wav_max):
        # Check that the input wav_min and wav_max are within the spectrum's original wav_range
            self.wav_range = np.array([wav_min,wav_max])
            self.wav = self.wav[indices]
            self.flux = self.flux[indices]
        else:
            print('Error: The input wavelength range ({}-{}) is not a subset of the initial wavelength range of this spectrum'.format(wav_min,wav_max))

    def change_R(self,R):
        '''Change the resolution of the spetrum -- updates the wav, flux and R attributes'''
        # Adapted from Ian's O2/utils.py resample function
        ### ADD IN A CHECK THAT THE RESULTING SELF._R() ACTUALLY MATCHES THE INPUT R
        wav_min, wav_max = self.wav_range
        wav_central = (wav_min + wav_max) / 2
        wav_delta = wav_central / R
        wav_resampled = np.arange(wav_min, wav_max, wav_delta)
        flux_resampled = spectres(wav_resampled, self.wav, self.flux)
        if (np.round(min(self.wav))==np.round(wav_min)) and (np.round(max(self.wav))==np.round(wav_max)):
            self.wav = wav_resampled
            self.flux = flux_resampled
            self.R = np.ones(self.wav.shape)*R
        else:
            print('Error: Spectrum wav_range has changed')


### Testing ###
import data_io
telluric_spec_file = 'Atm_Transmission_Kurucz_2005.txt'
exo_spec_file = 'O2_1E6.txt'

tel_spec_df, exo_spec_df = data_io.load_data(data_io.get_data_file_path(telluric_spec_file), data_io.get_data_file_path(exo_spec_file))

wav = np.array(tel_spec_df['wavelength'])
flux = np.array(tel_spec_df['flux'])

test_spec = Spectrum(wav, flux, 'nm')
#print(test_spec)
test_spec.change_wav_range(750,780)
print(test_spec)
#test_spec.change_R(3e5)
#print(type(test_spec.medium))
test_spec.to_air()
print(test_spec)

