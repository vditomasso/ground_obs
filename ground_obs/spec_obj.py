#!/bin/usr python3

import numpy as np
import astropy.units as u
from spectres import spectres
from PyAstronomy import pyasl

class Spectrum():

    def __init__(self,wav,flux,wav_unit,medium='vac',name='Spectrum'):
        '''
        Initialize a Spectrum object
        wav: wavelength (array)
        flux: flux (array)
        wav_unit: wavelength unit (string)
        medium: either 'vac' or 'air' (string), default: 'vac'
        '''
        try:
            # If the wavelength array is not increasing order, flip the order of the wav and flux arrays
            self.name=name
            if (min(wav)!=wav[0]) and (max(wav)!=wav[-1]):
                wav = wav[::-1]
                flux = flux[::-1]
                print('Spectrum "{}" was not in increasing wavelength order. This has been corrected'.format(self.name))
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
            return('Spectrum object "{}" with {} wavelength range = {}-{} {} and avg R = {}'.format(self.name,self.medium,np.round(self.wav_range[0],5),np.round(self.wav_range[1],5),self.wav_unit,np.round(np.mean(self.R))))
        except AttributeError:
            return('Spectrum object was not initiated properly. \n Check that it has attributes: wav, flux, wav_range, R, medium and name.')
            
    def _R(self):
        '''Find the resolution of the Spectrum'''
        diffs = np.diff(self.wav)  # Calculates Delta lambdas
        diffs = np.append(diffs, diffs[-1])  # Keeps len(diffs) == len(wavs)
        return self.wav / diffs  # R = lambda / Delta lambda
        
    def normalize(self):
        new_Spectrum = Spectrum(self.wav, self.flux, self.wav_unit, self.medium, self.name)
        new_Spectrum._normalize()
        return(new_Spectrum)
    
    def _normalize(self):
        '''Normalize self.flux by dividing by the maximum flux value in the spectrum'''
        print('This is not a proper normalization function! This simply divides the flux values in the Spectrum by the maximum flux value! Use at your own risk')
        self.flux = self.flux/np.max(self.flux)
        
    def to_air(self):
        new_Spectrum = Spectrum(self.wav, self.flux, self.wav_unit, self.medium, self.name)
        new_Spectrum._to_air()
        return(new_Spectrum)

    def _to_air(self):
        '''Change from vacuum to air wavelengths
        Following VALD3: http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
        This equation is for wavelength in Angstroms'''
        if self.medium == 'air':
            print('Spectrum "{}" is already in air wavelengths'.format(self.name))
        elif self.medium == 'vac':
            unit_before = self.wav_unit
            self._to_unit('angstrom')
            s = 10**4/self.wav
            n = 1 + 0.0000834254 + 0.02406147 / (130-s**2) + 0.00015998 / (38.9-s**2)
            self.wav = self.wav / n # in Angstroms
            self._to_unit(unit_before) # change back to original unit
            self.wav_range = np.array([min(self.wav),max(self.wav)])
            self.medium = 'air'
        else:
            print('Error: self.medium is neither air nor vac')
         
    def to_vac(self):
        new_Spectrum = Spectrum(self.wav, self.flux, self.wav_unit, self.medium, self.name)
        new_Spectrum._to_vac()
        return(new_Spectrum)

    def _to_vac(self):
        ''' Change from air to vacuum wavelengths
        Following VALD3: http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
        This equation is for wavelength in Angstroms'''
        if self.medium=='vac':
            print('Spectrum "{}" is already in vacuum wavelengths'.format(self.name))
        elif self.medium == 'air':
            unit_before = self.wav_unit
            self._to_unit('angstrom')
            s = 10**4/self.wav
            n =1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522-s**2) + 0.0001599740894897 / (38.92568793293-s**2)
            self.wav = self.wav*n
            self._to_unit(unit_before) # change back to original unit
            self.wav_range = np.array([min(self.wav),max(self.wav)])
            self.medium = 'vac'
        else:
            print('Error: self.medium is neither air nor vac')

    def to_unit(self, new_unit):
        new_Spectrum = Spectrum(self.wav, self.flux, self.wav_unit, self.medium, self.name)
        new_Spectrum._to_unit(new_unit)
        return(new_Spectrum)

    def _to_unit(self,new_unit):
        '''Update the wav and wav_unit of the Spectrum instance'''
        # Catch error is new_unit is not an Astropy unit
        try:
            if new_unit == self.wav_unit:
                print('Spectrum "{}" is already in unit {}'.format(self.name, new_unit))
            else:
                unit_before = self.wav_unit
                wav_before = self.wav*u.Unit(unit_before)
                wav_after = wav_before.to(u.Unit(new_unit))
                self.wav = wav_after.value
                self.wav_unit = u.Unit(new_unit)
                self.wav_range = np.array([min(self.wav),max(self.wav)])
        except ValueError:
            print('ValueError: {} is not an astropy unit. \n Maybe you would like cm, micron, or nm?'.format(new_unit))

    def change_wav_range(self, wav_min, wav_max):
        new_Spectrum = Spectrum(self.wav, self.flux, self.wav_unit, self.medium, self.name)
        new_Spectrum._change_wav_range(wav_min, wav_max)
        return(new_Spectrum)

    def _change_wav_range(self, wav_min, wav_max):
        '''Change the wavelength range of the spectrum -- updates the wav and flux attribute to only include the points that fall within the input wavelength range'''
        ### ADD IN A CHECK THAT THE RESULTING MIN(SELF.WAV) AND MAX(SELF.WAV) ACTUALLY MATCH THE INPUT WAV_MIN AND WAV_MAX
        indices = np.where((self.wav>wav_min)&(self.wav<wav_max))
        if (min(self.wav)<wav_min) and (max(self.wav)>wav_max):
        # Check that the input wav_min and wav_max are within the spectrum's original wav_range
            self.wav = self.wav[indices]
            self.flux = self.flux[indices]
            self.wav_range = np.array([wav_min,wav_max])
            return(self)
        else:
            print('Error: The input wavelength range ({}-{}) is not a subset of the initial wavelength range of spectrum "{}"'.format(wav_min,wav_max,self.name))

    def change_R(self, R):
        new_Spectrum = Spectrum(self.wav,self.flux,self.wav_unit,self.medium,self.name)
        new_Spectrum._change_R(R)
        return(new_Spectrum)

    def _change_R(self,R):
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
            self.wav_range = np.array([min(self.wav),max(self.wav)])
        else:
            print('Error: Wav_range of Spectrum "{}" has changed'.format(self.name))

    def doppler_shift(self,v,v_unitlen='km',v_unittime='second'):
        new_Spectrum = Spectrum(self.wav, self.flux, self.wav_unit, self.medium, self.name)
        new_Spectrum._doppler_shift(v,v_unitlen,v_unittime)
        return(new_Spectrum)

    def _doppler_shift(self,v,v_unitlen='km',v_unittime='second'):
        '''Doppler shift using pyasl, adapted from Ian's O2/notebook.ipynb
        Optional v_unitlen and v_unittime inputs incase v is not in km/s'''
        # Catch error if v units are not in Astropy
        try:
            # Change wav_unit to A for pyasl
            wav_unit_before = self.wav_unit
            self._to_unit('angstrom')
            # Change v to km/s if necessary
            if u.Unit(v_unitlen) != u.km or u.Unit(v_unittime) != u.second:
                v = v*(u.Unit(v_unitlen)/u.Unit(v_unittime))
                v = v.to(u.km/u.second)
            flux_shifted, _ = pyasl.dopplerShift(
                                            self.wav,
                                            self.flux,
                                            v,
                                            edgeHandling="firstlast")
            self.flux = flux_shifted
            self._to_unit(wav_unit_before)
        except ValueError:
            print('ValueError: {} and/or {} is not an astropy unit. \n Maybe you would like km or m, hour or second?'.format(v_unitlen,v_unittime))

### Testing ###
#import data_io
#telluric_spec_file = 'Atm_Transmission_Kurucz_2005.txt'
#exo_spec_file = 'O2_1E6.txt'
#
#tel_spec_df, exo_spec_df = data_io.load_data(data_io.get_data_file_path(telluric_spec_file), data_io.get_data_file_path(exo_spec_file))
#
#wav = np.array(tel_spec_df['wavelength'])
#flux = np.array(tel_spec_df['flux'])
###
###wav = np.array(exo_spec_df['wavelength'])
###flux = np.array(exo_spec_df['flux'])
###
#test_spec = Spectrum(wav, flux, 'nm', name='test')
###print(test_spec.flux)
###print(test_spec.wav)
###print(test_spec.wav_unit)
###print(test_spec.medium)
##
#print(test_spec)
#test_spec = test_spec.change_wav_range(750,780)
#print('change_wav_range:')
#print('test_spec =\n',test_spec)
##print('test_spec2 =\n',test_spec2)
#test_spec = test_spec.change_R(3e5)
#print('change_R \n',test_spec)
#test_spec = test_spec.to_air()
#print('to_air \n',test_spec)
#test_spec = test_spec.to_vac()
#print('to_vac \n',test_spec)
#test_spec = test_spec.normalize()
#print('normalize \n',test_spec)
#test_spec = test_spec.to_unit('cm')
#print('to_unit \n',test_spec)
#test_spec = test_spec.doppler_shift(20)
#print('Doppler shift \n',test_spec)
#print(test_spec.flux) ### Somehow this comes out at nans, even though it works in the notebook??

