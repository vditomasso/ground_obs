import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import numpy as np
import pandas as pd
import scipy as sp
import os
import pickle
import utils

from tqdm import tqdm
from astropy.io import fits
from astropy import units as u
from astropy import constants as c
from PyAstronomy import pyasl
from spectres import spectres
from num2tex import num2tex

# pickle loader convenience function
def pkl_load(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f, encoding="latin")  # Python 2 -> 3
    return data


def read_fits(fpath, ext_idx=0):
    with open(fpath, "rb") as f:
        hdu = fits.open(f)
        header = hdu[ext_idx].header
        data = hdu[ext_idx].data
    return header, data


def count_photons(
    m_1=None,
    m_2=None,
    d_1=None,
    D=None,
    tau=None,
    delta_lambda=None,
    lambda_0=None,
):
    if m_1 is None:
        m_1 = m_2 + 5.0 * np.log10(
            d_1 / (10.0 * u.pc)
        )  # assumes m_2=M (absolute magnitude)

    flux_unit = u.Unit("erg cm-2 s-1 angstrom-1")
    nm = u.nm
    # F_2 values taken from Johnson monochromatic tables at m_2=0
    if (507 * nm <= lambda_0) and (lambda_0 <= 595 * nm):  # Visible
        F_2 = 3.61e-9 * flux_unit
    elif (731.5 * nm <= lambda_0) and (lambda_0 <= 880.5 * nm):  # Infrared
        F_2 = 1.14e-9 * flux_unit
    else:
        raise Exception("Bandpass not implemented")

    F_1 = 10.0 ** (-m_1 / 2.5) * F_2
    E = c.h * c.c / lambda_0

    return float(F_1 * tau * delta_lambda * 1.0 / E * np.pi * (D / 2.0) ** 2)


def get_R(wavs):
    diffs = np.diff(wavs)  # Calculates Delta lambdas
    diffs = np.append(diffs, diffs[-1])  # Keeps len(diffs) == len(wavs)
    return wavs / diffs  # R = lambda / Delta lambda


def resample(wav, wav_band, R, flux):
    wav_min, wav_max = wav_band
    wav_central = (wav_min + wav_max) / 2
    wav_delta = wav_central / R
    wav_resampled = np.arange(wav_min, wav_max, wav_delta)
    flux_resampled = spectres(wav_resampled, wav, flux)
    return wav_resampled, flux_resampled


def get_C(
    R,
    wav=None,
    wav_band=None,
    flux_S=None,
    flux_T=None,
    flux_P=None,
    v_star=2e6,
    v_planet=0,
    eps=35000,
):
    # Computes Eq. 3 for both in and out-of-transit observations
    # M1V: 3600K, eps=125,000
    # M4V: 3000K, eps=35,000
    # M9V: 2300K, eps=4,000
    # solar abundances and logg=4.5dex for all stellar spectra used

    ##############
    # Get a and C~
    ##############
    c = 3e10  # Speed of light in vacuum

    # a
    coeff_S = 1 + v_star / c
    coeff_P = (1 + (v_star + v_planet) * (1 / c)) * (1 / eps)
    a_in = coeff_S * flux_S + coeff_P * flux_P
    a_out = coeff_S * flux_S  # P = 0 here

    # C~
    coeff_T = 1 / (1 + (1 / eps))
    C_tilde_in = a_in * coeff_T * flux_T
    C_tilde_out = a_out * coeff_T * flux_T

    ###########################
    # Define instrument profile
    ###########################
    wav_min, wav_max = wav_band
    wav_central = (wav_min + wav_max) / 2
#    print('type(wav_central) = ',type(wav_central))
#    print('type(R) = ',type(R))
    FWHM = wav_central / R
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))  # Width of Gaussian

    ###############################################################
    # Convolve selected portion of spectrum with instrument profile
    ###############################################################
    C_in_model = pyasl.broadGaussFast(
        wav,
        C_tilde_in,
        sigma,
        edgeHandling="firstlast",
    )
    C_out_model = pyasl.broadGaussFast(
        wav,
        C_tilde_out,
        sigma,
        edgeHandling="firstlast",
    )

    # Store inputs for diagnostics
    C_inputs = {
        # Inputs for C tilde
        "coeff_S": coeff_S,
        "coeff_T": coeff_T,
        "coeff_P": coeff_P,
        "a_in": a_in,
        "a_out": a_out,
        "C_tilde_in": C_tilde_in,
        "C_tilde_out": C_tilde_out,
        # Inputs for G
        "wav_central": wav_central,
        "FWHM": FWHM,
        "sigma": sigma,
    }

    return C_in_model, C_out_model, C_inputs


def get_P_obs(
    R=None,
    I=None,
    N=None,
    sigma_w=None,
    wav=None,
    wav_band=None,
    flux_S=None,
    flux_T=None,
    flux_P=None,
):
    # Compute model spectrum C
    C_in_model, C_out_model, _ = get_C(
        R,
        wav=wav,
        wav_band=wav_band,
        flux_S=flux_S,
        flux_T=flux_T,
        flux_P=flux_P,
    )
    # Create pool of I in-transit spectra
    noise_in = np.random.normal(0, sigma_w, size=(I, wav.size)) # wav.size is the number of points in the spectra
    # noise_in size = # of spectra in the pool, # of points in a spectrum
    in_transit_pool = np.zeros_like(noise_in) + C_in_model + noise_in # I don't understand why we need to add an array of zeros here -VD 11/4/20
#    print('in_transit_pool.size = ',in_transit_pool.size) # 5095000
#    print('noise_in.size = ',noise_in.size) # 5095000
#    print('C_in_model.size = ',C_in_model.size) # 5095

    # Create pool of N out-of-transit spectra
    noise_out = np.random.normal(0, sigma_w, size=(I, wav.size))
    out_of_transit_pool = np.zeros_like(noise_out) + C_out_model + noise_out

    # Randomly sample N spectra from each pool
    s = np.arange(0, I)
    np.random.shuffle(s)  # create random state -- shuffles the order of the 1000 spectra
    in_transit_samples = in_transit_pool[s][0:N, :]
    out_of_transit_samples = out_of_transit_pool[s][0:N, :]
    noise_samples_in = noise_in[s][0:N]
    noise_samples_out = noise_out[s][0:N]
    noise_samples_in_avg = np.mean(noise_samples_in, axis=0)
    noise_samples_out_avg = np.mean(noise_samples_out, axis=0)

    # Compute `P_obs`
    C_in = np.sum(in_transit_samples, axis=0) + noise_samples_in_avg
    C_out = np.sum(out_of_transit_samples, axis=0) + noise_samples_out_avg
    P_obs = C_in / C_out

    # Store model outputs for diagnostic
    inputs = {
        "C_in_model": C_in_model,
        "C_out_model": C_out_model,
        "in_transit_pool": in_transit_pool,
        "out_of_transit_pool": out_of_transit_pool,
        "C_in": C_in,
        "C_out": C_out,
    }
    return P_obs, inputs

################################################
# VD added 11/4/20
################################################

def get_noisy_P_obs(
    R=None, # resolution
    sigma_w=None, # white noise level
    wav=None, # wavelength array (cm)
    wav_band=None, # wavelength range (cm)
    flux_S=None, # stellar flux
    flux_T=None, # telluric flux
    flux_P=None, # planetary flux
    v_star=2e6 # RV of the star -- input to get_C
    ):
    '''Return a single P_obs spectrum with random noise at an input level.
    Note: this has the rv of the system (20km/s), the instrumental profile and the area of the star/area of the planet's atmospheric ring baked into it.'''

    # Compute model spectrum C
    C_in_model, C_out_model, _ = get_C(
        R,
        wav=wav,
        wav_band=wav_band,
        flux_S=flux_S,
        flux_T=flux_T,
        flux_P=flux_P,
        v_star=v_star
    )
    # Compute random noise at level sigma_w
    # noise to add to every point of the spectrum
    # ASSUMING THAT THE NOISE LEVEL IS THE SAME FOR IN AND OUT OF TRANSIT SPECTRA
    noise_in = np.random.normal(0, sigma_w, size=(wav.size))
    noise_out = np.random.normal(0, sigma_w, size=(wav.size))

    # Add that random noise to C_in_model and C_out_model
    C_in_noisy = C_in_model + noise_in
    C_out_noisy = C_out_model + noise_out
    
    P_obs_noisy = C_in_noisy / C_out_noisy

    # Store model outputs for diagnostic
    inputs = {
        "C_in_model": C_in_model,
        "C_out_model": C_out_model,
        "noise_in": noise_in,
        "noise_out": noise_out,
        "C_in_noisy": C_in_noisy,
        "C_out_noisy": C_out_noisy,
    }
    
    return P_obs_noisy, inputs

################################################


################################################
# VD added 11/4/20
################################################

def get_single_CCF(
    wav,
    wav_band,
    R,
    flux_S,
    flux_T,
    flux_P,
    template_wav,
    template_flux,
    sigma_w=1e-3,
    rvmin=-30,
    rvmax=30,
    drv=0.5,
    v_star=2e6
):
    '''Get the CCF for one pair of spectra'''

    P_obs = get_noisy_P_obs(R=R,
    sigma_w=sigma_w,
    wav=wav,
    wav_band=wav_band,
    flux_S=flux_S,
    flux_T=flux_T,
    flux_P=flux_P,
    v_star=v_star)[0]
    
    # Compute v_max
    rv, cc = pyasl.crosscorrRV(
        wav, P_obs, template_wav, template_flux, rvmin, rvmax, drv
    )
    maxind = np.argmax(cc)
    v_max = rv[maxind]
    res = {
        "rv": rv,
        "cc": cc,
        "v_max": v_max,
        "wav": wav,
        "P_obs": P_obs,
#        "C_in": C_in,
#        "C_out": C_out,
        "sigma_w": sigma_w,
        "template_wav": template_wav,
        "template_flux": template_flux,
#        "in_transit_samples": in_transit_samples,
#        "out_of_transit_samples": out_of_transit_samples,
    }
    # return rv, cc, v_max, P_obs, C_in, C_out
    return res

################################################



def get_CCF(
    wav,
    wav_band,
    R,
    flux_S,
    flux_T,
    flux_P,
    template_wav,
    template_flux,
    sigma_w=1e-3,
    I=1_000,
    N=10,
):

    P_obs = get_P_obs()

    # Compute v_max
    rv, cc = pyasl.crosscorrRV(
        wav, P_obs, template_wav, template_flux, -30, 30, 0.5
    )
    maxind = np.argmax(cc)
    v_max = rv[maxind]
    res = {
        "rv": rv,
        "cc": cc,
        "v_max": v_max,
        "wav": wav,
        "P_obs": P_obs,
        "C_in": C_in,
        "C_out": C_out,
        "sigma_w": sigma_w,
        "template_wav": template_wav,
        "template_flux": template_flux,
        "in_transit_samples": in_transit_samples,
        "out_of_transit_samples": out_of_transit_samples,
    }
    # return rv, cc, v_max, P_obs, C_in, C_out
    return res
