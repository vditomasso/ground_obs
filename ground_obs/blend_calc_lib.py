from __future__ import print_function, division # copied from pyasl
from PyAstronomy.pyaC import pyaErrors as PE # copied from pyasl
from PyAstronomy import funcFit as fuf # copied from pyasl
from spectres import spectres # copied from Ian's O2/utils.py
import numpy as np
import timeit
from detecta import detect_peaks
from scipy.interpolate import interp1d
import pandas as pd
from PyAstronomy import pyasl

def prep_wl_flux_arrays(tel_spec_df, exo_spec_df, wl_range, exozeropt_shift):
    '''Data cleaning function

    Load in wavelength and flux data for the telluric spectrum and the exoplanet spectrum
    Restrict the telluric spectrum to a given wavelength range
    Shift the exoplanet spectrum to the appropriate RV zero point -> ideally add in redshift_zeropt.py when Juli sends it
    Make sure both spectra have the wavelength in increasing order

    Inputs:
    ---
    tel_spec_df: pandas dataframe
    exo_spec_df: pandas dataframe
    wl_range: tuple
    exozeropt_shift: int or float'''

    # Ensure that the wavelengths of both spectra are increasing
    tel_spec_df = tel_spec_df.sort_values('wavelength',ascending=True)
    exo_spec_df = exo_spec_df.sort_values('wavelength',ascending=True)

    # Restrict the wavelength range of the spectra
    tel_spec_wlrange_df = tel_spec_df[(tel_spec_df['wavelength']>min(wl_range)) & (tel_spec_df['wavelength']<max(wl_range))]
    exo_spec_wlrange_df = exo_spec_df[(exo_spec_df['wavelength']>min(wl_range)) & (exo_spec_df['wavelength']<max(wl_range))]

    # Save the wavelength and flux as arrays
    exo_wl = exo_spec_wlrange_df['wavelength'].values
    exo_flux = exo_spec_wlrange_df['flux'].values
    tel_wl = tel_spec_wlrange_df['wavelength'].values
    tel_flux = tel_spec_wlrange_df['flux'].values

    # Copied from O2_blendcount_760nm_Apr27.py
    # Change the zero-point of the exoplanet spectrum. Some Context:
    # Teluric and Exo files are misaligned, so I must shift exo spec by -83.503 km/sec.
    # This was determined empirically via Redshift_zeropt.py, and error should be less than delta_arr for highest res.

    exo_flux_rs = np.clip(redshift_pad_interp(exo_wl, exo_flux, exozeropt_shift), 0.0, max(exo_flux))
    # and shift it up so that its continuum matches the telluric continuum at Flux = 1
    exo_flux_rs = exo_flux_rs + (np.max(tel_flux) - np.max(exo_flux_rs))

    return(tel_wl, tel_flux, exo_wl, exo_flux_rs)

def redshift_pad_interp(new_lambda, flux_array, velocity):
    """Copied entirely from O2_function_lib3
    This function redshifts the inputted original wavelength grid, extends it to include all values
    of the original wavelength grid, and pads the inputted flux array so that it matches the extended
    wavelength grid. Then it interpolates the redshifted wavelength grid back to the original wavelength array.
    If confused, run code O2_redshift_spectra.py to understand what this is doing.
    Note: run <<from scipy.interpolate import interp1d>> in console before running
    Inputs:
        - new_lambda: original wavelength grid of non-redshifted spectrum
        - flux_array: flux of non-redshifted spectrum. The function assumes this is "gridded" onto new_lambda.
        - velocity (km/s): target velocity to redshift spectra to. Specify negative or positive.
        Input in km/s !!!!
    Outputs:
        - np.array with redshifted flux values in the wavelength grid of new_lambda.
        """
    if velocity == 0:
        return flux_array
    # generate redshifted wavelength grid given velocity, new_lambda
    else:
        c = 299792.0  # speed of light in km/s
        rshed_lam_arr = np.array([])
        for lam in new_lambda:
            # lam += lam * (velocity / c)
            lam += lam * (velocity / c)
            rshed_lam_arr = np.append(rshed_lam_arr, lam)

    # depending of sign of velocity, padding is done differently
    # Assuming v > 0, padding the front of the redshifted spectrum
    if velocity > 0.0:
        for index in range(len(new_lambda)):
            if new_lambda[index] <= rshed_lam_arr[0] and new_lambda[index + 1] >= rshed_lam_arr[0]:
                if np.abs(new_lambda[index] - rshed_lam_arr[0]) < np.abs(new_lambda[index + 1] - rshed_lam_arr[0]):
                    ind_near_min = index
                else:
                    ind_near_min = index + 1
        rshed_lam_arr_pad = np.append(new_lambda[:ind_near_min], rshed_lam_arr)
        # test padded redshifted wavelength array
        if rshed_lam_arr_pad[0] != new_lambda[0]:
            raise ValueError("v > 0: The first value of padded array does not match that of new_lambda.")
        # test expectation for kept value at padding location
        if rshed_lam_arr[0] != rshed_lam_arr_pad[ind_near_min]:
            raise ValueError("v > 0: Kept value at padding location is NOT what expected")
        # test expectation for kept value before padding location
        if new_lambda[ind_near_min - 1] != rshed_lam_arr_pad[ind_near_min - 1]:
            raise ValueError("v > 0: Kept value before padding location is NOT what expected")
        # since all tests passed, then generate padded flux array
        flux_array_pad = np.append(np.ones(len(rshed_lam_arr_pad) - len(new_lambda)), flux_array)

    elif velocity < 0.0:
        for index in range(len(new_lambda)):
            if new_lambda[index] >= rshed_lam_arr[-1] and new_lambda[index - 1] <= rshed_lam_arr[-1]:
                if np.abs(new_lambda[index] - rshed_lam_arr[-1]) < np.abs(new_lambda[index - 1] - rshed_lam_arr[-1]):
                    ind_near_max = index
                else:
                    ind_near_max = index - 1
        rshed_lam_arr_pad = np.append(rshed_lam_arr, new_lambda[ind_near_max + 1:])
        # test padded redshifted wavelength array
        if rshed_lam_arr_pad[-1] != new_lambda[-1]:
            raise ValueError("v < 0: The last value of padded array does not match that of new_lambda.")
        # test expectation for kept value at padding location
        if rshed_lam_arr[-1] != rshed_lam_arr_pad[len(rshed_lam_arr) - 1]:
            raise ValueError("v < 0: Kept value at padding location is NOT what expected")
        # test expectation for kept value after padding location
        if rshed_lam_arr_pad[len(rshed_lam_arr)] != new_lambda[ind_near_max + 1]:
            raise ValueError("v < 0: Kept value after padding location is NOT what expected")
            # since all tests passed, then generate padded flux array
        flux_array_pad = np.append(flux_array, np.ones(len(rshed_lam_arr_pad) - len(new_lambda)))

    # test that padded arrays have same lengths
    if len(rshed_lam_arr_pad) != len(flux_array_pad):
        # print len(rshed_lam_arr_pad)
        # print len(flux_array_pad)
        raise ValueError("Padded wavelength and padded flux arrays DO NOT have same lengths")

    # create function that describes redshifted spectra
    F_flux = interp1d(rshed_lam_arr_pad, flux_array_pad, kind="cubic")
    # interpolate onto good old new_lambda and return
    flx_arr_interp = np.clip(F_flux(new_lambda), 0.0, 1.0)

    return flx_arr_interp





###
### Copied from pyasl broad.py, without the strict linearity requirement ###
###





def broadGaussFast(x, y, sigma, edgeHandling=None, maxsig=None):
    """
    Apply Gaussian broadening.

    This function broadens the given data using a Gaussian
    kernel.

    Parameters
    ----------
    x, y : arrays
        The abscissa and ordinate of the data.
    sigma : float
        The width (i.e., standard deviation) of the Gaussian
        profile used in the convolution.
    edgeHandling : string, {None, "firstlast"}, optional
        Determines the way edges will be handled. If None,
        nothing will be done about it. If set to "firstlast",
        the spectrum will be extended by using the first and
        last value at the start or end. Note that this is
        not necessarily appropriate. The default is None.
    maxsig : float, optional
        The extent of the broadening kernel in terms of
        standard deviations. By default, the Gaussian broadening
        kernel will be extended over the entire given spectrum,
        which can cause slow evaluation in the case of large spectra.
        A reasonable choice could, e.g., be five.

    Returns
    -------
    Broadened data : array
        The input data convolved with the Gaussian
        kernel.
    """
    # Check whether x-axis is linear
    dxs = x[1:] - x[0:-1]
#    if abs(max(dxs) - min(dxs)) > np.mean(dxs) * 1e-6:
#        raise(PE.PyAValError("The x-axis is not equidistant, which is required.",
#                             where="broadGaussFast"))

    if maxsig is None:
        lx = len(x)
    else:
        lx = int(((sigma * maxsig) / dxs[0]) * 2.0) + 1
    # To preserve the position of spectral lines, the broadening function
    # must be centered at N//2 - (1-N%2) = N//2 + N%2 - 1
    nx = (np.arange(lx, dtype=np.int) - sum(divmod(lx, 2)) + 1) * dxs[0]
    gf = fuf.GaussFit1d()
    gf["A"] = 1.0
    gf["sig"] = sigma
    e = gf.evaluate(nx)
    # This step ensured that the
    e /= np.sum(e)

    if edgeHandling == "firstlast":
        nf = len(y)
        y = np.concatenate((np.ones(nf) * y[0], y, np.ones(nf) * y[-1]))
        result = np.convolve(y, e, mode="same")[nf:-nf]
    elif edgeHandling is None:
        result = np.convolve(y, e, mode="same")
    else:
        raise(PE.PyAValError("Invalid value for `edgeHandling`: " + str(edgeHandling),
                             where="broadGaussFast",
                             solution="Choose either 'firstlast' or None"))
    return result


def instrBroadGaussFast(wvl, flux, resolution, edgeHandling=None, fullout=False, maxsig=None):
    """
    Apply Gaussian instrumental broadening.

    This function broadens a spectrum assuming a Gaussian
    kernel. The width of the kernel is determined by the
    resolution. In particular, the function will determine
    the mean wavelength and set the Full Width at Half
    Maximum (FWHM) of the Gaussian to
    (mean wavelength)/resolution.

    Parameters
    ----------
    wvl : array
        The wavelength
    flux : array
        The spectrum
    resolution : int
        The spectral resolution.
    edgeHandling : string, {None, "firstlast"}, optional
        Determines the way edges will be handled. If None,
        nothing will be done about it. If set to "firstlast",
        the spectrum will be extended by using the first and
        last value at the start or end. Note that this is
        not necessarily appropriate. The default is None.
    fullout : boolean, optional
        If True, also the FWHM of the Gaussian will be returned.
    maxsig : float, optional
        The extent of the broadening kernel in terms of
        standard deviations. By default, the Gaussian broadening
        kernel will be extended over the entire given spectrum,
        which can cause slow evaluation in the case of large spectra.
        A reasonable choice could, e.g., be five.

    Returns
    -------
    Broadened spectrum : array
        The input spectrum convolved with a Gaussian
        kernel.
    FWHM : float, optional
        The Full Width at Half Maximum (FWHM) of the
        used Gaussian kernel.
    """
    # Check whether wvl axis is linear
    dwls = wvl[1:] - wvl[0:-1]
#    if abs(max(dwls) - min(dwls)) > np.mean(dwls) * 1e-6:
#        raise(PE.PyAValError("The wavelength axis is not equidistant, which is required.",
#                             where="instrBroadGaussFast"))
    meanWvl = np.mean(wvl)
    fwhm = 1.0 / float(resolution) * meanWvl
    sigma = fwhm / (2.0 * np.sqrt(2. * np.log(2.)))

    result = broadGaussFast(
        wvl, flux, sigma, edgeHandling=edgeHandling, maxsig=maxsig)

    if not fullout:
        return result
    else:
        return (result, fwhm)





###
### Copied from Ian's O2 utils.py
###





def resample(wav, wav_band, R, flux):
    wav_min, wav_max = wav_band
    wav_central = (wav_min + wav_max) / 2
    wav_delta = wav_central / R
    wav_resampled = np.arange(wav_min, wav_max, wav_delta)
    flux_resampled = spectres(wav_resampled, wav, flux)
    return wav_resampled, flux_resampled
