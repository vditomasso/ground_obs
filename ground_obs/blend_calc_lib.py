import numpy as np
import timeit
from detecta import detect_peaks
from PyAstronomy import pyasl
from scipy.interpolate import interp1d
import pandas as pd

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