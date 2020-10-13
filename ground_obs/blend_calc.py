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
from ground_obs import blend_calc_lib
from scipy import interpolate


# Create km/s unit
km_per_s = u.km / u.s
m_per_s = u.m / u.s

def rv_blended_frac(rv_df, tel_spec_df, exo_spec_df, resolution=np.array([10**5]), wl_range=[750,780], exozeropt_shift=-83.503):
    '''Calculate the corresponding fraction of blended spectra lines for given telluric and exoplanet spectra at a given resolution and wavelength range
    Maybe this functionality should be added with a kwarg into the rv function?

    Inputs:
    ---
    rv_df: pandas DataFrame
        Columns: 'date', 'rv'. Output of rv function
    tel_spec_df: pandas Dataframe, columns ['wavelength','flux']
    exo_spec_df: pandas Dataframe, columns ['wavelength','flux']
    resolution: tuple (optional)
        Can only do one resolution at a time, but will expand to multiple resolutions. Default: np.array([10**5])
    wl_range: tuple (optional)
        Default: [750,780]
    exozeropt_shift: int or float in km/s
        Default: -83.503

    Return:
    ---
    rv_frac_df: pandas dataframe
        columns: date (jd), rv (in km/s), blend_frac
    '''

    # Construct vel_range for blended_fraction.blended_fraction
    vel_min = np.round(rv_df['rv'].min())-1
    vel_max = np.round(rv_df['rv'].max())+1
    vel_range = [vel_min, vel_max]

    # Run blended_fraction.blended_fraction to get 'vel' and 'blend_frac'
    blend_frac_df = blended_fraction(tel_spec_df, exo_spec_df, vel_range=vel_range, resolution=resolution, wl_range=wl_range, exozeropt_shift=exozeropt_shift)

    # Interpolate blended fraction onto the input rvs
    f = interpolate.interp1d(blend_frac_df['vel'].values,blend_frac_df['blend_frac'].values)
    rv_df_blend_frac = f(rv_df['rv'].values)

    # Save those values as a new column on rv_df
    rv_df['blend_frac'] = rv_df_blend_frac

    return(rv_df)

def rv(obj_name, v_sun_star, obstime_range=['2020-01-01','2020-12-31'], obsloc='Las Campanas Observatory'):
    '''Calculate the relative velocity between an exoplanet and the Earth
    Inputs
    ---
    obj_name: string
        The name of the object (e.g. star) that you want the coordinates of
    v_sun_star: integer or float in m/s
        The relative velocity between the Sun and the exoplanet system
    obstime_range: list of strings, optional
        A list containing the beginning and end date for the range of time that you're interested in calculating the
        RV between the Earth and the exoplanet of interest for. Defaults to ['2020-01-01','2020-12-31'].
    obsloc: string, optional
        Name of relevant observatory. Defaults to 'Las Campanas Observatory'. Not sure how much this affects the
        results.

    Returns
    ---
    rv_df: pandas dataframe
        columns: date (jd), rv (in km/s)
    '''

    # Give v_sun_star units
    v_sun_star = v_sun_star*m_per_s

    # Find the location of the observatory
    loc = coord.EarthLocation.of_site(obsloc)

    # Find the coordinates of the object (e.g. host star)
    sc = SkyCoord.from_name(obj_name)

    # Create an astropy time object with the specified range of times
    t1 = Time(obstime_range[0])
    t1.format = 'jd'  # 'jd' could be a function input

    t2 = Time(obstime_range[-1])
    t2.format = 'jd'  # 'jd' could be a function input

    times = Time(np.arange(t1.value, t2.value), format='jd')

    with coord.solar_system_ephemeris.set('jpl'):  # 'jpl' could be a function input
        rv_corrs = sc.radial_velocity_correction(obstime=times, location=loc)

    # Calculate Earth-centric RVs in km/s
    v_star = (v_sun_star - rv_corrs).to(km_per_s)

    # Get rid of time's unit
    times = times.value

    rv_df = pd.DataFrame({"date":times,'rv':v_star})

    return(rv_df)

def blended_fraction(tel_spec_df, exo_spec_df, vel_range=[-25,25], resolution=np.array([10**5]), wl_range=[750,780], exozeropt_shift=-83.503):
    '''Functionalized, but fundamentally unchanged from O2_blendcount_760nm_Apr27.py
    Calculate the fraction of blended spectral features between a telluric spectrum and an exoplanet spectrum at (a) given resolution(s) and wavelength range.
    Need to provide an initial shift value between the exoplanet and the telluric spectrum.
    ZEROPT SHOULD EVENTUALLY DEFAULT TO 0

    Inputs:
    ---
    tel_spec_df: pandas Dataframe, columns ['wavelength','flux']
    exo_spec_df: pandas Dataframe, columns ['wavelength','flux']
    vel_range: tuple (optional)
        Default: [-25,25]
    resolution: tuple (optional)
        Default: np.array([10**5])
    wl_range: tuple (optional)
        Default: [750,780]
    exozeropt_shift: int or float in km/s
        Default: -83.503

    Returns:
    ---
    blend_frac_df: pandas DataFrame
        columns: 'vel', 'blend_frac'
    '''

    ##############################################################################
    # VALLEY IDENTIFICATION (valley = negative peak)

    # Start peak identification
    # State the minimum peak height and minimum peak distance for the peak ID algortihm below
    # mph = 0.07 # vals discussed with Mercedes - March 6 2018

    tel_lambda, tel_flx, O22_lambda, O22_flux_rs = blend_calc_lib.prep_wl_flux_arrays(tel_spec_df, exo_spec_df, wl_range, exozeropt_shift)

    mean = np.mean(O22_lambda)

    print(fr'Starting code for O2 band near {np.round(mean)} nm...')
    start_time = timeit.default_timer()

    # ==============================================================================
    # # ID Telluric valleys (aka negative peaks)
    # e_indpeaks = detect_peaks(tel_flx - np.max(tel_flx), mph = mph, mpd = mpd, valley=True, show=True) # must write mpd and mph explicitly
    # lam_eindpeaks = tel_lambda[e_indpeaks] # wavelength value at each peak
    # # Fix April 27 - append all locations where tel_flux = 0 as valleys
    # lam_eindpeaks = np.append(lam_eindpeaks, tel_lambda[np.argwhere(tel_flx <= 0.3)])
    #
    # flx_eindpeaks = tel_flx[e_indpeaks]
    #
    # print 'Amount of peaks in Telluric: '+str(len(e_indpeaks))
    # ==============================================================================

    # define resolution element array, and velocity array
    delta_arr = mean / resolution
    target_res = resolution

    mpd = 1
    flag_lam_arr = [0.22]  # optimized for min mphval # VD look into this, was this resolution specific?
    mph_arr = [0.3]  # VD only one resolution

    vel_array = np.arange(min(vel_range), max(vel_range)+0.5, 0.5)  # VD changed range to be input

    # Define empty arrays to fill up with loop below
    count_arr = np.array([])
    count_per_delta = np.array([])
    exo_peaks = np.array([])  # change at each resolution

    resolved_frac = np.array([])
    blend_frac = np.array([])

    # loop to ID amount of peaks at each velocity and quantify blend fraction
    for ind in range(len(delta_arr)):
        print('Starting blend frac calculations for R =' + str(target_res[ind]))
        # ID Telluric valleys at R (aka negative peaks)
        broadtelflx = pyasl.instrBroadGaussFast(tel_lambda, tel_flx, target_res[ind], edgeHandling='firstlast',
                                                fullout=False, maxsig=None)
        e_indpeaks = detect_peaks(broadtelflx - np.max(broadtelflx), mph=mph_arr[ind], mpd=mpd, valley=True,
                                  show=False)  # must write mpd and mph explicitly
        lam_eindpeaks = tel_lambda[e_indpeaks]  # wavelength value at each peak
        # Fix April 27 - append all locations where tel_flux <= 0.3 as valleys
        lam_eindpeaks = np.append(lam_eindpeaks, tel_lambda[np.argwhere(tel_flx <= flag_lam_arr[ind])])

        # print 'Amount of peaks in Telluric: '+str(len(e_indpeaks))

        broad_flux = pyasl.instrBroadGaussFast(O22_lambda, O22_flux_rs, target_res[ind], edgeHandling='firstlast',
                                               fullout=False, maxsig=None)
        count_arr = np.array([])
        # tot_peaks_atR = len(detect_peaks(broad_flux - np.max(broad_flux), mph = mph_arr[ind], mpd = mpd, valley=True, show=True))

        peakinds_atR = detect_peaks(broad_flux - np.max(broad_flux), mph=mph_arr[ind], mpd=mpd, valley=True, show=False)
        fluxpeaks_atR = broad_flux[peakinds_atR]
        tot_peaks_atR = np.sum(1.0 - fluxpeaks_atR)

        exo_peaks = np.append(exo_peaks, tot_peaks_atR)
        # print 'Amount of peaks in Exoplanet Spectrum at R(above): '+str(tot_peaks_atR)

        for vel in vel_array:
            count = 0
            exo_flux = np.clip(blend_calc_lib.redshift_pad_interp(O22_lambda, broad_flux, vel), 0.0, max(O22_flux_rs))
            exo_indpeaks = detect_peaks(exo_flux - max(exo_flux), mph=mph_arr[ind], mpd=mpd, valley=True, show=False)
            lam_exoindpeaks = O22_lambda[exo_indpeaks]  # wavelength value at each peak
            flux_exoindpeaks = exo_flux[exo_indpeaks]
            # this loop can be made WAY more efficient. Right now it is very thorough
            # diff_array = np.array([])
            # at each redshift, and for each peak identify peak in telluric (rest frame) closest to it
            for peak_index in range(len(lam_exoindpeaks)):
                diff_arr = np.abs(lam_exoindpeaks[peak_index] - lam_eindpeaks)
                if min(diff_arr) >= delta_arr[ind]:
                    count += (1.0 - flux_exoindpeaks[peak_index])
                # if tel peak is closer to exo peak than one resolution element, then not resolvable and add zero.
                else:
                    count += 0.0
            count_arr = np.append(count_arr, count)
        # populate 1D arrays with resolved fraction, blended fraction and count per delta
        resolved_frac = np.append(resolved_frac, count_arr / tot_peaks_atR)
        blend_frac = np.append(blend_frac, (tot_peaks_atR - count_arr) / tot_peaks_atR)
        count_per_delta = np.append(count_per_delta, count_arr)

    vel_array = np.asarray(list(vel_array)*len(resolution))

    elapsed = timeit.default_timer() - start_time

    print('Execution time in hours: ' + str(elapsed / 3600.0))

    blend_frac_df = pd.DataFrame({'vel':vel_array,'blend_frac':blend_frac})

    return(blend_frac_df)
