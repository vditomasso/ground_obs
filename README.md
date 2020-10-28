# ground_obs

### A Python package for determining seasonal blending variation between a spectral feature in a transiting exoplanet's atmosphere and the Earth's atmospheric spectral features.

This package was developed to determine the feasibility of detecting O2 in the atmospheres of transiting, terrestrial exoplanets from ground-based telescopes, but can be applied to other optical and IR spectral features. This package builds on the work of [Snellen et al. 2013](https://arxiv.org/pdf/1302.3251.pdf), [Rodler & Lopez-Morales 2014](https://arxiv.org/pdf/1312.1585.pdf) and [Lopez-Morales et al. 2019](https://arxiv.org/pdf/1905.05862.pdf), which show that, although telluric features will undoubtedly blend with the Earth-like atmospheric signatures that we aim to detect on other planets, for some systems, radial velocity (RV)-induced Doppler shifting will deblend the signals enough for us to disentangle them using high-resolution spectroscopy and cross-correlation techniques.

#### To achieve this goal, this package can:
1. Calculate the seasonal variation in radial velocity between the Earth and the target exoplanet due to the Earth's orbit around the Sun (see fig 7 in [Lopez-Morales et al. 2019](https://arxiv.org/pdf/1905.05862.pdf)).

2. Calculate the seasonal variation between the exoplanet and telluric spectral features based on the aforementioned RV variation (see fig 6 in [Lopez-Morales et al. 2019](https://arxiv.org/pdf/1905.05862.pdf)).

3. Determine the times during which telluric blending will be minimal, i.e. the optimal times for observing the target planet in search of the input spectral features.

#### These calculations are done using:
* position of the target exoplanet
* systemic RV of the exoplanet/host star system
* stellar spectrum
* exoplanet transmission spectrum
* telluric spectrum

