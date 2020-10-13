from unittest import TestCase
from ground_obs import data_io
from ground_obs import blend_calc
from ground_obs import blend_calc_lib
import numpy as np
import pandas as pd

telluric_spec_file = 'Atm_Transmission_Kurucz_2005.txt'
exo_spec_file = 'O2_1E6.txt'

tel_spec_df, exo_spec_df = data_io.load_data(data_io.get_data_file_path(telluric_spec_file), data_io.get_data_file_path(exo_spec_file))

vel_range = [-2,2]
resolution = np.asarray([2])
wl_range = [755,760]
exozeropt_shift=-83.503

tel_wl, tel_flux, exo_wl, exo_flux = blend_calc_lib.prep_wl_flux_arrays(tel_spec_df, exo_spec_df, wl_range, exozeropt_shift)

blend_frac_df = blend_calc.blended_fraction(tel_spec_df, exo_spec_df, vel_range, resolution, wl_range, exozeropt_shift)

class test_blended_fraction(TestCase):

    def test_vel_range(self):
        self.assertEqual(blend_frac_df['vel'].max(),max(vel_range))
        self.assertEqual(blend_frac_df['vel'].min(),min(vel_range))

    def test_blend_fraction_range(self):
        self.assertTrue((blend_frac_df['blend_frac'].min() >= 0)&(blend_frac_df['blend_frac'].max() <= 1))

    def test_output_type(self):
        self.assertTrue(isinstance(blend_frac_df, pd.core.frame.DataFrame))

rv_df = blend_calc.rv('LTT1445',2.11,obstime_range=['2020-01-01','2020-01-02'])

class test_rv(TestCase):

    def test_output_type(self):
        self.assertTrue(isinstance(rv_df, pd.core.frame.DataFrame))

    def test_rv_range(self):
        self.assertTrue((rv_df['rv'].min()>=(2.11-30))&(rv_df['rv'].max()<=(2.11+30)))

rv_df_frac = blend_calc.rv_blended_frac(rv_df, tel_spec_df, exo_spec_df, resolution=resolution, wl_range=wl_range, exozeropt_shift=exozeropt_shift)

class test_rv_blended_frac(TestCase):

    def test_output_type(self):
        self.assertTrue(isinstance(rv_df, pd.core.frame.DataFrame))

    def test_blend_fraction_range(self):
        self.assertTrue((rv_df_frac['blend_frac'].min() >= 0)&(rv_df_frac['blend_frac'].max() <= 1))
