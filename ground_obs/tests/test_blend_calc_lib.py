from unittest import TestCase
from ground_obs import data_io
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

class test_prep_wl_flux_arrays(TestCase):

    def test_tel_wl_range(self):
        self.assertGreaterEqual(min(tel_wl),min(wl_range))
        self.assertLessEqual(max(tel_wl),max(wl_range))

    def test_wl_increasing(self):
        self.assertTrue((tel_wl == sorted(tel_wl)).all())
        self.assertTrue((exo_wl == sorted(exo_wl)).all())

    def test_output_type(self):
        self.assertTrue(isinstance(tel_wl, np.ndarray))
        self.assertTrue(isinstance(tel_flux, np.ndarray))
        self.assertTrue(isinstance(exo_wl, np.ndarray))
        self.assertTrue(isinstance(exo_flux, np.ndarray))
