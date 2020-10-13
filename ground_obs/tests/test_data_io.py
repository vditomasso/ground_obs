from unittest import TestCase
from ground_obs import data_io
from ground_obs import blended_fraction
import numpy as np
import pandas as pd

exo_spec_filepath = data_io.get_data_file_path('O2_1E6.txt')
tel_spec_filepath = data_io.get_data_file_path('Atm_Transmission_Kurucz_2005.txt')
tel_spec_df, exo_spec_df = data_io.load_data(tel_spec_filepath,exo_spec_filepath)

class test_get_data_file_path(TestCase):

    def test_get_data_file_path_output_type(self):
        self.assertTrue(isinstance(exo_spec_filepath, str))
        self.assertTrue(isinstance(tel_spec_filepath, str))

class test_load_data(TestCase):
    
    def test_load_data_output_type(self):
        self.assertTrue(isinstance(tel_spec_df, pd.core.frame.DataFrame))
        