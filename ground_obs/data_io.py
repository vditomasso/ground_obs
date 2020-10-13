import os
import pandas as pd

def get_data_file_path(filename, data_dir='data'):
    '''Get filepath to data
    Input:
    ---
    filename: str
    data_dir: str (optional)
        Default: 'data'

    Returns:
    ---
    filepath: str
        Path to data file specified by filename.
        '''

    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)
    data_dir = os.path.join(start_dir, data_dir)
    return os.path.join(start_dir, data_dir, filename)

def load_data(tel_spec_filepath, exo_spec_filepath):

    '''Load in wavlength and flux for exoplanet and telluric spectra, very specific to Atm_Transmission_Kurucz_2005.txt and O2_1E6.txt
    Input:
    ---
    tel_spec_filepath: str
    exo_spec_filepath: str

    Returns:
    ---
    tel_spec_df: Pandas DataFrame
    exo_spec_df: Pandas DataFrame
    '''

    tel_spec_df = pd.read_csv(tel_spec_filepath, skipinitialspace=True, skiprows=0, header=0, delim_whitespace=True,
                           names=['wavelength', 'flux'], index_col=False)
    exo_spec_df = pd.read_csv(exo_spec_filepath, delim_whitespace=True, header=0, skiprows=[0, 1, 2, 3], index_col=False,
                           names=['wavenumber', 'wavelength', 'flux'])

    return(tel_spec_df, exo_spec_df)
