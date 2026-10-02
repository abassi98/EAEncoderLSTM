"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., & Nearing, G. S. ( 2019). 
Toward improved predictions in ungauged basins: Exploiting the power of machine learning.
Water Resources Research, 55. https://doi.org/10.1029/2019WR026065 

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import sqlite3
from pathlib import Path, PosixPath
from typing import List, Tuple

import numpy as np
import pandas as pd
from numba import njit


#CAMELS catchment characteristics ignored in this study
INVALID_ATTR = [
    'gauge_name', 'area_geospa_fabric', 'geol_1st_class', 'glim_1st_class_frac', 'geol_2nd_class',
    'glim_2nd_class_frac', 'dom_land_cover_frac', 'dom_land_cover', 'high_prec_timing',
    'low_prec_timing', 'huc', 'q_mean', 'runoff_ratio', 'stream_elas', 'slope_fdc',
    'baseflow_index', 'hfd_mean', 'q5', 'q95', 'high_q_freq', 'high_q_dur', 'low_q_freq',
    'low_q_dur', 'zero_q_freq', 'geol_porostiy', 'root_depth_50', 'root_depth_99', 'organic_frac',
    'water_frac', 'other_frac', 'p_seasonality'
]

STD_ORDER = ['soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity',
       'soil_conductivity', 'max_water_content', 'sand_frac', 'silt_frac',
       'clay_frac', 'p_mean', 'pet_mean', 'frac_snow', 'aridity',
       'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur',
       'frac_forest', 'lai_max', 'lai_diff', 'gvf_max', 'gvf_diff',
        'elev_mean', 'slope_mean', 'area_gages2',
       'carbonate_rocks_frac', 'geol_permeability']

# = SOIL_ATTRS + CLIM_ATTRS + VEGE_ATTRS + TOPO_ATTRS + GEOL_ATTRS

CLIM_ATTRS = [
    'p_mean', 'pet_mean', 'frac_snow','aridity', 'high_prec_freq', 'high_prec_dur', 
    'low_prec_freq', 'low_prec_dur'
]

LANDSCAPE_ATTRS = [
    'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
    'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 
     'carbonate_rocks_frac',  'geol_permeability',
     'frac_forest', 'lai_max', 'lai_diff', 'gvf_max', 'gvf_diff',
     'elev_mean', 'slope_mean', 'area_gages2'
    ]

HYDRO_ATTRS = [
    'q_mean', 'runoff_ratio', 'stream_elas', 'slope_fdc',
    'baseflow_index', 'hfd_mean', 'q5', 'q95', 'high_q_freq', 'high_q_dur', 'low_q_freq',
    'low_q_dur'
]

SOIL_ATTRS = [
    'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
    'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 
]

TOPO_ATTRS = [
    'elev_mean', 'slope_mean', 'area_gages2'
]

VEGE_ATTRS = [
    'frac_forest', 'lai_max', 'lai_diff', 'gvf_max', 'gvf_diff',
]

GEOL_ATTRS = [
    'carbonate_rocks_frac',  'geol_permeability',
]

#SOIL_ATTRS + CLIM_ATTRS + VEGE_ATTRS + TOPO_ATTRS + GEOL_ATTRS

OTHER = ['soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
    'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac',
    'p_mean', 'pet_mean', 'high_prec_freq', 'high_prec_dur', 
    'low_prec_freq', 'frac_forest', 'lai_max', 'lai_diff', 'gvf_max', 'gvf_diff', 'slope_mean', 'area_gages2', 'carbonate_rocks_frac',  'geol_permeability']

# Daymet, Maurer extended and NLDA extended mean/std calculated over all basins in period 01.10.1999 until 30.09.2008
SCALER = {
    'input_means': np.array([3.31320548, 339.07974725, 16.79154184, 4.58007576, 977.38776331, 
                             3.19255517, 372.22696801, 17.34537588, 3.98892497, 925.3947897, 
                             3.0865477, 358.29964531 , 16.36210656, 7.09117542, 1088.47310759] ),
    'input_stds': np.array([7.94811627, 130.6653107, 10.9400193, 9.94505455, 658.87274362,
                            7.0045609, 131.69828102, 10.84375894, 10.36779269, 628.90686392,
                            7.93875686, 129.97207408, 11.18196999, 9.93380347, 709.00044057]),
    'output_mean': np.array([1.51463497]),
    'output_std': np.array([3.69925263])
}

def get_variable_name(var):
    for name, value in globals().items():
        if value is var:
            return name
    return None

def add_camels_attributes(camels_root: PosixPath, db_path: str = None):
    """Load catchment characteristics from txt files and store them in a sqlite3 table
    
    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    db_path : str, optional
        Path to where the database file should be saved. If None, stores the database in the 
        `data` directory in the main folder of this repository., by default None
    
    Raises
    ------
    RuntimeError
        If CAMELS attributes folder could not be found.
    """
    attributes_path = Path(camels_root) / 'camels_attributes_v2.0'

    if not attributes_path.exists():
        raise RuntimeError(f"Attribute folder not found at {attributes_path}")

    txt_files = attributes_path.glob('camels_*.txt')

    # Read-in attributes into one big dataframe
    df = None
    for f in txt_files:
        df_temp = pd.read_csv(f, sep=';', header=0, dtype={'gauge_id': str})
        df_temp = df_temp.set_index('gauge_id')

        if df is None:
            df = df_temp.copy()
        else:
            df = pd.concat([df, df_temp], axis=1)

    # convert huc column to double digit strings
    df['huc'] = df['huc_02'].apply(lambda x: str(x).zfill(2))
    df = df.drop('huc_02', axis=1)

    if db_path is None:
        db_path = str(Path(__file__).absolute().parent.parent / 'data' / 'attributes.db')

    with sqlite3.connect(db_path) as conn:
        # insert into databse
        df.to_sql('basin_attributes', conn)

    print(f"Sucessfully stored basin attributes in {db_path}.")


def load_attributes(db_path: str,
                    basins: List,
                    keep_attributes: List,) -> pd.DataFrame:
    """Load attributes from database file into DataFrame

    Parameters
    ----------
    db_path : strd
        Path to sqlite3 database file
    basins : List
        List containing the 8-digit USGS gauge id
    drop_lat_lon : bool
        If True, drops latitude and longitude column from final data frame, by default True
    keep_attributes : List
        A pd.DataFrame containing these attributes will be returned.

    Returns
    -------
    pd.DataFrame
        Attributes in a pandas DataFrame. Index is USGS gauge id. Latitude and Longitude are
        transformed to x, y, z on a unit sphere.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM 'basin_attributes'", conn, index_col='gauge_id')

    # drop rows of basins not contained in data set
    drop_basins = [b for b in df.index if b not in basins]
    df = df.drop(drop_basins, axis=0)

    # filter attributes
    df = df[keep_attributes]
    # drop_names = [c for c in df.columns if c in INVALID_ATTR]
    # df = df.drop(drop_names, axis=1)

    df.index.name = None
   
    return df

def load_signatures(signature_file) -> pd.DataFrame:
    """Load a hydrological-signature table indexed by 8-digit USGS gauge id.

    Accepts basin ids either in a ``basin`` column (the layout written for
    analysis/signatures/camels_us_*.csv) or in the first column.
    """
    df = pd.read_csv(signature_file, dtype={"basin": str})
    if "basin" in df.columns:
        df = df.drop(columns=[col for col in df.columns if col.startswith("Unnamed")])
        df = df.set_index("basin")
    else:
        df = df.set_index(df.columns[0])
    df.index = [str(basin).zfill(8) for basin in df.index]
    if df.index.has_duplicates:
        raise ValueError(f"Duplicate basin ids in {signature_file}")
    return df


def normalize_features(feature: np.ndarray, variable: str) -> np.ndarray:
    """Normalize features using global pre-computed statistics.

    Parameters
    ----------
    feature : np.ndarray
        Data to normalize
    variable : str
        One of ['inputs', 'output'], where `inputs` mean, that the `feature` input are the model
        inputs (meteorological forcing data) and `output` that the `feature` input are discharge
        values.

    Returns
    -------
    np.ndarray
        Normalized features

    Raises
    ------
    RuntimeError
        If `variable` is neither 'inputs' nor 'output'
    """
    if variable == 'inputs':
        feature = (feature - SCALER["input_means"]) / SCALER["input_stds"]
    elif variable == 'output':
        feature = (feature - SCALER["output_mean"]) / SCALER["output_std"]
    else:
        raise RuntimeError(f"Unknown variable type {variable}")
  
    return feature


def rescale_features(feature: np.ndarray, variable: str) -> np.ndarray:
    """Rescale features using global pre-computed statistics.

    Parameters
    ----------
    feature : np.ndarray
        Data to rescale
    variable : str
        One of ['inputs', 'output'], where `inputs` mean, that the `feature` input are the model
        inputs (meteorological forcing data) and `output` that the `feature` input are discharge
        values.

    Returns
    -------
    np.ndarray
        Rescaled features

    Raises
    ------
    RuntimeError
        If `variable` is neither 'inputs' nor 'output'
    """
    if variable == 'inputs':
        feature = feature * SCALER["input_stds"] + SCALER["input_means"]
    elif variable == 'output':
        feature = feature * SCALER["output_std"] + SCALER["output_mean"]
    else:
        raise RuntimeError(f"Unknown variable type {variable}")

    return feature


@njit
def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape data into LSTM many-to-one input samples

    Parameters
    ----------
    x : np.ndarray
        Input features of shape [num_samples, num_features]
    y : np.ndarray
        Output feature of shape [num_samples, 1]
    seq_length : int
        Length of the requested input sequences.

    Returns
    -------
    x_new: np.ndarray
        Reshaped input features of shape [num_samples*, seq_length, num_features], where 
        num_samples* is equal to num_samples - seq_length + 1, due to the need of a warm start at
        the beginning
    y_new: np.ndarray
        The target value for each sample in x_new
    """
    num_samples, num_features = x.shape
    
    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]

    return x_new, y_new



def load_forcings(camels_root: PosixPath, basin: str, forcing: str,
                  file_path: Path = None) -> Tuple[pd.DataFrame, int]:
    """Load NLDAS forcing data from text files.

    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    basin : str
        8-digit USGS gauge id
    forcings: str
        forcings product (daymet, maurer, nldas_extended)
    file_path: Path, optional
        Pre-indexed forcing file; avoids searching the directory for each basin.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the Maurer forcing
    area: int
        Catchment area (read-out from the header of the forcing file)

    Raises
    ------
    RuntimeError
        If not forcing file was found.
    """
    if file_path is None:
        forcing_path = camels_root / 'basin_mean_forcing' / forcing
        if not forcing_path.is_dir():
            raise OSError(f"{forcing_path} does not exist")

        files = list(forcing_path.glob('**/*_forcing_leap.txt'))
        matches = [f for f in files if f.name[:8] == basin]
        if not matches:
            raise RuntimeError(f'No file for Basin {basin} at {forcing_path}')
        file_path = matches[0]
    
    df = pd.read_csv(file_path, sep=r'\s+', header=3)
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")
    
    # select five meteorological forcing variables
    dyn_variables = ["prcp", "srad", "tmax", "tmin", "vp"]
    filtered_columns = [col for col in df.columns if any(col.lower().startswith(dyn_var) for dyn_var in dyn_variables)]
    df = df[filtered_columns]

    # load area from header
    with open(file_path, 'r') as fp:
        next(fp)
        next(fp)
        area = int(next(fp))

    return df, area


def load_discharge(camels_root: PosixPath, basin: str, area: int) -> pd.Series:
    """[summary]

    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    basin : str
        8-digit USGS gauge id
    area : int
        Catchment area, used to normalize the discharge to mm/day

    Returns
    -------
    pd.Series
        A Series containing the discharge values.

    Raises
    ------
    RuntimeError
        If no discharge file was found.
    """
    discharge_path = camels_root / 'usgs_streamflow'
    files = list(discharge_path.glob('**/*_streamflow_qc.txt'))
    file_path = [f for f in files if f.name[:8] == basin]
    if len(file_path) == 0:
        raise RuntimeError(f'No file for Basin {basin} at {file_path}')
    else:
        file_path = file_path[0]

    col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=col_names)
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    # normalize discharge from cubic feed per second to mm per day
    df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10**6)

    return df.QObs
