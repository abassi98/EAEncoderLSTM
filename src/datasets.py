
from pathlib import PosixPath
from typing import List, Tuple

#import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import h5py

from .datautils import (CLIM_ATTRS, SOIL_ATTRS, VEGE_ATTRS, TOPO_ATTRS, GEOL_ATTRS, load_attributes, load_discharge, load_forcings, normalize_features,
                        reshape_data)


class CamelsTXT(Dataset):
    """PyTorch data set to work with the raw text files in the CAMELS data set.
       
    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    basin : str
        8-digit usgs-id of the basin
    dates : List
        Start and end date of the period.
    is_train : bool
        If True, discharge observations are normalized and invalid discharge samples are removed
    seq_length : int, optional
        Length of the input sequence, by default 270
    with_attributes : bool, optional
        If True, loads and returns addtionaly attributes, by default False
    attribute_means : pd.Series, optional
        Means of catchment characteristics, used to normalize during inference, by default None
    attribute_stds : pd.Series, optional
        Stds of catchment characteristics, used to normalize during inference,, by default None
    concat_static : bool, optional
        If true, adds catchment characteristics at each time step to the meteorological forcing
        input data, by default False
    db_path : str, optional
        Path to sqlite3 database file, containing the catchment characteristics, by default None
    """

    def __init__(self,
                 camels_root: PosixPath,
                 basin: str,
                 dates: List,
                 is_train: bool,
                 seq_length: int = 365,
                 with_attributes: bool = False,
                 attribute_means: pd.Series = None,
                 attribute_stds: pd.Series = None,
                 db_path: str = None,
                 attr_left_out: List = [],
                 forcings: list = ["daymet", "maurer_extended", "nldas_extended"],
                 reshape_data: bool = True):
        
        self.camels_root = camels_root
        self.basin = basin
        self.seq_length = seq_length
        self.is_train = is_train
        self.dates = dates
        self.with_attributes = with_attributes
        self.attribute_means = attribute_means
        self.attribute_stds = attribute_stds
        self.db_path = db_path
        self.attr_left_out = attr_left_out
        self.forcings = forcings
        self.reshape_data = reshape_data

        if (isinstance(self.forcings, list)) and (len(self.forcings) == 1):
            self.forcings = self.forcings[0]
        else:
            self.forcings =self.forcings


        # placeholder to store std of discharge, used for rescaling losses during training
        self.q_std = None

        # placeholder to store start and end date of entire period (incl warmup)
        self.period_start = None
        self.period_end = None
        self.attribute_names = None

        self.x, self.y = self._load_data()

        if self.with_attributes:
            self.attributes = self._load_attributes()

        self.num_samples = self.x.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        if self.with_attributes:
            return self.x[idx], self.y[idx], self.attributes
        else:
            return self.x[idx], self.y[idx]

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load input and output data from text files."""
        # get forcings
        if isinstance(self.forcings, list):
            dfs = []
            for forcing in self.forcings:
                df, area = load_forcings(self.camels_root, self.basin, forcing)
                # rename columns
                df = df.rename(columns={col: f"{col}_{forcing}" for col in df.columns})
                dfs.append(df)
            df = pd.concat(dfs, axis=1)
        else:
            df, area = load_forcings(self.data_dir, self.basin, self.forcings)

        df['QObs(mm/d)'] = load_discharge(self.camels_root, self.basin, area)
       
        # we use (seq_len) time steps before start for warmup
        start_date = self.dates[0] - pd.DateOffset(days=self.seq_length - 1)
        end_date = self.dates[1]
        df = df[start_date:end_date]

        # store first and last date of the selected period (including warm_start)
        self.period_start = df.index[0]
        self.period_end = df.index[-1]

        # use all meteorological variables as inputs
        x = np.array(df.iloc[:,:5*len(self.forcings)])
        y = np.array([df['QObs(mm/d)'].values]).T

        # normalize data, reshape for LSTM training and remove invalid sample
        x = normalize_features(x, variable='inputs')
        if self.reshape_data:
            x, y = reshape_data(x, y, self.seq_length)
        
        if self.is_train:
            # Deletes all records where no discharge was measured (-999)
            x = np.delete(x, np.argwhere(y < 0)[:, 0], axis=0)
            y = np.delete(y, np.argwhere(y < 0)[:, 0], axis=0)

            # Delete all samples, where discharge is NaN 
            if np.sum(np.isnan(y)) > 0:
                arg_nans = np.isnan(y)
                print(
                    f"Deleted {np.sum(np.isnan(y))} of {len(y)} records because of NaNs in basin {self.basin}"
                )
                x = np.delete(x, np.argwhere(arg_nans), axis=0)
                y = np.delete(y, np.argwhere(arg_nans), axis=0)

            # store std of discharge before normalization
            self.q_std = np.std(y)

            y = normalize_features(y, variable='output')

        # convert arrays to torch tensors
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        return x, y

    def _load_attributes(self) -> torch.Tensor:
        KEEP_ATTRS = SOIL_ATTRS + CLIM_ATTRS + VEGE_ATTRS + TOPO_ATTRS + GEOL_ATTRS
        KEEP_ATTRS = [attr for attr in KEEP_ATTRS if attr not in  self.attr_left_out]
        df = load_attributes(self.db_path, [self.basin], keep_attributes=KEEP_ATTRS)
            
        # normalize data
        df = (df - self.attribute_means) / self.attribute_stds
        # store attribute names
        self.attribute_names = df.columns

        # store feature as PyTorch Tensor
        attributes = df.loc[df.index == self.basin].values
        return torch.from_numpy(attributes.astype(np.float32))


class CamelsH5(Dataset):
    """PyTorch data set to work with pre-packed hdf5 data base files.

    Should be used only in combination with the files processed from `create_h5_files` in the 
    `papercode.utils` module.

    Parameters
    ----------
    h5_file : PosixPath
        Path to hdf5 file, containing the bundled data
    basins : List
        List containing the 8-digit USGS gauge id
    db_path : str
        Path to sqlite3 database file, containing the catchment characteristics
    """

    def __init__(self,
                 h5_file: PosixPath,
                 basins: List,
                 db_path: str,
                 attr_left_out: List = [],):
        self.h5_file = h5_file
        self.basins = basins
        self.db_path = db_path
        self.attr_left_out = attr_left_out # list of static attributes to leave out

        # Placeholder for catchment attributes stats
        self.df = None
        self.attribute_means = None
        self.attribute_stds = None
        self.attribute_names = None

       
        # load attributes into data frame
        self._load_attributes()

        # determine number of samples once
        with h5py.File(h5_file, 'r') as f:
            self.num_samples = f["target_data"].shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        with h5py.File(self.h5_file, 'r') as f:
            x = f["input_data"][idx]
            y = f["target_data"][idx]
            basin = f["sample_2_basin"][idx]
            basin = basin.decode("ascii")
            q_std = f["q_stds"][idx]

        # get attributes from data frame and create 2d array with copies
        attributes = self.df.loc[self.df.index == basin].values
        attributes = torch.from_numpy(attributes.astype(np.float32))

        # convert to torch tensors
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))
        q_std = torch.from_numpy(q_std)

        return x, attributes, y, q_std

    def _get_basins(self):
        with h5py.File(self.h5_file, 'r') as f:
            str_arr = f["sample_2_basin"][:]
        str_arr = [x.decode("ascii") for x in str_arr]
        basins = list(set(str_arr))
        return basins

    def _load_attributes(self):
        KEEP_ATTRS = SOIL_ATTRS + CLIM_ATTRS + VEGE_ATTRS + TOPO_ATTRS + GEOL_ATTRS
        KEEP_ATTRS = [attr for attr in KEEP_ATTRS if attr not in  self.attr_left_out]
        df = load_attributes(self.db_path, self.basins, keep_attributes=KEEP_ATTRS)
    
        # store means and stds
        self.attribute_means = df.mean()
        self.attribute_stds = df.std()
        
        # normalize data
        df = (df - self.attribute_means) / self.attribute_stds

        self.attribute_names = df.columns
        self.df = df

    def get_attribute_means(self) -> pd.Series:
        """Return means of catchment attributes
        
        Returns
        -------
        pd.Series
            Contains the means of each catchment attribute
        """
        return self.attribute_means

    def get_attribute_stds(self) -> pd.Series:
        """Return standard deviation of catchment attributes
        
        Returns
        -------
        pd.Series
            Contains the stds of each catchment attribute
        """
        return self.attribute_stds






