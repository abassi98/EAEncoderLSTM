"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., & Nearing, G. S. ( 2019). 
Toward improved predictions in ungauged basins: Exploiting the power of machine learning.
Water Resources Research, 55. https://doi.org/10.1029/2019WR026065 

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
import sys
from pathlib import Path, PosixPath
from typing import List
import warnings 
import h5py
import numpy as np
import argparse
from tqdm import tqdm
from pytorch_lightning import Callback
import os 
import torch
import torch.nn as nn
import copy
import warnings

from .datasets import CamelsTXT
from torch.optim.lr_scheduler import _LRScheduler
import pandas as pd
import math

def retrieve_seeds(df : pd.DataFrame) -> List:
    seeds = []
    for col in df.columns:
        if "seed" in col:
            seed = col.split('seed')[1]
            seeds.append(seed)
    return list(set(seeds))

def update_metric_df(df, experiment, eval_period, encoded_features, metric):
    experiment_name = experiment.replace('_', ' ').upper().replace("AE", "").strip()
    ### retrieve metrics
    stats = pd.read_csv(f"analysis/stats/{eval_period}/{experiment}_es{encoded_features}.csv", sep=",", index_col=0)
    stats.index = [str(s).rjust(8,"0") for s in stats.index]
    
    ### retrieve NSE  Distribution
    if encoded_features != None:
        df[f"{experiment_name}-{encoded_features}"] = stats[metric]
    else:
        df[f"{experiment_name}-A"] = stats[metric]

def compute_grid(n):
    """
    Compute a near-square grid for n panels.

    Returns
    -------
    nrows, ncols : int, int

    Example:
        n=12 -> (3, 4), i.e. a visual 4x3 grid
    """
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols



def create_h5_files(camels_root: PosixPath,
                    out_file: PosixPath,
                    basins: List,
                    dates: List,
                    with_basin_str: bool = True,
                    seq_length: int = 365):
    """[summary]
    
    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    out_file : PosixPath
        Path of the location, where the hdf5 file should be stored
    basins : List
        List containing the 8-digit USGS gauge id
    dates : List
        List of start and end date of the discharge period to use, when combining the data.
    with_basin_str : bool, optional
        If True, stores for each sample the corresponding USGS gauged id, by default True
    seq_length : int, optional
        Length of the requested input sequences., by default 365
    
    Raises
    ------
    FileExistsError
        If file at this location already exists.
    """
    if out_file.is_file():
        warnings.warn(f"File already exists at {out_file}")
    else:
        with h5py.File(out_file, 'w') as out_f:
            input_data = out_f.create_dataset(
                'input_data',
                shape=(0, seq_length, 15),
                maxshape=(None, seq_length, 15),
                chunks=True,
                dtype=np.float32,
                compression='gzip')
            target_data = out_f.create_dataset(
                'target_data',
                shape=(0, 1),
                maxshape=(None,1),
                chunks=True,
                dtype=np.float32,
                compression='gzip')
            q_stds = out_f.create_dataset(
                'q_stds',
                shape=(0, 1),
                maxshape=(None, 1),
                dtype=np.float32,
                compression='gzip',
                chunks=True)

            if with_basin_str:
                sample_2_basin = out_f.create_dataset(
                    'sample_2_basin',
                    shape=(0,),
                    maxshape=(None,),
                    dtype="S10",
                    compression='gzip',
                    chunks=True)
            for basin in tqdm(basins, file=sys.stdout):
                dataset = CamelsTXT(
                    camels_root=camels_root,
                    basin=basin,
                    is_train=True,
                    seq_length=seq_length,
                    dates=dates)
                
                num_samples = len(dataset)
                total_samples = input_data.shape[0] + num_samples

                # store input and output samples
                if num_samples > 0:
                    input_data.resize((total_samples, seq_length, 15))
                    target_data.resize((total_samples,1))
                    input_data[-num_samples:, :, :] = dataset.x
                    target_data[-num_samples:, :] = dataset.y

                    # additionally store std of discharge of this basin for each sample
                    q_stds.resize((total_samples, 1))
                    q_std_array = np.array([dataset.q_std] * num_samples, dtype=np.float32).reshape(-1, 1)
                    q_stds[-num_samples:, :] = q_std_array

                    if with_basin_str:
                        sample_2_basin.resize((total_samples,))
                        str_arr = np.array([basin.encode("ascii", "ignore")] * num_samples)
                        sample_2_basin[-num_samples:] = str_arr

                out_f.flush()


def get_basin_list() -> List:
    """Read list of basins from text file.
    
    Returns
    -------
    List
        List containing the 8-digit basin code of all basins
    """
    basin_file = Path(__file__).absolute().parent.parent / f"data/basin_list.txt" 
    with basin_file.open('r') as fp:
        basins = fp.readlines()
    basins = [basin.strip() for basin in basins]
    return basins


def ShannonEntropy(input: torch.Tensor,dim: int=-1, epsilon: float =1e-12, reduction: bool="mean"):
    """
    Compute the Shannon entropy along specified dimension. Reduce the other dimension by the mean.
    The functions assumes that the sum of inputis normalized along the specified dimension.
    Parameters:
    -----------
    input : torch.Tensor
        Tensor containing the attention scores
    dim : int
        Dimension along which to compute Sahnnon entropy
    epsilon : float
        Regularization constant to avoid NaN when computing the log
    reduction : bool
        How to reduce the output, if average or sum over the remaining dimension.
        If None, no reduction is applied
    """
    input = input + epsilon
    if reduction is None:
        return - torch.sum(input * torch.log(input), dim=dim)
    elif reduction == "mean":
        return torch.mean(- torch.sum(input * torch.log(input), dim=dim))
    elif reduction == "sum":
        return torch.sum(- torch.sum(input * torch.log(input), dim=dim))
    else:
        raise ValueError("Invalid reduction argument given.")
    
def NSELoss(y_pred: torch.Tensor, y_true: torch.Tensor, q_stds: torch.Tensor, eps: float = 0.1):
    """Calculate (batch-wise) NSE Loss.

    Each sample i is weighted by 1 / (std_i + eps)^2, where std_i is the standard deviation of the 
    discharge from the basin, to which the sample belongs.

    Parameters:
    -----------
    y_pred : torch.Tensor
        Tensor containing the network prediction.
    y_true : torch.Tensor
        Tensor containing the true discharge values
    q_stds : torch.Tensor
        Tensor containing the discharge std (calculate over training period) of each sample
    eps : float
        Constant, added to the weight for numerical stability and smoothing, default to 0.1
    Returns
    -------
    torch.Tensor
        The (batch-wise) NSE Loss
    """
    squared_error = (y_pred - y_true)**2
    weights = 1 / (q_stds + eps)**2
    scaled_loss = weights * squared_error 

    return torch.mean(scaled_loss)
    
class MetricsCallback(Callback):
    """
    PyTorch Lightning metric callback.
    Save logged metrics
    """

    def __init__(self, dirpath, filename):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.path = os.path.join(dirpath, filename)
        exists = os.path.exists(self.path)
        # if already exists a saving, load it and update
        if exists:
            self.dict_metrics = torch.load(self.path, map_location=torch.device('cpu'))
        else:
            os.makedirs(self.dirpath, exist_ok = True) 
            self.dict_metrics = {}
            
        
    def on_validation_epoch_end(self,trainer, pl_module):
        epoch_num = int(trainer.logged_metrics["epoch_num"].cpu().item())
        self.dict_metrics["Epoch: "+str(epoch_num)] = copy.deepcopy(trainer.logged_metrics)
        torch.save(self.dict_metrics, self.path)



class AdaptiveScheduler(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (dict): Dictionary of epoch indices/learning rates. Keys must be increasing.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.002    if 30 <= epoch < 80
        >>> # lr = 0.0004   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones={30 : 0.002 , 80 :0.0004})
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, milestones, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self.milestones = milestones
        super().__init__(optimizer, last_epoch, verbose)

 
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        lrs = [] #[self.optimizer.param_groups[0]['lr']] # fixed learning rate of first group (encoder)
        for group in self.optimizer.param_groups: # run over all other groups
            if self.last_epoch not in self.milestones.keys():
                lrs.append(group['lr']) # do nothing
            else:
                lrs.append(self.milestones[self.last_epoch]) # set lr as scheduler 

        return lrs


def find_best_epoch(dirpath):
    """
    Find the epoch at which the validation error is minimized, or quivalently
    when thevalidation NSE is maximized
    Returns
    -------
        best_epoch : (int)
    """
    path_metrics = os.path.join(dirpath, "metrics.pt")
    data = torch.load(path_metrics, map_location=torch.device('cpu'))
    epochs_mod = []
    nse_mod = []
    for key in data:
        epoch_num = data[key]["epoch_num"]
        nse = -data[key]["val_loss"]
        if isinstance(epoch_num, int):
            epochs_mod.append(epoch_num)
            nse_mod.append(nse)
        else:
            epochs_mod.append(int(epoch_num.item()))
            nse_mod.append(nse.item())
    idx_ae = np.argmax(nse_mod)
    return int(epochs_mod[idx_ae])


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def clean_and_capitalize(input_string):
    # Split the input string into words
    words = input_string.split("_")
    out = ""

    for w in words:
        if w=="gages2":
             w="Catchment"
        if w=="freq":
             w="Frequency"
        if w=="dur":
             w="Duration"
        if w=="elev":
             w="Elevation"
        if w=="prec" or w=="p":
            w="Prec"
        if w=="frac":
            w="Fraction"
        if w=="geol":
            w="Geological"
        if w=="carbonate":
            w="Carb."
        if w=="permeability":
            w="Perm."
        if w=="porostiy":
            w="Porosity"
        w = w.capitalize()
        if w=="Nse":
             w = "NSE"
        if w=="Statsgo":
            w="(STATSGO)"
        if w=="Pelletier":
             w="(Pelletier)"
        if w=="Pet":
            w="PET"
        if w=="Elas":
            w="ELAS"
        if w=="Fdc":
             w="FDC"
        if w=="Gvf":
             w="GVF"
        if w=="Aridity":
             w="Aridity Index"
        if w=="Hfd":
             w="HFD"
        if w=="Lai":
             w="LAI"
        
        out += w
        out += " "

    return out


### Varitaional Autoencoder Functions
def sample(mean, logvar):
    """
    Sample according to a standard Gaussian with mean = 0 and std = 1 and
    implement reparametrization trick.
    --------
    Returns:
    Torch tensor 
    """
    randn = torch.randn_like(logvar)
    return mean + randn * torch.exp(0.5*logvar)


class nKLDivLoss(nn.Module):
    """
    Compute Kulback-Libler divergence loss for a gaussian variational autoencoder
    """
    def __init__(self,):
        super(nKLDivLoss, self).__init__()
        
    def forward(self,  mean, logvar):
        return torch.mean(0.5*torch.sum(torch.exp(logvar) + mean**2 - 1 - logvar, dim=1), dim = 0)


def give_report_file(encoded_features, leftout , experiment, run):
    if leftout != "none":
        fname = f"reports/{experiment}_es{encoded_features}_leftout{leftout}_run{run}.out"
    else:
        fname = f"reports/{experiment}_es{encoded_features}_run{run}.out"

    return fname

def give_result_file(encoded_features, leftout, eval_period, experiment):
    if leftout != "none":
        fname = f"analysis/results_data/{experiment}_es{encoded_features}_leftout{leftout}_{eval_period}.pkl"
    else:
        fname = f"analysis/results_data/{experiment}_es{encoded_features}_{eval_period}.pkl"

    return fname

def give_stat_file(encoded_features, leftout, eval_period, experiment):
    if leftout != "none":
        fname = f"analysis/stats/{eval_period}/{experiment}_es{encoded_features}_leftout{leftout}.csv"
    else:
        fname = f"analysis/stats/{eval_period}/{experiment}_es{encoded_features}.csv"

    return fname



attribute_draw_style = {
    # soil features
    'silt_frac': {
        'color': '#ffffe5',
        'marker': 'v'
    },
    'soil_depth_pelletier': {
        'color': '#fff7bc',
        'marker': 'v'
    },
    'clay_frac': {
        'color': '#fee391',
        'marker': 'v'
    },
    'soil_conductivity': {
        'color': '#fec44f',
        'marker': 'v'
    },
    'max_water_content': {
        'color': '#fe9929',
        'marker': 'v'
    },
    'geol_permeability': {
        'color': '#ec7014',
        'marker': 'v'
    },
    'soil_porosity': {
        'color': '#cc4c02',
        'marker': 'v'
    },
    'sand_frac': {
        'color': '#993404',
        'marker': 'v'
    },
    'soil_depth_statsgo': {
        'color': '#662506',
        'marker': 'v'
    },
    'carbonate_rocks_frac': {
        'color': '#000000',
        'marker': 'v'
    },
    # climate indices
    'low_prec_dur': {
        'color': '#ece7f2',
        'marker': 'o'
    },
    'aridity': {
        'color': '#d0d1e6',
        'marker': 'o'
    },
    'pet_mean': {
        'color': '#a6bddb',
        'marker': 'o'
    },
    'frac_snow': {
        'color': '#74a9cf',
        'marker': 'o'
    },
    'low_prec_freq': {
        'color': '#3690c0',
        'marker': 'o'
    },
    'p_mean': {
        'color': '#0570b0',
        'marker': 'o'
    },
    'high_prec_dur': {
        'color': '#045a8d',
        'marker': 'o'
    },
    'high_prec_freq': {
        'color': '#023858',
        'marker': 'o'
    },
    # vegetation properties
    'gvf_max': {
        'color': '#d9f0a3',
        'marker': '*'
    },
    'frac_forest': {
        'color': '#addd8e',
        'marker': '*'
    },
    'lai_max': {
        'color': '#78c679',
        'marker': '*'
    },
    'lai_diff': {
        'color': '#41ab5d',
        'marker': '*'
    },
    'gvf_diff': {
        'color': '#238443',
        'marker': '*'
    },
    # general
    'slope_mean': {
        'color': '#fcc5c0',
        'marker': 's'
    },
    'elev_mean': {
        'color': '#f768a1',
        'marker': 's'
    },
    'area_gages2': {
        'color': '#7a0177',
        'marker': 's'
    },
}