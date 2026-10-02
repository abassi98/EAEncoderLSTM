

import argparse
import json
import pickle
import random
import glob
from collections import defaultdict
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm
#from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb
from pytorch_lightning.loggers import WandbLogger
#import optuna
from functools import partial
import optuna
from src.datasets import CamelsTXT, CamelsH5
from src.datautils import add_camels_attributes, rescale_features, normalize_features, TOPO_ATTRS, GEOL_ATTRS, VEGE_ATTRS, SOIL_ATTRS, OTHER, get_variable_name
#from papercode.models import Hydro_LSTM
from src.utils import get_basin_list, str2bool,create_h5_files, NSELoss
from src.models import Hydro_Attention


###########
# Globals #
###########

def NSE(y_obs, y_pred):
    num = torch.mean(torch.square(y_pred - y_obs))
    den = torch.mean(torch.square(y_obs - torch.mean(y_obs)))
    return 1.0 - num/den


# fixed settings for all experiments
GLOBAL_SETTINGS = {
    'batch_size': 256,
    'clip_norm': True,
    'clip_value': 1,
    'hidd_layers' : 4*[300],
    'dropout': 0.4,
    'epochs': 30,
    'hidden_size': 256,
    'initial_forget_gate_bias': 3,
    'log_interval': 50,
    'learning_rate': 1e-3,
    'seq_length': 365,
    'train_start': pd.to_datetime('01101999', format='%d%m%Y'),
    'train_end': pd.to_datetime('30092008', format='%d%m%Y'),
    'val_start': pd.to_datetime('01101981', format='%d%m%Y'),
    'val_end': pd.to_datetime('30091989', format='%d%m%Y'),
    'test_start': pd.to_datetime('01101989', format='%d%m%Y'),
    'test_end': pd.to_datetime('30091999', format='%d%m%Y'),
}

###############
# Prepare run #
###############

def int_or_none(value):
    if value == "None":
        return None
    return int(value)

def get_args() -> Dict:
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "evaluate", "gradients", "optimize", "create_datasets", "save_create_splits", "visualize_study"])
    parser.add_argument('--name', type=str, default="benchmark", help="Name of the experiment.")

    parser.add_argument(
        '--camels_root',
        type=str,
        default='../Datasets/CAMELS-US/',
        help="Root directory of CAMELS data set")
    
    parser.add_argument(
        '--db_path',
        type=str,
        default='data/attributes.db',
        help="Path for  basin attributes.")
    
    parser.add_argument(
        '--train_file',
        type=str,
        default='data/train_data.h5',
        help="Path to train .h5 file.")
    
    parser.add_argument(
        '--val_file',
        type=str,
        default='data/val_data.h5',
        help="Path to validation .h5 file.")
    
    parser.add_argument('--seed', type=int, required=False, help="Random seed")

    parser.add_argument('--run_dir', type=str, default=None, help="Path to run directory.")

    parser.add_argument(
        '--gpu',
        type=int,
        default=-1,
        help="User-selected GPU ID - if none chosen, will default to cpu")
    
    parser.add_argument(
        '--num_workers', type=int, default=12, help="Number of parallel threads for data loading")
    
    
    parser.add_argument(
        '--encoded_space_dim',
        type=int_or_none,
        default=None,
        help="If put_attention is False, it is the output dimension of the encoder."
    )

    parser.add_argument(
        "--eval_period",
        type=str,
        default="test",
        choices=["val", "test"]
    )   
    parser.add_argument(
        "--random_features",
        action="store_true",
        help=(
            "During evaluation, replace encoder outputs fed to the LSTM decoder "
            "with one basin-constant standard-normal random feature vector."
        ),
    )

    parser.add_argument(
        '--basin_file',
        type=str,
        default=None,
        help="Path to file containing usgs basin ids. Default is data/basin_list.txt")
    
    parser.add_argument(
        '--n_splits',
        type=int,
        default=None,
        help="Number of splits to create for cross validation")
    
    parser.add_argument(
        '--split',
        type=int,
        default=None,
        help="Defines split to use for training/testing in kFold cross validation")
    
    parser.add_argument(
        '--split_file',
        type=str,
        default=None,
        help="Path to file created from the `create_splits` function.")
    
    cfg = vars(parser.parse_args())

    # Validation checks
    if (cfg["mode"] in ["train", "create_datasets", "create_splits"]) and cfg["seed"] is None:
        # generate random seed for this run
        cfg["seed"] = int(np.random.uniform(low=0, high=1e6))

    if (cfg["mode"] in ["evaluate"]) and (cfg["run_dir"] is None):
        raise ValueError("In evaluation mode a run directory (--run_dir) has to be specified")
    if cfg["random_features"] and cfg["mode"] != "evaluate":
        raise ValueError("--random_features is only supported in evaluate mode.")

   
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # combine global settings with user config
    cfg.update(GLOBAL_SETTINGS)

    # convert path to PosixPath object
    if cfg["camels_root"] is not None:
        cfg["camels_root"] = Path(cfg["camels_root"])
 
    return cfg


def _setup_run(cfg: Dict) -> Dict:
    """Create folder structure for this run

    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config

    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
  
    if cfg["run_dir"] is None:
        # initialise wandb runs
        wandb.init(name=cfg["name"],  project="Attention4Hydro", dir=f"runs/")
        cfg['run_dir'] = Path(__file__).absolute().parent / wandb.run.dir
    else:
        cfg['run_dir'] = Path(cfg['run_dir'])
       
    # dump a copy of cfg to run directory
    with (cfg["run_dir"] / 'cfg.json').open('w') as fp:
        temp_cfg = {}
        for key, val in cfg.items():
            if isinstance(val, PosixPath):
                temp_cfg[key] = str(val)
            elif isinstance(val, pd.Timestamp):
                temp_cfg[key] = val.strftime(format="%d%m%Y")
            else:
                temp_cfg[key] = val
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)

    if cfg["mode"] == "train":
        # print config to terminal
        for key, val in cfg.items():
            print(f"{key}: {val}")

    return cfg


def _prepare_data(cfg: Dict, train_basins: List, val_basins: List = None) -> Dict:
    """Preprocess training data.

    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config
    train_basins : List
        List containing the 8-digit USGS gauge id used for training
    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    # create database file containing the static basin attributes
    #add_camels_attributes(cfg["camels_root"], db_path="data/")

    if cfg["split_file"] is None:
        train_file = Path(cfg["train_file"])
        val_file = Path(cfg["val_file"])
    else:
        seed_split = cfg["split_file"].split(".p")[0].split("seed")[1]
        train_file = Path(f"data/train_data_seed{seed_split}_split{cfg['split']}.h5")
        val_file =  Path(f"data/val_data_seed{seed_split}_split{cfg['split']}.h5")
    
    # create .h5 files for train data
    create_h5_files(
        camels_root=cfg["camels_root"],
        out_file=train_file,
        basins=train_basins,
        dates=[cfg["train_start"], cfg["train_end"]],
        with_basin_str=True,
        seq_length=cfg["seq_length"])
    
    if val_basins is not None:
        # create .h5 files for train data
        create_h5_files(
            camels_root=cfg["camels_root"],
            out_file=val_file,
            basins=val_basins,
            dates=[cfg["val_start"], cfg["val_end"]],
            with_basin_str=True,
            seq_length=cfg["seq_length"])
    
    return cfg


###########################
# Train or evaluate model #
###########################

def train(cfg):
    """Train model.

    Parameters
    ----------
    cfg : Dict
        Dictionary containing the run config
    """
    # fix random seeds
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    if cfg["split_file"] is not None:
        with Path(cfg["split_file"]).open('rb') as fp:
            splits = pickle.load(fp)
        basins = splits[cfg["split"]]["train"]
        val_basins = splits[cfg["split"]]["test"]
    else:
        basins = get_basin_list()
        val_basins = get_basin_list()

    # create folder structure for this run"
    cfg = _setup_run(cfg)
    cfg = _prepare_data(cfg, basins, val_basins=val_basins) 
    
    input_dim = 26
   
    # prepare PyTorch DataLoader
    if cfg["split_file"] is None:
        train_file = Path(cfg["train_file"])
        val_file = Path(cfg["val_file"])
    else:
        seed_split = cfg["split_file"].split(".p")[0].split("seed")[1]
        train_file = Path(f"data/train_data_seed{seed_split}_split{cfg['split']}.h5")
        val_file =  Path(f"data/val_data_seed{seed_split}_split{cfg['split']}.h5")
        
    ds = CamelsH5(
        h5_file=train_file,
        basins=basins,
        db_path=cfg["db_path"])

    ds_val = CamelsH5(
            h5_file=val_file,
            basins=val_basins,
            db_path=cfg["db_path"])
    
    print("Attributes normalized", ds.df)
    
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    loader_val = DataLoader(ds_val, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    ### Pytorch Lightning model
    milestones = {20: 5e-4, 25: 1e-4}
    model =  Hydro_Attention(
                 input_dim = input_dim,
                 output_dim= cfg["encoded_space_dim"],
                 hidden_layers= cfg["hidd_layers"],
                 lstm_hidden_units =  cfg["hidden_size"], 
                 initial_forget_bias=cfg["initial_forget_gate_bias"],
                 act = nn.LeakyReLU(), 
                 drop_p = cfg["dropout"], 
                 seq_length = cfg["seq_length"],
                 lr = cfg["learning_rate"],
                 milestones = milestones,
                 )
    
    
    # print model
    print(model)
    logger = WandbLogger()

    checkpoint_model = ModelCheckpoint(
            save_top_k=1,
            save_last=True,
            monitor="val_loss",
            mode="min",
            dirpath= cfg["run_dir"],
            filename= "model-{epoch:02d}",
        )
    
    # define trainer 
    torch.set_float32_matmul_precision('medium')
    
    if cfg["clip_norm"]:
        trainer = pl.Trainer(max_epochs=cfg["epochs"], callbacks=[checkpoint_model], accelerator=str(DEVICE), devices=1, logger=logger, gradient_clip_val=cfg["clip_value"])
    else:
        trainer = pl.Trainer(max_epochs=cfg["epochs"], callbacks=[checkpoint_model], accelerator=str(DEVICE), devices=1, logger=logger)


    torch.cuda.empty_cache()
    path_last = cfg["run_dir"] / "last.ckpt"
    if path_last.exists():
        print("Resuming training from checkpoint")
        trainer.fit(model=model, train_dataloaders=loader, val_dataloaders=loader_val, ckpt_path=path_last)
    else:
        print("No checkpoint found, training from scratch")
        trainer.fit(model=model, train_dataloaders=loader, val_dataloaders=loader_val)
   

def evaluate(user_cfg: Dict):
    """Train model for a single epoch.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
        
    """
    with open(Path(user_cfg["run_dir"]) / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)

    if user_cfg["random_features"] and not run_cfg["encoded_space_dim"]:
        raise ValueError("--random_features requires a model with encoded_space_dim > 0.")
    if user_cfg["random_features"]:
        evaluation_seed = user_cfg["seed"] if user_cfg["seed"] is not None else run_cfg["seed"]
        torch.manual_seed(evaluation_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(evaluation_seed)

    # get basins
    if user_cfg["split_file"] is not None:
        with Path(user_cfg["split_file"]).open('rb') as fp:
            splits = pickle.load(fp)
        basins = splits[run_cfg["split"]]["test"]
    else:
        basins = get_basin_list()

    input_dim = 26 

  
    # get attribute means/stds from trainings dataset
    train_file = user_cfg["train_file"] 
    db_path = user_cfg["db_path"] 
    ds_train = CamelsH5(
        h5_file=train_file, db_path=db_path, basins=basins)
    means = ds_train.get_attribute_means()
    stds = ds_train.get_attribute_stds()
    
    
    #print("Mean/std/lenght", means, stds, len(means))
    ### Pytorch Lightning model
    model =  Hydro_Attention(
                 input_dim = input_dim,
                 output_dim=run_cfg["encoded_space_dim"],
                 hidden_layers= run_cfg["hidd_layers"],
                 lstm_hidden_units =  run_cfg["hidden_size"], 
                 initial_forget_bias=run_cfg["initial_forget_gate_bias"],
                 act = nn.LeakyReLU(),
                 drop_p = run_cfg["dropout"], 
                 seq_length = run_cfg["seq_length"],
                 lr = run_cfg["learning_rate"],
                 )
  
    weight_file =  glob.glob(f"{user_cfg['run_dir']}/last.ckpt")[0]
    print("Analaysing model at ", weight_file)
    p = torch.load(weight_file, map_location=torch.device("cpu"))
    model.load_state_dict(p["state_dict"])

    # print model
    print(model)

    # set data range
    date_range = pd.date_range(start=GLOBAL_SETTINGS[f"{user_cfg['eval_period']}_start"], end=GLOBAL_SETTINGS[f"{user_cfg['eval_period']}_end"])
    results = {}
    results_hidden = {}
    results_cell = {}
    #print("Data range:", date_range)

    for basin in tqdm(basins):
        ds_test = CamelsTXT(
            camels_root=user_cfg["camels_root"],
            basin=basin,
            dates=[GLOBAL_SETTINGS[f"{user_cfg['eval_period']}_start"], GLOBAL_SETTINGS[f"{user_cfg['eval_period']}_end"]],
            is_train=False,
            seq_length=run_cfg["seq_length"],
            with_attributes=True,
            attribute_means=means,
            attribute_stds=stds,
            db_path=db_path
            )

        
        loader = DataLoader(ds_test, batch_size=1024, shuffle=False, num_workers=user_cfg["num_workers"])

        basin_random_features = None
        if user_cfg["random_features"]:
            basin_random_features = torch.randn(
                run_cfg["encoded_space_dim"], device=DEVICE
            )

        preds, obs, enc, hidden, cell = evaluate_basin(
            model, loader, random_features=basin_random_features
        )
        df = pd.DataFrame(data={'qobs': obs.flatten(), f'qsim_{run_cfg["seed"]}': preds.flatten()}, index=date_range)
        df.attrs[f"enc_{run_cfg['seed']}"] = enc
        
        results[basin] = df
        results_hidden[basin] = pd.DataFrame(data=hidden, index=date_range)
        results_cell[basin] = pd.DataFrame(data=cell, index=date_range)

    # save predictions
    _store_results(user_cfg, run_cfg, results)

    # save hidden and cell states
    output_suffix = "_random_features" if user_cfg["random_features"] else ""
    file_name = Path(user_cfg["run_dir"]) / f"hidden_{run_cfg['seed']}_{user_cfg['eval_period']}{output_suffix}.p"
    with (file_name).open('wb') as fp:
        pickle.dump(results_hidden, fp)

    file_name = Path(user_cfg["run_dir"]) / f"cell_{run_cfg['seed']}_{user_cfg['eval_period']}{output_suffix}.p"
    with (file_name).open('wb') as fp:
        pickle.dump(results_cell, fp)

def evaluate_basin(
    model: nn.Module, loader: DataLoader, random_features: torch.Tensor | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate model on a single basin

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    loader : DataLoader
        PyTorch DataLoader containing the basin data in batches.

    Returns
    -------
    preds : np.ndarray
        Array containing the (rescaled) network prediction for the entire data period
    random_features : torch.Tensor, optional
        One encoded feature vector for the basin, reused for every evaluation
        window. If omitted, use the model's learned encoder.

    obs : np.ndarray
        Array containing the observed discharge for the entire data period

    """
    model.eval()
    preds, obs, enc, hidden, cell = None, None, None, None, None
    

    # compute predictions
    with torch.no_grad():
        for data in loader:
            x, y, attr = data
            x, y, attr = x.to(DEVICE), y.to(DEVICE), attr.to(DEVICE)
            enc, p, r = model(x, attr.squeeze(), random_features=random_features)
            h, c = r
            #print(enc)

            if preds is None:
                preds = p.cpu()
                obs = y.detach().cpu()
                hidden = h.detach().cpu()
                cell = c.detach().cpu()
            else:
                preds = torch.cat((preds, p.cpu()), 0)
                obs = torch.cat((obs, y.detach().cpu()), 0)
                hidden = torch.cat((hidden, h.cpu()), 1)
                cell = torch.cat((cell, c.detach().cpu()), 1)
             
    
    preds = rescale_features(preds.detach().numpy(), variable='output')
    if enc is not None:
        enc = enc.detach().cpu().numpy()[-1,:]

    obs = obs.numpy()
    hidden = hidden.detach().squeeze().numpy()
    cell = cell.detach().squeeze().numpy()
    # set discharges < 0 to zero
    preds[preds < 0] = 0

    return preds, obs, enc, hidden, cell

    
def _store_results(user_cfg: Dict, run_cfg: Dict, results: pd.DataFrame):
    """Store results in a pickle file.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
    run_cfg : Dict
        Dictionary containing the run config loaded from the cfg.json file
    results : pd.DataFrame
        DataFrame containing the observed and predicted discharge.

    """
    # save time series
    output_suffix = "_random_features" if user_cfg["random_features"] else ""
    file_name = Path(user_cfg["run_dir"]) / f"lstm_{run_cfg['seed']}_{user_cfg['eval_period']}{output_suffix}.p"
    with (file_name).open('wb') as fp:
        pickle.dump(results, fp)

    print(f"Sucessfully store results at {file_name}")

####################
# Cross Validation #
####################


def create_splits(cfg: dict):
    """Create random k-Fold cross validation splits.
    
    Takes a set of basins and randomly creates n splits. The result is stored into a dictionary,
    that contains for each split one key that contains a `train` and a `test` key, which contain
    the list of train and test basins.

    Parameters
    ----------
    cfg : dict
        Dictionary containing the user entered evaluation config
    
    Raises
    ------
    RuntimeError
        If file for the same random seed already exists.
    FileNotFoundError
        If the user defined basin list path does not exist.
    """
    
    # set random seed for reproduceability
    np.random.seed(cfg["seed"])

    # read in basin file
    if cfg["basin_file"] is not None:
        if not Path(cfg["basin_file"]).is_file():
            raise FileNotFoundError(f"Not file found at {cfg['basin_file']}")
        with open(cfg["basin_file"], 'r') as fp:
            basins = fp.readlines()
        basins = [b.strip() for b in basins]
        """
        Delete some basins because of missing data:
        - '06775500' & '06846500' no attributes
        - '09535100' no streamflow records
        """
        ignore_basins = ['06775500', '06846500', '09535100']
        basins = [b for b in basins if b not in ignore_basins]
    else:
        basins = get_basin_list()

    # create folds
    kfold = KFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])
    kfold.get_n_splits(basins)

    # dict to store the results of all folds
    splits = defaultdict(dict)

    for split, (train_idx, test_idx) in enumerate(kfold.split(basins)):
        # further split train_idx into train/val idx into train and val set

        train_basins = [basins[i] for i in train_idx]
        test_basins = [basins[i] for i in test_idx]

        splits[split] = {'train': train_basins, 'test': test_basins}


    return splits 

def save_create_splits(cfg: dict):
    output_file = (Path(__file__).absolute().parent / f'data/kfold_splits_seed{cfg["seed"]}.p')
    # check if split file already already exists
    if output_file.is_file():
        raise RuntimeError(f"File '{output_file}' already exists.")

    splits = create_splits(cfg)
    with output_file.open('wb') as fp:
        pickle.dump(splits, fp)

    print(f"Stored dictionary with basin splits at {output_file}")


if __name__ == "__main__":
    config = get_args()
    globals()[config["mode"]](config)
