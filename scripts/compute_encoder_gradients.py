import numpy as np
from src.plot_utils import load_us_states
from src.datautils import load_attributes
from pathlib import PosixPath
import argparse
from src.utils import get_basin_list
from src.datautils import  CLIM_ATTRS, SOIL_ATTRS, VEGE_ATTRS, TOPO_ATTRS, GEOL_ATTRS

from src.models import Hydro_Attention
import glob
import torch
from tqdm import tqdm
import pickle
import os
from sklearn.decomposition import FastICA
from pathlib import Path

def get_args():
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--encoded_features',
        type=int,
        default=2,
        help="Encoded space dimension"
    )


    parser.add_argument(
        "--experiment",
        type=str,
        default="global"
    )

  

    cfg = vars(parser.parse_args())
    return cfg


def update_grads(grads, features, basins, attrs, report_file):
    with open(report_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("seed"):
            seed = line.split('seed: ')[1].strip("\n")
        if line.startswith("run_dir"):
            run_dir = line.split('Attention4Hydro/')[1].strip("\n")
            break 

    model_file = glob.glob(str(Path(run_dir) / f"last.ckpt"))[0]
    num_attrs = attrs.shape[1]
    experiment = report_file.split("/")[1].split("_")[0]
    
    if experiment == "global":
        run = int(report_file.split("run")[1].split(".")[0])
    elif experiment == "pub":
        run = int(report_file.split("run")[1].split("_")[0])
    else:
        raise ValueError("Undefined experiment.")

    encoded_features = int(report_file.split("es")[1].split("_")[0])
    ### Pytorch Lightning model
    model =  Hydro_Attention(
        input_dim = num_attrs,
        output_dim=encoded_features,
        hidden_layers= 4*[300],
    )
    print(run_dir)

    ### load model
    p = torch.load(model_file, map_location=torch.device("cpu"), weights_only=False)
    model.load_state_dict(p["state_dict"])
    model.eval()

    num_basins = len(basins)
    pbar = tqdm(range(num_basins), total=num_basins, desc=f"Computing gradients {experiment}_es{encoded_features} seed={seed}")

    with torch.no_grad():
        enc = model.encoder(attrs)

    for ii in pbar:
        basin = basins[ii]
        features[basin][f"run{run}"] = enc[ii,:].detach().numpy()
        attrs_ii = attrs[ii,:].unsqueeze(0).requires_grad_()
        enc_ii = model.encoder(attrs_ii)
        grads_basin = np.zeros((encoded_features, num_attrs))
        for jj in range(encoded_features):
            model.zero_grad()
            grads_ii = torch.autograd.grad(
                    outputs=enc_ii[:,jj],          # scalar output
                    inputs=attrs_ii,              # full tensor
                    grad_outputs=torch.ones_like(enc_ii[:,jj]),  # needed since output is scala
                    retain_graph=True,
                    create_graph=False,
                )[0]
            
            grads_basin[jj, :] = grads_ii.detach().numpy()

        grads[basin][f"run{run}"] = grads_basin

if __name__ == '__main__':
    # Load encoded features
    cfg = get_args()
    encoded_features = cfg["encoded_features"]
    experiment = cfg["experiment"]
    nruns = 4
    nsplits = 12
    us_states = load_us_states()
    train_file = 'data/train_data.h5'
    db_path = "data/attributes.db"

    ### load static attrs
    all_basins =  get_basin_list()
    num_basins = len(all_basins)

    # load attributes
    camels_root = PosixPath("../Datasets/CAMELS-US/")
    #add_camels_attributes(camels_root)
    KEEP_ATTRS = SOIL_ATTRS + CLIM_ATTRS + VEGE_ATTRS + TOPO_ATTRS + GEOL_ATTRS 
    df_S = load_attributes("data/attributes.db", all_basins, keep_attributes=KEEP_ATTRS)
   
    # normalize attributes
    df_S = (df_S - df_S.mean())/df_S.std()
    attrs = torch.tensor(df_S.values, dtype=torch.float32, requires_grad=True)
    num_attrs = df_S.shape[1]

    grads = {basin : {} for basin in all_basins}
    features = {basin : {} for basin in all_basins}
    for run in range(nruns):
        if experiment == "global":
            basins = get_basin_list()  
            report_file = f"reports/{experiment}_ae_es{encoded_features}_run{run}.out"
            attrs = torch.tensor(df_S.loc[basins].values, dtype=torch.float32, requires_grad=False)
            update_grads(grads, features, basins, attrs, report_file)
        elif experiment == "pub":
            split_seed = 300 + run
            split_file = f"data/kfold_splits_seed{split_seed}.p"
            with Path(split_file).open('rb') as fp:
                splits = pickle.load(fp)
            for split in range(nsplits):
                report_file = f"reports/{experiment}_ae_es{encoded_features}_run{run}_split{split}.out"
                basins = splits[split]["test"]
                attrs = torch.tensor(df_S.loc[basins].values, dtype=torch.float32, requires_grad=False)
                update_grads(grads, features, basins, attrs, report_file)

    new_grads = {}
    new_features = {}
    all_attrs = torch.tensor(df_S.values, dtype=torch.float32, requires_grad=False)
    all_grads = np.zeros((num_basins, nruns, encoded_features, num_attrs))
    enc = np.zeros((nruns, num_basins, encoded_features))
    for ii, basin in enumerate(all_basins):
        for run in range(nruns):
            enc[run, ii, :] = features[basin][f"run{run}"]
            all_grads[ii, run, :, :] = grads[basin][f"run{run}"]

        new_grads[basin] = all_grads[ii, :, :, :]

    with open(f"analysis/gradients/encoder_{experiment}_es{encoded_features}_icaFalse.pkl", "wb") as f:
        pickle.dump(new_grads, f)


    
    # apply ICA
    for run in range(nruns):
        ica = FastICA(n_components=encoded_features, whiten=False)
        Z = enc[run, :, :]     
        S = ica.fit_transform(Z)
        W = ica.components_
        for ii, basin in enumerate(all_basins):
            all_grads[ii, run, :, :] = W @ all_grads[ii, run, :, :]
     
    ica_grads = {}
    for ii, basin in enumerate(all_basins):
        ica_grads[basin] = all_grads[ii, :, :, :]
    
    with open(f"analysis/gradients/encoder_{experiment}_es{encoded_features}_icaTrue.pkl", "wb") as f:
        pickle.dump(ica_grads, f)
