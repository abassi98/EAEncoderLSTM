import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from src.datautils import load_attributes
from pathlib import PosixPath
from sklearn.decomposition import PCA
import argparse
from src.utils import str2bool,  get_basin_list, clean_and_capitalize, compute_grid
from src.datautils import LANDSCAPE_ATTRS, CLIM_ATTRS, HYDRO_ATTRS, SOIL_ATTRS, VEGE_ATTRS, TOPO_ATTRS, GEOL_ATTRS
import pickle
from src.datasets import CamelsTXT, CamelsH5
from main import GLOBAL_SETTINGS
from sklearn.feature_selection import mutual_info_regression


if __name__ == '__main__':
    ### load static attrs
    basins =  get_basin_list()
    num_basins = len(basins)
    db_path = "data/attributes.db"
    train_file = PosixPath("data/train_data.h5")
    db_path = "data/attributes.db"
    ds_train = CamelsH5(
        h5_file=train_file, db_path=db_path, basins=basins)
    means = ds_train.get_attribute_means()
    stds = ds_train.get_attribute_stds()

    # computing forcing stats
    forcing_stats = {}
    for ii, basin in enumerate(basins):
        ds_test = CamelsTXT(
                camels_root=PosixPath("../Datasets/CAMELS-US/"),
                basin=basin,
                dates=[GLOBAL_SETTINGS['test_start'], GLOBAL_SETTINGS['test_end']],
                is_train=False,
                seq_length=GLOBAL_SETTINGS["seq_length"],
                with_attributes=True,
                attribute_means=means,
                attribute_stds=stds,
                db_path=db_path,
            )
        x = ds_test.x[:,-1,:]
        mean = x.mean(0)
        std = x.std(0)
        forcing_stats[basin] = np.concatenate((mean, std), axis=0)

    with open("analysis/encoded_features/forcing_summary.pkl", "wb") as f:
        pickle.dump(forcing_stats, f)