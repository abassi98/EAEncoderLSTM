

import argparse
import pickle
from pathlib import Path
import pandas as pd
import glob
import gc
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("encoded_features")
    parser.add_argument("eval_period", choices=["val", "test"])
    parser.add_argument("nruns", type=int)
    parser.add_argument("init_run", type=int)
    parser.add_argument("--random_features", action="store_true")
    return parser.parse_args()

# number of ensemble members
args = get_args()
experiment = args.experiment
encoded_features = args.encoded_features
eval_period = args.eval_period
nruns = args.nruns
init_run = args.init_run
output_suffix = "_random_features" if args.random_features else ""

ens_dict = {}
for run in range(init_run, init_run + nruns):
    fname = f"reports/global_{experiment}_es{encoded_features}_run{run}.out"
    #print("size report file: ", os.path.getsize(fname))
    with open(fname, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("seed"):
            seed = line.split('seed: ')[1].strip("\n")
        if line.startswith("run_dir"):
            run_dir = line.split('Attention4Hydro/')[1].strip("\n")
            break 
    
    # grab the test output file for this split
    #results_file = glob.glob(str(Path(run_dir) / f"lstm*.p"))[0]
    results_file = glob.glob(str(Path(run_dir) / f"lstm*{eval_period}{output_suffix}.p"))[0]
    print("result_file", results_file)
    with open(results_file, 'rb') as g:
        run_dict = pickle.load(g)
    
   
    if run == init_run: 
        ens_dict = run_dict
    else:
        for basin in run_dict:
            merged_attrs = {**ens_dict[basin].attrs, **run_dict[basin].attrs}
            ens_dict[basin] = pd.merge(
                ens_dict[basin],
                run_dict[basin][f"qsim_{seed}"],
                left_index=True,
                right_index=True,
                how="inner")
            ens_dict[basin].attrs = merged_attrs
           
    del run_dict
    # Force garbage collection
    gc.collect()

# calculate ensemble mean
for basin in ens_dict:
    #print(ens_dict[basin].keys())
    simdf = ens_dict[basin].filter(regex='qsim_')
    ensMean = simdf.mean(axis=1)
    ens_dict[basin].insert(0, 'qsim', ensMean)


# save the ensemble results as a pickle
# The report tag "ae" marks the encoder models, saved under the bare family name
# (global_es*); other tags such as "linear" are kept (global_linear_es*).
model_name = "global" if experiment == "ae" else f"global_{experiment}"
fname = f"analysis/results_data/{model_name}_es{encoded_features}_{eval_period}{output_suffix}.pkl"


with open(fname, 'wb') as f:
    pickle.dump(ens_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
