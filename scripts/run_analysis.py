

import pickle
import sys
from pathlib import Path
import pandas as pd
import glob
import gc
import os

# number of ensemble members
nruns = 2
ens_dict = {}
for run in range(nruns):
    fname = f"reports/global_{sys.argv[1]}_{sys.argv[2]}_run{run}.out"
    #print("size report file: ", os.path.getsize(fname))
    with open(fname, "r") as f:
        lines = f.readlines()

    print(f"Working on run: {run} -- file: {fname}")
    for line in lines:
        if line.startswith("seed"):
            seed = line.split('seed: ')[1].strip("\n")
        if line.startswith("run_dir"):
            run_dir = line.split('Attention4Hydro/')[1].strip("\n")
            break 
    
    # grab the test output file for this split
    results_file = glob.glob(str(Path(run_dir) / f"lstm*.p"))[0]
  
    with open(results_file, 'rb') as g:
        run_dict = pickle.load(g)
    
   
    if run == 0: 
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
            print(ens_dict[basin], ens_dict[basin].attrs)
            

            
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
fname = f"analysis/results_data/global_{sys.argv[1]}_{sys.argv[2]}.pkl"
with open(fname, 'wb') as f:
    pickle.dump(ens_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

