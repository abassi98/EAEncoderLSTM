

import pickle
import sys
from pathlib import Path
import glob
import numpy as np

# number of ensemble members
encoded_features = int(sys.argv[1])
init_run = int(sys.argv[2])
nruns = int(sys.argv[3])
nsplits = int(sys.argv[4])
eval_period = str(sys.argv[5])


ens_dicts = []
for run in range(init_run, init_run + nruns):
    run_dict = {}
    for split in range(nsplits):
        fname = f"reports/pub_ae_es{encoded_features}_run{run}_split{split}.out"
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
        grads_file = glob.glob(str(Path(run_dir) / f"grads_{seed}_{eval_period}.p"))[0]
        print("grads_file", grads_file)
        with open(grads_file, 'rb') as g:
            split_dict = pickle.load(g)

        run_dict = merge(run_dict, split_dict)

    ens_dicts.append(run_dict)

# aggregate results
results = {}

for basin in ens_dicts[0].keys():
    grad  = np.zeros((nruns, 26))
    for run in range(nruns):
        grad[run,:] = ens_dicts[run][basin]
        #print(f"basin: {basin} -- run: {run} -- grad shape: {ens_dicts[run][basin].shape}")
    results[basin] = grad


with open(f"analysis/gradients/pub_es{encoded_features}_{eval_period}.pkl", 'wb') as f:
    pickle.dump(results, f) 





