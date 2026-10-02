#!/bin/bash
conda activate ea-encoder-lstm

encoded_features=$1
init_run=$2
nruns=$3
nsplits=$4
init_split=$5

gpu=0
num_workers=4

for (( run = $init_run ; run < $nruns + $init_run; run++ )); do
  seed=$(($run + 300))
  #python main.py --n_splits=$nsplits --seed=$seed create_splits 
  for ((split = $init_split ; split < $nsplits + $init_split ; split++ )); do  
    outfile="reports/pub_ae_es${encoded_features}_run${run}_split${split}.out"
    if [[ -f $outfile ]]; then
        run_dir=$(grep -m1 '^run_dir:' "$outfile" | sed 's/^run_dir:[[:space:]]*//')
        echo "$run_dir"
        python main.py --num_workers=$num_workers --encoded_space_dim=$encoded_features --name="Pub-Linear${encoded_features}-run${run}-split${split}" --gpu=$gpu --split=$split --split_file="data/kfold_splits_seed$seed.p" --run_dir=$run_dir train > $outfile & 
    else
        echo "Output file $outfile does not exist. Train from scratch"
        python main.py --num_workers=$num_workers --encoded_space_dim=$encoded_features --name="Pub-Linear${encoded_features}-run${run}-split${split}" --gpu=$gpu --split=$split --split_file="data/kfold_splits_seed$seed.p" train > $outfile & 
    fi
  done
done
wait

  