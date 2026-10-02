#!/bin/bash
conda activate ea-encoder-lstm

encoded_features=$1
init_run=$2
nruns=$3

gpu=0
num_workers=12

for (( run = $init_run ; run < $nruns + $init_run; run++ )); do
  outfile="reports/global_ae_es${encoded_features}_run${run}.out"
  if [[ -f $outfile ]]; then
    run_dir=$(grep -m1 '^run_dir:' "$outfile" | sed 's/^run_dir:[[:space:]]*//')
    echo "$run_dir"
    python main.py --num_workers=$num_workers --encoded_space_dim=$encoded_features --name="Global-Linear${encoded_features}-run${run}" --gpu=$gpu --run_dir=$run_dir train > $outfile &
  else
    echo "Output file $outfile does not exist. Train from scratch"
    python main.py --num_workers=$num_workers --encoded_space_dim=$encoded_features --name="Global-Linear${encoded_features}-run${run}" --gpu=$gpu train > $outfile &
  fi
done
wait    

    
  
