#!/bin/bash

#SBATCH --job-name=ae_forcing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=4
#SBATCH --time=02-00:00:00
#SBATCH --partition=earth-4
#SBATCH --constraint=rhel8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=100GB
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END,FAIL     
#SBATCH --output=gpu_global.out
#SBATCH --error=gpu_global.err

module load gcc/9.4.0-pe5.34 miniconda3/4.12.0 lsfm-init-miniconda/1.0.0	# comment to run on your machine
conda activate ea-encoder-lstm
wandb online

encoded_features=$1
init_run=$2
nruns=$3

# Optional final argument: --forcing_summary
forcing_summary_flag=${4:-}
extra_args=()
experiment=ae
name_suffix=""
if [[ -n "$forcing_summary_flag" ]]; then
  if [[ "$forcing_summary_flag" != "--forcing_summary" ]]; then
    echo "Optional final argument must be --forcing_summary"
    exit 1
  fi
  extra_args+=("--forcing_summary")
  experiment=ae_forcing
  name_suffix="-forcing"
fi

gpu=0
num_workers=12

for (( run = $init_run ; run < $nruns + $init_run; run++ )); do
  outfile="reports/global_${experiment}_es${encoded_features}_run${run}.out"
  if [[ -f $outfile ]]; then
    run_dir=$(grep -m1 '^run_dir:' "$outfile" | sed 's/^run_dir:[[:space:]]*//')
    echo "$run_dir"
    python main.py --num_workers=$num_workers --encoded_space_dim=$encoded_features --name="Global-Linear${encoded_features}${name_suffix}-run${run}" --gpu=$gpu "${extra_args[@]}" --run_dir=$run_dir train > $outfile &
  else
    echo "Output file $outfile does not exist. Train from scratch"
    python main.py --num_workers=$num_workers --encoded_space_dim=$encoded_features --name="Global-Linear${encoded_features}${name_suffix}-run${run}" --gpu=$gpu "${extra_args[@]}" train > $outfile &
  fi
done
wait    

    
  
