#!/bin/bash
conda activate ea-encoder-lstm

experiment=$1
encoded_features=$2 # encoded space dimension
eval_period=$3 # val or test
nruns=$4
init_run=$5
random_features_flag=${6:-}

evaluate_args=()
merge_args=()
if [[ -n "${random_features_flag}" ]]; then
    if [[ "${random_features_flag}" != "--random_features" ]]; then
        echo "Optional sixth argument must be --random_features"
        exit 1
    fi
    evaluate_args+=("--random_features")
    merge_args+=("--random_features")
fi


## run analyisis for different runs in parallel 
for (( run = $init_run ; run < $nruns + $init_run; run++ )); do
    # Get the correct run directory by reading the screen report
    fname="reports/global_${experiment}_es${encoded_features}_run${run}.out"
    echo "Working on run: ${run} -- file: ${fname}"
    echo "Experiment: global"
    echo "Encoded space dimension: ${encoded_features}"
    run_dir=$(grep -m 1 '^run_dir' "$fname" | awk -F'Attention4Hydro/' '{print $2}' | tr -d '\n')
    # Let Slurm bind this evaluation to CPUs available in the job allocation.
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=4 \
        python main.py evaluate --eval_period="${eval_period}" --num_workers=4 --run_dir="${run_dir}" "${evaluate_args[@]}" &
done
wait
python -m scripts.merge_analysis_global $experiment $encoded_features $eval_period $nruns $init_run "${merge_args[@]}"

