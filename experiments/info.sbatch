#!/bin/bash

#SBATCH --job-name info
#SBATCH --output outputs/%x.out

#SBATCH --partition short
#SBATCH --time 1:00:00
#SBATCH --ntasks 2
#SBATCH --mem-per-cpu 16G

#SBATCH --mail-type FAIL,TIME_LIMIT
#SBATCH --mail-user baisero.a@northeastern.edu

pomdp=$1
shift

models=(psr rpsr)

for model in ${models[@]}; do
  srun --ntasks 1 bash info.bash $pomdp $model &
done

wait
