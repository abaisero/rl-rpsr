#!/bin/bash

#SBATCH --job-name search
#SBATCH --output outputs/%x.out

#SBATCH --partition short
#SBATCH --time 24:00:00
#SBATCH --ntasks 3
#SBATCH --mem-per-cpu 15G

#SBATCH --mail-type FAIL,TIME_LIMIT
#SBATCH --mail-user baisero.a@northeastern.edu

pomdp=$1
shift

models=(psr rpsr)

for model in ${models[@]}; do
  srun --ntasks 1 bash search.bash $pomdp $model &
done

wait
