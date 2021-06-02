#!/bin/bash

#SBATCH --job-name vi
#SBATCH --output outputs/%x.out

#SBATCH --partition short
#SBATCH --time 24:00:00
#SBATCH --cpus-per-task 8
#SBATCH --mem-per-cpu 32G

#SBATCH --mail-type FAIL,TIME_LIMIT
#SBATCH --mail-user baisero.a@northeastern.edu

pomdp=$1
shift
model=$1
shift

bash vi.bash $pomdp $model $@
