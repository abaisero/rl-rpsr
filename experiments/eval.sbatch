#!/bin/bash

#SBATCH --job-name eval
#SBATCH --output outputs/%x.out

#SBATCH --partition short
#SBATCH --time 24:00:00
#SBATCH --cpus-per-task 2
#SBATCH --mem-per-cpu 4G

#SBATCH --mail-type FAIL,TIME_LIMIT
#SBATCH --mail-user baisero.a@northeastern.edu

pomdp=$1
shift
model=$1
shift
policy=$1
shift

bash eval.bash $pomdp $model $policy $@
