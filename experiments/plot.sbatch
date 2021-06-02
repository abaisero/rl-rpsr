#!/bin/bash

#SBATCH --job-name plot
#SBATCH --output outputs/%x.out

#SBATCH --partition short
#SBATCH --time 24:00:00
#SBATCH --cpus-per-task 2
#SBATCH --mem-per-cpu 2G

#SBATCH --mail-type FAIL,TIME_LIMIT
#SBATCH --mail-user baisero.a@northeastern.edu

pomdp=$1
shift
model=$1
shift

mkdir -p plots/ logs/

core_filename=cores/$pomdp.$model.core
vf_filename=vfs/$pomdp.$model.vf
plot_filename=plots/plot.$pomdp.$model.png
log_filename=logs/plot.$pomdp.$model.log

cmd_options=()
if [ $model != bsr ]; then
  cmd_options+=(--load-core $core_filename)
fi
cmd_options+=(--load-vf $vf_filename)
cmd_options+=(--save-plot $plot_filename)
cmd_options+=(--log-filename $log_filename --log-level DEBUG)

rl-psr-vi-plot.py pomdps/$pomdp $model ${cmd_options[@]} $@
