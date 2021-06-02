#!/bin/bash

pomdp=$1
shift
model=$1
shift

mkdir -p vfs/ alphas/ logs/

core_filename=cores/$pomdp.$model.core
vf_filename=vfs/$pomdp.$model.vf
alpha_filename=alphas/$pomdp.$model.alpha
log_filename=logs/vi.$pomdp.$model.log

cmd_options=()
if [ $model != bsr ]; then
  cmd_options+=(--load-core $core_filename)
fi
cmd_options+=(--load-vf $vf_filename)
cmd_options+=(--save-vf $vf_filename)
cmd_options+=(--save-alpha $alpha_filename)
cmd_options+=(--log-filename $log_filename --log-level DEBUG)
cmd_options+=(--disable-pbar)

rl-psr-vi.py pomdps/$pomdp $model ${cmd_options[@]} $@
