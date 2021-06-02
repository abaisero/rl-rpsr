#!/bin/bash

pomdp=$1
shift
model=$1
shift

mkdir -p cores/ logs/

core_filename=cores/$pomdp.$model.core
log_filename=logs/search.$pomdp.$model.log

cmd_options=()
cmd_options+=(--save-core $core_filename)
cmd_options+=(--log-filename $log_filename --log-level DEBUG)

rl-psr-search.py pomdps/$pomdp $model ${cmd_options[@]} $@
