#!/bin/bash

pomdp=$1
shift
model=$1
shift

mkdir -p logs/ infos/

core_filename=cores/$pomdp.$model.core
log_filename=logs/info.$pomdp.$model.log
info_filename=infos/$pomdp.$model.txt

cmd_options=()
cmd_options+=(--load-core $core_filename)
cmd_options+=(--stats --decimals 4)
cmd_options+=(--log-filename $log_filename --log-level DEBUG)

rl-psr-info.py pomdps/$pomdp $model ${cmd_options[@]} $@ | tee $info_filename
