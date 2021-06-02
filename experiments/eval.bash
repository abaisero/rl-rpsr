#!/bin/bash

pomdp=$1
shift
env=$1
shift
policy=$1
shift

mkdir -p evals/ logs/

log_filename=logs/eval.$pomdp.$env.$policy.log

cmd_options=()
cmd_options+=(--load-core-psr cores/$pomdp.psr.core)
cmd_options+=(--load-core-rpsr cores/$pomdp.rpsr.core)

cmd_options+=(--load-vf-bsr vfs/$pomdp.bsr.vf)
cmd_options+=(--load-vf-psr vfs/$pomdp.psr.vf)
cmd_options+=(--load-vf-rpsr vfs/$pomdp.rpsr.vf)

cmd_options+=(--log-filename $log_filename --log-level DEBUG)

rl-psr-eval.py pomdps/$pomdp $env $policy ${cmd_options[@]} $@
