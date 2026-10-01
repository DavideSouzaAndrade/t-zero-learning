#!/bin/bash
IFS='|' read -r name seed ov <<< "$1"
cd /home/claude/t-zero-learning
OMP_NUM_THREADS=1 python scripts/run_logged.py logs/${name}__s${seed}.jsonl --config a2c_cartpole --override seed=$seed capture_video=false $ov > logs/${name}__s${seed}.out 2>&1
echo "done $name s$seed $(date +%T)" >> logs/progress.txt
