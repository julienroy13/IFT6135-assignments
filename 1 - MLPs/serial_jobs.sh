#!/bin/sh
GPU="$1"
echo "GPU = $GPU"
shift

while [ "$#" -gt "0" ]
do
  echo "Running experiment for config$1"
  time python train.py --config $1 --gpu $GPU > results/config"$1".out
  shift
done  