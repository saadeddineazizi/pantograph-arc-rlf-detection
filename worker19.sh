#!/bin/bash
# usage: worker19.sh <worker index 0|1>; takes every second job of jobs19.txt; skips finished ones
cd "$(dirname "$0")"
w=$1; n=0
while read -r kind a b; do
  n=$((n+1)); [ $(( (n-1) % 2 )) -ne $w ] && continue
  tag=$a; [ -n "$b" ] && tag="${a}_${b}"
  [ -f out_v19/$kind/$tag.json ] && continue
  echo "$(date +%T) start $kind $a $b" >> logs19/worker$w.log
  python3 eval19.py $kind $a $b > logs19/${kind}_$tag.log 2>&1 || echo "FAIL $kind $a $b" >> logs19/worker$w.log
done < jobs19.txt
echo "$(date +%T) worker $w done" >> logs19/worker$w.log
