#!/bin/bash
cd "$(dirname "$0")"
awk -v w=$1 'NR%2==w' rlf18_jobs.txt | while read k a b; do
  if [ "$k" = held ]; then t="${a}_${b}"; else t="$a"; fi
  [ -f out_v19/rlf18/${k}_${t}.json ] && continue
  python3 rlf18.py $k $t >> out_v19/rlf18/log_$1.txt 2>&1
done
