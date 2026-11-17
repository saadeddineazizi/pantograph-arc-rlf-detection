#!/bin/bash
cd "$(dirname "$0")"; w=$1; n=0
while read -r k t; do
  n=$((n+1)); [ $(( (n-1) % 2 )) -ne $w ] && continue
  [ -f out_v19/rlf/${k}_${t}.json ] && continue
  python3 rlf19.py $k $t >> logs19/rlf_w$w.log 2>&1 || echo "FAIL $k $t" >> logs19/rlf_w$w.log
done < jobs_rlf.txt
