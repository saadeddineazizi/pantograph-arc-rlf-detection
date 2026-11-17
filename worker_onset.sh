#!/bin/bash
cd "$(dirname "$0")"; w=$1; n=0
while read -r i D; do
  n=$((n+1)); [ $(( (n-1) % 2 )) -ne $w ] && continue
  [ -f out_v19/onset/${i}_D${D}.json ] && continue
  python3 onset19.py $i $D >> logs19/onset_w$w.log 2>&1 || echo "FAIL $i $D" >> logs19/onset_w$w.log
done < jobs_onset.txt
