#!/bin/bash
cd "$(dirname "$0")"
W=$1
awk -v w=$W 'NR%2==w' eta_jobs.txt | while read k a b; do python3 eta19.py $k $a $b >> out_v19/eta/log_$W.txt 2>&1; done
echo done >> out_v19/eta/log_$W.txt
