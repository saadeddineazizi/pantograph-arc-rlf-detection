#!/bin/bash
cd /home/claude/r17
awk -v w=$1 'NR%2==w' theory_jobs.txt | while read i; do [ -f out_v19/theory2/$i.json ] || python3 theory_flip2.py $i >> out_v19/theory2/log_$1.txt 2>&1; done
