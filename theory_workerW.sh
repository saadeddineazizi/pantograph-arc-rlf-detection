#!/bin/bash
cd /home/claude/r17
awk -v w=$1 'NR%2==w' theory3_jobs.txt | while read i; do [ -f out_v19/theoryW/$i.json ] || python3 theory_w.py $i >> out_v19/theoryW/log_$1.txt 2>&1; done
