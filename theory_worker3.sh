#!/bin/bash
cd /home/claude/r17
awk -v w=$1 'NR%2==w' theory3_jobs40.txt | while read i; do [ -f out_v19/theory3/$i.json ] || python3 theory_flip3.py $i >> out_v19/theory3/log_$1.txt 2>&1; done
