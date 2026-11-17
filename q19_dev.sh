#!/bin/bash
cd "$(dirname "$0")"
(for rc in "main clean" "iid1 clean" "ge3 clean"; do python3 dev_base.py $rc > logs19/devb_${rc// /_}.log 2>&1; python3 mix_dev2.py $rc > logs19/devm_${rc// /_}.log 2>&1; done) &
(for rc in "main sd8" "iid1 sd8" "ge3 sd8"; do python3 dev_base.py $rc > logs19/devb_${rc// /_}.log 2>&1; python3 mix_dev2.py $rc > logs19/devm_${rc// /_}.log 2>&1; done) &
wait
echo DEV19DONE
