#!/usr/bin/env bash
S="${SCRATCH:-.}"
scp -q -P 29155 -o ConnectTimeout=20 "$S/tier3d.sh" root@99.82.217.53:/root/tier3d.sh
ssh -o ConnectTimeout=20 -p 29155 root@99.82.217.53 'sed -i "s/\r$//" /root/tier3d.sh; setsid nohup bash /root/tier3d.sh > /root/tier3d.log 2>&1 < /dev/null & sleep 2; cat /root/tier3d.log' 2>&1 | grep -v -E "^Welcome|^Have fun"
ssh -o ConnectTimeout=20 -p 29071 root@99.82.217.53 'setsid nohup bash /root/tier2f_acc.sh > /root/tier2f.log 2>&1 < /dev/null & sleep 2; ps -eo cmd | grep -c "[t]ier2f_acc"' 2>&1 | grep -v -E "^Welcome|^Have fun"
echo "box 2: Desktop + jer race queued after our runs; box 1: jer accuracy queued after tier2e"
