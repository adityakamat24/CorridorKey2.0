#!/usr/bin/env bash
S="${SCRATCH:-.}"
K="$S/vast_known_hosts"
PY="/d/corridor key/CorridorKey2.0/venv/Scripts/python.exe"
OFFER=$1 LABEL=$2 NAME=$3
vastai create instance $OFFER --image pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel --disk 60 \
  --ssh --direct --label $LABEL --raw 2>&1 | grep -E "success|new_contract"
ID=""
for i in $(seq 1 60); do
  line=$(vastai show instances --raw 2>/dev/null | "$PY" -c "
import json, sys
for d in json.load(sys.stdin):
    if d.get('label') == '$LABEL':
        print(d['id'], d.get('actual_status'))
")
  echo "poll $i: $line"
  set -- $line
  ID=${1:-}
  [ "${2:-}" = running ] && break
  sleep 20
done
[ -n "$ID" ] || { echo "no instance"; exit 1; }
echo "$ID" > "$S/${NAME}_instance.txt"
url=$(vastai ssh-url "$ID")
HOST=$(echo "$url" | sed 's#ssh://root@##; s#:.*##'); PORT=$(echo "$url" | sed 's#.*:##')
echo "ssh $HOST:$PORT" | tee "$S/${NAME}_ssh.txt"
SSH="ssh -p $PORT -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$K root@$HOST"
for i in $(seq 1 20); do $SSH true 2>/dev/null && break; sleep 15; done
scp -q -P "$PORT" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$K" "$S/show_code.tgz" "root@$HOST:/root/code.tgz"
scp -q -P "$PORT" -o UserKnownHostsFile="$K" "$S/arch_check.sh" "$S/selfstop.sh" "$S/det_run.py" "root@$HOST:/root/"
$SSH "sed -i 's/\r$//' /root/*.sh /root/*.py; chmod +x /root/*.sh; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; (nohup bash /root/arch_check.sh > /root/arch.log 2>&1 &); echo started" 2>&1 | grep -v "Welcome\|Have fun"
