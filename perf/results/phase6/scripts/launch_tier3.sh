#!/usr/bin/env bash
# Rent the RTX 5080 (race3 host), wait for SSH, upload code and scripts, start tier2.sh.
S="${SCRATCH:-.}"
K="$S/vast_known_hosts"
PY="/d/corridor key/CorridorKey2.0/venv/Scripts/python.exe"
LABEL=ck-perf-tier3-5080
vastai create instance 55041076 --image pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel --disk 150 \
  --ssh --direct --label $LABEL --raw 2>&1 | grep -E "success|new_contract"
ID=""
for i in $(seq 1 45); do
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
echo "$ID" > "$S/tier3_instance.txt"
url=$(vastai ssh-url "$ID")
HOST=$(echo "$url" | sed 's#ssh://root@##; s#:.*##'); PORT=$(echo "$url" | sed 's#.*:##')
echo "ssh $HOST:$PORT" | tee "$S/tier3_ssh.txt"
SSH="ssh -p $PORT -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$K root@$HOST"
for i in $(seq 1 20); do $SSH true 2>/dev/null && break; sleep 15; done
scp -q -P "$PORT" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$K" "$S/tier3_code.tgz" "root@$HOST:/root/code.tgz"
scp -q -P "$PORT" -o UserKnownHostsFile="$K" "$S/tier3_race.sh" "$S/selfstop.sh" "$S/ckd_setup.sh" "$S/ckd_trt_setup.sh" "$S/ckd_drive.py" "root@$HOST:/root/"
$SSH "sed -i 's/\r$//' /root/*.sh; chmod +x /root/*.sh; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; nproc; (nohup bash /root/tier3_race.sh > /root/tier3.log 2>&1 &); echo started" 2>&1 | grep -v "Welcome\|Have fun"
