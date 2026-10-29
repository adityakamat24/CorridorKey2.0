#!/usr/bin/env bash
# Stop this Vast instance (GPU billing ends) after $1 seconds.
sleep "$1"
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^CONTAINER_(ID|API_KEY)=' | sed 's/^/export /')"
curl -s -X PUT "https://console.vast.ai/api/v0/instances/${CONTAINER_ID}/" \
  -H "Authorization: Bearer ${CONTAINER_API_KEY}" -H "Content-Type: application/json" \
  -d '{"state": "stopped"}' > /root/selfstop.log 2>&1
