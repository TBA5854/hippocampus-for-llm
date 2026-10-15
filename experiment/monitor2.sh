#!/usr/bin/env bash
# monitor2.sh <round> — same as monitor.sh but for box 2 (<user>@<box2-host>), which needs a
# password helper for ssh. Logs there are named *_box2.log / _progress_box2.log so the two boxes'
# files never collide in the same round folder.
round=${1:?round, e.g. r20}
box=<user>@<box2-host>
helper=${B2SSH:?set B2SSH to an ssh wrapper that authenticates to box 2}
# box2's ~/research/framework is owned by another account and is not writable by us; our work lives in ~/fw
remote=${B2ROOT:-fw}/experiment/$round
local_dir=$(dirname "$0")/$round
# persist across re-arms: a fresh mktemp replays the whole log every time the
# 30-minute Monitor expires and is restarted
seen="$local_dir/.seen_box2"
touch "$seen"
down=0
# A round folder often holds ROUND DONE from an EARLIER pass (a rerun, a second
# script). Exiting on any occurrence kills the watch instantly; only a NEW one counts.
done0=$(grep -c "ROUND DONE" "$local_dir/logs/_progress_box2.log" 2>/dev/null || echo 0)
while true; do
  if timeout 60 rsync -az -e "$helper" "$box:$remote/logs/" "$local_dir/logs/" 2>/dev/null; then
    timeout 60 rsync -az -e "$helper" "$box:$remote/results/" "$local_dir/results/" 2>/dev/null
    [ "$down" = 1 ] && echo "[net] box2 reachable again" && down=0
    grep -HE "^=== |Traceback|Error|error:|CUDA|out of memory|Killed|nan" \
        "$local_dir"/logs/*_box2.log 2>/dev/null | sed "s#^$local_dir/logs/##" |
      while IFS= read -r line; do
        grep -qxF "$line" "$seen" || { echo "$line"; echo "$line" >> "$seen"; }
      done
    now=$(grep -c "ROUND DONE" "$local_dir/logs/_progress_box2.log" 2>/dev/null || echo 0)
    [ "$now" -gt "$done0" ] && {
      echo "[$round/box2] ROUND DONE — logs and results synced to $local_dir"; exit 0; }
  else
    [ "$down" = 0 ] && echo "[net] box2 unreachable, retrying every 60s" && down=1
  fi
  sleep 60
done
