#!/bin/sh
# 组合方案：单图+循环为主；非成功时自动走「17图选图 → 局部图循环」兜底
set -e
cd "$(dirname "$0")"
. /tmp/ds.env
IDS=$(paste -sd, /tmp/ids20.txt)
DESC=results/l1/tightcrop/descriptions_v7.json
for r in 1 2 3 4 5; do
  echo "===== 第 $r 轮 阶段1：单图+循环 ====="
  python3 -u verify_loop.py --backend deepseek --ids "$IDS" --l1-px 2.5e6 \
    --desc-file "$DESC" --tag h1r$r --max-iter 3 --cursor-px 48 > /dev/null 2>&1
  # 找出非成功的 id
  FAIL=$(python3 -c "
import json
f=[json.loads(l) for l in open('results/loop/h1r$r.jsonl')]
bad=[x['id'] for x in f if x.get('status')!='SUCCESS']
print(','.join(bad))")
  if [ -n "$FAIL" ]; then
    echo "  阶段1 失败 $(echo $FAIL | tr ',' '\n' | wc -l | tr -d ' ') 条 → 触发兜底"
    echo "===== 第 $r 轮 阶段2：17图选图 ====="
    python3 -u ds_17.py --ids "$FAIL" --desc-file "$DESC" --tag h2r$r > /dev/null 2>&1
    echo "===== 第 $r 轮 阶段3：局部图上循环 ====="
    python3 -u verify_tile.py --ids "$FAIL" \
      --l1-file results/l1/g17_h2r$r.jsonl --desc-file "$DESC" \
      --tag h3r$r --max-iter 3 --cursor-px 48 > /dev/null 2>&1
  else
    echo "  阶段1 全部成功，无需兜底"
  fi
  echo "===== 第 $r 轮结束 ====="
done
echo ALL_DONE
