#!/bin/sh
# 6 条概率性样例，各跑 5 次
cd "$(dirname "$0")" || exit 1
. /tmp/qw.env
IDS="android_studio_macos_69,fruitloops_windows_50,vivado_windows_79,pycharm_macos_1,origin_windows_1,vscode_macos_45"
for r in 1 2 3 4 5; do
  echo "===== 第 $r 轮 $(date +%H:%M:%S) ====="
  python3 -u verify_loop.py --ids "$IDS" \
    --desc-file results/l1/tightcrop/descriptions_v5h.json \
    --tag "rep$r" --max-iter 3 --cursor-px 48
  echo "===== 第 $r 轮结束 $(date +%H:%M:%S) ====="
done
echo "ALL_DONE"
