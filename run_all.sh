#!/usr/bin/env bash
# 一键复现：检查环境 → 下载数据 → 生成划分 → 检查描述 → 跑基线 → 出成绩
#
#   bash run_all.sh                     # 默认：M1 + b4_crop_ladder + qwen
#   SPLIT=M1 BACKEND=qwen BASE=b4_crop_ladder RUNS=5 bash run_all.sh
#   SKIP_DOWNLOAD=1 bash run_all.sh     # 图片已就绪时跳过下载
set -euo pipefail
cd "$(dirname "$0")"

SPLIT="${SPLIT:-M1}"
BACKEND="${BACKEND:-qwen}"
BASE="${BASE:-b4_crop_ladder}"
RUNS="${RUNS:-1}"
TAG="${TAG:-${BASE}_${BACKEND}_${SPLIT}}"
PY="${PY:-python3}"

echo "=============================================================="
echo " ScreenSpot-Pro benchmark —— 一键复现"
echo "   划分=$SPLIT  后端=$BACKEND  基线=$BASE  重复=$RUNS  输出=$TAG"
echo "=============================================================="

echo -e "\n[1/6] 依赖检查"
$PY -c "import PIL; print('  ✅ Pillow', PIL.__version__)" || {
  echo "  ❌ 缺 Pillow：pip install -r requirements.txt"; exit 1; }

echo -e "\n[2/6] 数据下载"
if [ "${SKIP_DOWNLOAD:-0}" = "1" ]; then
  echo "  跳过（SKIP_DOWNLOAD=1）"
else
  $PY scripts/download_data.py || echo "  ⚠️ 有失败项，可重跑本脚本补齐"
fi

echo -e "\n[3/6] 生成划分"
$PY scripts/make_splits.py

echo -e "\n[4/6] 描述文件检查"
DESC="descriptions/${SPLIT}.json"
if [ -f "$DESC" ]; then
  N=$($PY -c "import json;print(len(json.load(open('$DESC',encoding='utf-8'))))")
  echo "  ✅ $DESC（$N 条）"
else
  echo "  ⚠️ 缺 $DESC —— 现在生成��耗时较长）"
  $PY scripts/gen_descriptions.py --split "$SPLIT" --workers 8
fi

echo -e "\n[5/6] 跑基线"
if [ ! -f "baselines/${BASE}.py" ]; then
  echo "  ❌ 找不到 baselines/${BASE}.py"; echo "  可选：$(ls baselines/*.py | xargs -n1 basename | sed 's/\.py//' | tr '\n' ' ')"; exit 1
fi
$PY "baselines/${BASE}.py" --split "$SPLIT" --backend "$BACKEND" --runs "$RUNS" --tag "$TAG"

echo -e "\n[6/6] 评估"
$PY scripts/evaluate.py "baselines/results/raw/${TAG}_r*.jsonl" --labels "${BASE}·${BACKEND}" 2>/dev/null \
  || $PY scripts/evaluate.py "baselines/results/raw/${TAG}.jsonl" --labels "${BASE}·${BACKEND}"
$PY scripts/make_report.py

echo -e "\n=============================================================="
echo " ✅ 完成"
echo "   原始结果  baselines/results/raw/${TAG}*.jsonl"
echo "   汇总表    baselines/RESULTS.md"
echo "=============================================================="
