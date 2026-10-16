# docs/ 索引

| 文档 | 内容 |
|---|---|
| [`PROTOCOL.md`](PROTOCOL.md) | 评测协议：M1/M2/M3 用途、标准流程、报告口径、可比性前提 |
| [`METRICS.md`](METRICS.md) | 指标定义：命中率 + 落点质量 + 随机基线判读 |
| [`SPLITS.md`](SPLITS.md) | 数据划分规范：M1/M2/M3 抽样方法与分布吻合度 |
| [`ARCHIVE.md`](ARCHIVE.md) … [`ARCHIVE4.md`](ARCHIVE4.md) | 实验历史（按时间顺序，**保留原始记录不改写**）|

## ⚠️ 归档里的路径已失效——新旧对照

`ARCHIVE*.md` 是重构**之前**写的实验记录，里面的脚本路径已随本次规范化改变。
查证历史时按下表换算：

| 归档里的旧路径 | 现在的位置 |
|---|---|
| `harness.py` | `bench/dataset.py` + `bench/vision.py` |
| `ds_client.py` / `qwen_client.py` | `bench/clients.py`（统一接口）+ **`bench/compat/`（逐字拷贝，供基线保真）** |
| `gen_desc.py` | `scripts/gen_descriptions.py`（新增并发）|
| `eval_full.py` | `scripts/evaluate.py` + `bench/metrics.py` |
| `dl_all.py` | `scripts/download_data.py` |
| `verify_loop.py` | `legacy/b2_verify_loop.py`（已淘汰）+ **`bench/compat/verify_loop.py`**（供 ZF/Z6 复用 3 个符号）|
| `ds_cursorZ6.py` | `baselines/b2_agent_marker.py`（**原样拷贝**，6 处改动）|
| `ds_cursorZ4.py` | 并入 `baselines/b2_agent_marker.py`（Z4/Z6 差异在标记画法）|
| `ds_cursorZF.py` | `baselines/b3_crop_ladder.py`（**原样拷贝**，6 处改动）|
| `ds_cursor*.py` / `ds_agent*.py`（其余 ~44 个）| `legacy/` |
| `eval_bbox.py` / `l1_only.py` / `two_level.py` / `verify_tile.py` / `ds_tiles.py` | `legacy/` |
| `_prompt*.txt` / `run_*.sh` | `legacy/` |
| `results/l1/` | `legacy/results_l1/` |
| `results/loop/` | `legacy/results_loop/` |
| `SPLITS.md` | `docs/SPLITS.md` |

**M1 的关键结果**已按新命名归位到 `baselines/results/raw/`，映射见 [`../baselines/results/README.md`](../baselines/results/README.md)。

## 归档索引

| 归档 | 覆盖范围 |
|---|---|
| `ARCHIVE.md` | 早期：基线偏移版、W 版 |
| `ARCHIVE2.md` | Z 系列机制演进 |
| `ARCHIVE3.md` | ZB/ZC/ZD/ZE 与 DeepSeek 对比 |
| `ARCHIVE4.md` | **主线归档（2023 行 / 38 节）**：§0 结论速览、§22 初筛集、§25 DS 差异、§31 ZF、§34 思考参数、§36 落点质量、§37 常驻评估脚本 + qwen3.7 判定、§38 M1/M2/M3 划分 |
