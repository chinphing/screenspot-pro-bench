# 结果目录（baselines/results/）

## 文件

| 路径 | 说明 |
|---|---|
| [`../RESULTS.md`](../RESULTS.md) | **汇总成绩表**（`scripts/make_report.py` 自动生成，勿手改）|
| `raw/<基线>_<后端>_<划分>_<描述版本>_<模型>[_r<N>\|s<N>].jsonl` | 每次运行的原始逐条结果 |

命名约定由 `scripts/make_report.py` 解析。**四段都不能省**：

| 段 | 为什么不能省 |
|---|---|
| `<描述版本>` | 标明这批结果用哪一版描述跑出来（见 [`../../descriptions/README.md`](../../descriptions/README.md)）。**跨版本的成绩不可比较。** |
| `<模型>` | B2/B3 的 `run_one` 是从 legacy **逐字拷贝**的，写不进结果行的 `model` 字段——**模型只能靠文件名区分**，否则 qwen3.7 与 qwen3.8 的成绩会撞名合并。标签由 `bench/clients.py` 的 `model_tag()` 统一生成（`q38` / `q37` / `dsflash`）。|

同一组的多份会自动合并统计：`_r1`/`_r2`… 是重复运行；`_s1`/`_s2`… 是并发分片（见 [`../../scripts/run_parallel.py`](../../scripts/run_parallel.py)）。

## 原始结果的字段

| 字段 | 说明 |
|---|---|
| `id` | 样例 id |
| `status` | `OK` / `ERROR` / `L1_NO_OUTPUT` / `MAX_ITER` / `TARGET_NOT_IN_CROP` … |
| `final_px` | 最终落点（原图像素）← **命中率/距边/擦边**的依据 |
| `final_bbox` | 最终框（B1 直接写；B2/B3 需从 `trace` 末个 `move` 的 `bbox_orig` 还原）← **宏微观覆盖率**的依据 |
| `gt_bbox` / `gt_img_size` | 真值框与图像尺寸 |
| `hit` | 落点是否在真值框内 |
| `latency_s` | 该条端到端耗时 |
| `n_move` | 模型调用次数（不含 L1）|
| `app` / `ui_type` / `platform` / `group` / `target_len` | 样例元信息（供 `--by` 分组；legacy 结果行缺这些字段，由 `bench/metrics.py` 的 `enrich()` 按 id 补齐）|
| `backend` / `model` | 后端与模型名（仅新写的基线会写；B2/B3 靠文件名）|
| **`desc_version`** | **描述版本**（v1/v2）——横向对比成绩前必须核对这一项 |

> ⚠️ `trace` 里的 `bbox_orig` **名字骗人**：它是**归一化 0–1000**，不是原图像素。
> 还原要乘 `gt_img_size / 1000`（`bench/metrics.py` 的 `final_box()` 已处理）。

## 运行记录

| 划分 | 基线 | 后端/模型 | 描述 | 分片 | 用时 | 命中 | 备注 |
|---|---|---|---|---|---|---|---|
| M2 | B3 | qwen3.8-flash | v2 | 5 | 15 min | **86%** | 首次 5 分片并发 |
| M2 | B2 | qwen3.8-flash | v2 | 8 | 10 min | 84% | |
| M2 | B1 | qwen3.8-flash | v2 | 10 | 1.3 min | 74% | |
| M2 | B3 | deepseek-flash | v2 | 20 | ~2 min | 70% | DS 端点 20 并发零失败 |
| M2 | B2 | deepseek-flash | v2 | 20 | ~3 min | 37% | 弱模型上「画标记」崩塌 |
| M2 | B3 | qwen3.7-flash | v2 | 10 | ~5 min | 74% | |
| M2 | B2 | qwen3.7-flash | v2 | 10 | ~9 min | 33% | 2 条瞬时 ERROR（0.7%）|
| **M3** | **B3** | **qwen3.8-flash** | v2 | 10 | **21 min** | **83%** | **最终验收，1261 条全跑完** |

M2 与 M3 的分片结果均与 `splits/<划分>.json` 逐条核对过：**无缺、无重、无多余**。

## ⚠️ 部分 M1 结果的来源（重要）

`raw/` 里的 **M1 `_v1`** 结果**不是新基线跑出来的**，而是从 `legacy/results_l1/`、`legacy/results_loop/`
按新命名**归位**的历史结果——它们是旧实验脚本产出的：

| 新命名 | 原始文件 | 机制对应 |
|---|---|---|
| `b3_crop_ladder_qwen_M1_v1_r1-5` | `cursorZF_qfix1-5` | ZF = 分阶段+只送裁片+阶梯 |
| `b3_crop_ladder_deepseek_M1_v1_r1-5` | `cursorZF_fx1-5` | 同上（DeepSeek 后端）|
| `b2_agent_marker_qwen_M1_v1_r1-5` | `cursorZ6_q1-5` | Z6 = 智能体+方块标记+done |
| `b1_direct_qwen_M1_v1` | **新跑（新基线）** | B1 单图直出 |

**注意 `_v1` 段**：这些历史结果用的都是 **V1** 描述（当时唯一存在的一版，且只有 M1）。
