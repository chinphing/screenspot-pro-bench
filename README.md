# ScreenSpot-Pro 元素定位 Benchmark

GUI 元素定位的评测框架：给定界面截图 + 目标元素描述，让模型/智能体输出落点，
用**命中率 + 落点质量**两层指标评估。

数据集：[`likaixin/ScreenSpot-Pro`](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro)（1581 条 / 26 个应用）。

---

## 快速开始

```bash
pip install -r requirements.txt          # 只需 Pillow

# 1. 下载数据（约 3.1GB，8 并发约 20 分钟）
python3 scripts/download_data.py

# 2. 生成划分 M1/M2/M3
python3 scripts/make_splits.py

# 3. 生成描述（M1 已随仓库提供；M2/M3 需自行生成）
python3 scripts/gen_descriptions.py --split M1 --workers 8

# 4. 跑一个基线
export QW_KEY=...  QW_BASE=...            # 或 DS_KEY / DS_BASE
python3 baselines/b3_crop_ladder.py --split M1 --backend qwen --runs 5 --tag b3_qwen_M1

# 5. 出成绩
python3 scripts/evaluate.py 'baselines/results/raw/b3_qwen_M1_r*.jsonl' --labels 'B3·Qwen'
python3 scripts/make_report.py            # 汇总表 → baselines/RESULTS.md
```

或一键：

```bash
bash run_all.sh
```

---

## 三个基线（机制演进）

| | 基线 | 机制一句话 | 谁决定下一步 |
|---|---|---|---|
| B1 | [`b1_direct.py`](baselines/b1_direct.py) | 整图 + 短提示，一次出框 | — |
| B2 | [`b2_agent_marker.py`](baselines/b2_agent_marker.py) | 原图+放大图，模型自己调 `done()` | **模型** |
| B3 | [`b3_crop_ladder.py`](baselines/b3_crop_ladder.py) ★ | 分阶段 + **只送裁片** + 阶梯扩大视野 | **模型** |

**→ B2/B3 都是智能体模式；B1 是无循环的下界参照。**

### M1 成绩（20 条 × 5 次）

| 基线 | 后端 | 命中 | ERROR | 单次 | 调用 | 到中心 | 距边 | 擦边 | 失败距离 |
|---|---|---|---|---|---|---|---|---|---|
| B2 `square` | qwen3.8-flash | **86%** | 0 | 10.3s | 2.79 | 4px | **50%** | 30% | 161px |
| B3 | qwen3.8-flash | 82% | 0 | 7.9s | 2.68 | 4px | 36% | 30% | 87px |
| B3 | deepseek-flash | 74% | 0 | **3.9s** | 2.96 | 6px | 36% | 36% | 248px |
| B1 | qwen3.8-flash | 60% | 0 | **1.7s** | 1.00 | 6px | 34% | 42% | 77px |

> 随机基线：距边中位 **29%**、擦边率 **44%**。全部基线都显著优于随机。
> ⚠️ **4pp 差距不显著**（86 vs 82，SE≈5.2pp）。选型请看落点质量与耗时，不要只看排名。

**核心结论**：
1. **智能体模式（模型自己决定下一步）全面优于脚本硬编码流程**——早期那代脚本驱动的校验循环
   在 qwen 上同样 82%，但慢 **2.3 倍**（18.5s），在 DS 上精度还输（70% < 74%），已被淘汰。
2. **B3 的"只送裁片"结构对弱模型收益最大**——同一个 qwen3.7-flash，
   从旧机制 33% 提升到 71%（**+38pp**），全错样例从 10/20 降到 2/20。

详见 [`baselines/README.md`](baselines/README.md)。

---

## 目录结构

```
screenspot_grid/
├── README.md              入口（本文件）
├── run_all.sh             一键复现
├── requirements.txt
├── .gitignore             排除 data/、baselines/results/、legacy/ 的旧结果
│
├── bench/                 核心库（可 import）
│   ├── dataset.py         数据集加载、图片下载缓存、坐标换算
│   ├── clients.py         模型客户端（DeepSeek / Qwen 统一接口）
│   ├── vision.py          缩放、裁片、画标记
│   ├── prompts.py         提示词与工具定义
│   ├── metrics.py         指标（命中率 + 落点质量 + 随机基线）
│   └── runner.py          基线共用的 CLI 驱动
│
├── baselines/             3 个基线脚本 + RESULTS.md（成绩汇总）+ results/raw/（原始逐条结果，不进 git）
├── scripts/               download_data / make_splits / gen_descriptions / evaluate / make_report
├── splits/                M1.json M2.json M3.json + 说明
├── descriptions/          每个样例的描述（M1 已冻结）+ 说明
├── docs/                  PROTOCOL / METRICS / SPLITS / ARCHIVE1-4（实验历史）
├── data/                  annotations/（上游标注）+ images/（图片缓存）—— 不进 git，download_data.py 可取回
└── legacy/                53 个实验脚本（入库，供 diff 复核）+ 旧结果（不进 git）
```

---

## 三条必读规则

### ① 数据集与图片不进版本库

`data/`（标注 + 图片 3.3GB）、`legacy/` 的旧结果、`baselines/results/`（原始逐条结果）已被 `.gitignore` 排除；**`legacy/` 的脚本**与成绩汇总表 `baselines/RESULTS.md` **进库**。
换机器只需跑 `scripts/download_data.py`。

### ② 描述是固定输入，**按版本冻结**

实测：用**原始脚本**重跑同样样例，产出的描述与已冻结版本**不一致**——
请求参数完全一致（PNG、`temperature=0`），差异来自模型本身（浮动别名会漂移）。

**因此描述按版本冻结**：`descriptions/v1/`（历史，只有 M1）、`descriptions/v2/`（当前，全量）。

| 版本 | 覆盖 | 说明 |
|---|---|---|
| **v1** | 只有 M1（20 条）| `docs/ARCHIVE*.md` 里的 M1 历史成绩用的就是它 |
| **v2** | M1+M2+M3（1581 条）| 三个集合同一次生成，**正式成绩一律用这版** |

基线通过 `--desc-version` 指定版本（默认 v2），**结果会自动记录版本**
（结果行 `desc_version` 字段 + 文件名带版本段 + 汇总表「描述版本」列）。
**跨版本的成绩不可比较。** 详见 [`descriptions/README.md`](descriptions/README.md)。

### ③ M1 调参 / M2 定成绩 / M3 验收

| 集合 | 条数 | 用途 |
|---|---|---|
| M1 | 20 | 迭代（随便试）|
| M2 | 300 | **冻结后才跑，跑完不许回头调参** |
| M3 | 1261 | 最终验收 |

**在 M2 上看到成绩后回头改提示词，M2 就变成了训练集。**
详见 [`docs/PROTOCOL.md`](docs/PROTOCOL.md)。

---

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/PROTOCOL.md`](docs/PROTOCOL.md) | 评测协议、报告口径、可比性前提 |
| [`docs/METRICS.md`](docs/METRICS.md) | 指标定义与随机基线判读 |
| [`docs/SPLITS.md`](docs/SPLITS.md) | 划分规范与分布吻合度 |
| [`docs/ARCHIVE*.md`](docs/README.md) | 实验历史（38 节，含机制演进全过程）|

## 引用

数据集来自 ScreenSpot-Pro，请引用其原始工作：

```bibtex
@inproceedings{li2025screenspotpro,
  title={ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use},
  author={Li, Kaixin and others},
  year={2025}
}
```

## 许可证

本仓库代码采用 [MIT 许可证](LICENSE)。
数据集 ScreenSpot-Pro 版权归原作者所有；本仓库不包含数据集文件（图片与标注由 `scripts/download_data.py` 从上游取回）。
