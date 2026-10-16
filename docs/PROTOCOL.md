# 评测协议（PROTOCOL）

> 划分定义见 [`SPLITS.md`](SPLITS.md)；指标定义见 [`METRICS.md`](METRICS.md)。

## 一、三个集合，三种用途

| 集合 | 条数 | 用途 | 允许调参？ |
|---|---|---|---|
| **M1** | 20 | 开发/迭代 | ✅ 允许反复试错 |
| **M2** | 300 | 定成绩 | ❌ **冻结后才跑，跑完不许回头调参** |
| **M3** | 1261 | 最终验收 | ❌ M2 定稿后一次性验收 |

**这是防止数据泄漏的核心规则。** 在 M2 上看到成绩后回头改提示词，M2 就变成了训练集，
它的成绩不再代表泛化能力——此时必须换 M3 报成绩。

## 二、标准流程

```bash
# 0. 一次性：下载数据、生成划分、生成描述
python3 scripts/download_data.py
python3 scripts/make_splits.py
python3 scripts/gen_descriptions.py --split M1 --workers 8

# 1. 在 M1 上迭代方案（随便试）
python3 baselines/b3_crop_ladder.py --split M1 --backend qwen --tag dev_b3

# 2. 方案冻结 → 在 M2 上定成绩（每个方案跑 5 次）
python3 baselines/b3_crop_ladder.py --split M2 --backend qwen --runs 5 --tag b3_qwen_M2

# 3. 统一评估（必须用这个脚本报告）
python3 scripts/evaluate.py 'baselines/results/raw/b3_qwen_M2_r*.jsonl' --labels 'B3·Qwen'
```

## 三、报告口径（硬性）

1. **必须用 `scripts/evaluate.py` 报告**，它同时给出：
   命中率 / ERROR / 单次耗时 / 调用次数 / 落点质量 / 随机基线对照。
   **只报命中率是不合格的**——命中只表示落点在真值框内，不表示落得准（见 METRICS.md）。
2. **必须报重复次数**：单次运行的 ±5pp 波动是噪声。默认 `--runs 5`。
3. **必须注明**：后端 + 模型名 + 描述文件 + 划分 + 运行日期。
4. **必须报 ERROR 数**：ERROR 会从命中率的分母里被剔除，掩盖稳定性问题。

## 四、可比性前提

两次成绩可直接比较，当且仅当以下全部相同：

| 项 | 说明 |
|---|---|
| 划分文件 | `splits/<split>.json` 的哈希 |
| **描述版本** | `descriptions/<版本>/<split>.json` —— **⚠️ 必须锁定版本，见下** |
| 图片缓存 | `data/images/` 全量就位（缺图会静默跳过样例）|
| 提示词 | `bench/prompts.py` |
| 后端模型 | **⚠️ 浮动别名会漂移，见下** |

### ⚠️ 两个已知的不可复现来源

**① 描述文本不可重新生成——必须按版本冻结。**
实测：用**原始脚本**重跑同样样例，产出的描述与已冻结的版本不一致——
请求参数完全一致（PNG、`temperature=0`、`max_tokens`），差异来自模型本身。

**因此描述按版本冻结进仓库**：`descriptions/v1/`（历史，只有 M1）、`descriptions/v2/`（当前，全量）。
基线通过 `--desc-version` 指定版本，结果会自动记录该版本（结果行 `desc_version` 字段 + 文件名 + 汇总表列）。
**跨版本的成绩不可比较**；正式成绩一律用 V2。
详见 [`../descriptions/README.md`](../descriptions/README.md)。

**② 模型别名会漂移。**
`qwen3.8-flash` / `deepseek-flash` 都是**浮动别名**，供应商更新后行为会变。
**因此"某个基线的历史成绩"与"今天重跑同一脚本的成绩"可能不同，这是正常的，不是 bug。**
报告时必须注明运行日期。

## 五、目录约定

```
baselines/results/raw/<baseline>_<backend>_<split>_r<N>.jsonl   # 每次运行的原始结果
baselines/RESULTS.md                                  # 汇总表（脚本生成）
```

生成汇总表：

```bash
python3 scripts/make_report.py
```
