# JustRL 复现实验设计

> 目标：重走论文从"零"到结论的完整论证链，逐步验证每一环。
> 本文档中的每条结论都基于已核实的论文全文（arXiv 2512.16649v1）与本仓库代码，并标注了出处。

---

## 0. 先厘清：论文里哪些东西需要"跑"实验

这是最容易走弯路的地方。论文共 6 表 3 图，但**大部分不需要做实验**：

| 论文元素 | 性质 | 是否需要实验 |
|---|---|---|
| Table 1 技术对比表 | 文献综述（读别人的论文填表） | ❌ 不需要 |
| Table 2 超参配置 | 配置说明 | ❌ 不需要 |
| **Table 3 弱底座结果** | 实测 | ✅ **需要（E1）** |
| Table 4 算力对比 | 由 token 预算做算术推导 | ❌ 不需要* |
| **Table 5 强底座结果** | 实测 | ✅ **需要（E2）** |
| Table 6 算力对比 | 算术推导 | ❌ 不需要* |
| Fig 1 训练曲线 | E1/E2 训练时自动记录 | ✅ 副产品 |
| Fig 2 训练动力学 | E1 训练时自动记录 | ✅ 副产品 |
| **Fig 3 消融曲线** | 实测 | ✅ **需要（E3/E4）** |

\* Table 4/6 的"2× less compute"是**估算值**，不是测量值。论文自己在 Table 4 脚注里写明：
> "∗Dynamic sampling with estimated 50% filter ratio following POLARIS."

即：作者假设动态采样会过滤掉 50% 的样本，据此推算对手的算力。**这是一个假设，不是实测**。复现时应明确记录这一点，不要试图"跑出"这个数字。

**结论：真正要跑的只有 4 个训练 run（E1/E2/E3/E4）+ 1 套评测流程。**

---

## 1. 论文的论证链（复现必须遵循的顺序）

论文不是"跑了 4 个实验然后总结"，而是一条**递进式论证链**，每一步都是下一步的前提：

```
① 别人都很复杂
   └─ Table 1：列出 10 个近期工作，每个都用 6-8 种技巧 → 建立"领域现状很复杂"
        ↓
② 我只用最简单的配方
   └─ Table 2 + §3.1：单阶段、固定超参、无 curriculum、无长度惩罚
        ↓
③ 这个配方在【弱底座】上就打赢了复杂方法
   └─ Table 3 + Table 4 + Fig 1(a)：DeepSeek-R1-Distill → 54.87%，超过 ProRL-V2 的 53.08%
        ↓
④ 而且原封不动搬到【强底座】上同样成立
   └─ Table 5 + Table 6 + Fig 1(b)：Nemotron → 64.32%，超过 QuestA 的 63.81%
        ↓
⑤ 为什么成立？因为根本没有出现那些"不稳定"
   └─ Fig 2：entropy 无漂移、reward 单调上升、长度自然收敛
        ↓
⑥ 而且那些"修复手段"反而是有害的
   └─ Fig 3：加长度惩罚 → 掉到 50%；再加鲁棒 verifier → 掉到 45%
```

**复现的关键认识**：③④⑤⑥ 是四个**独立的可证伪主张**，不是四个"跑一下"。
每一步都有明确的"如果…则不成立"判据（见下文各 Phase 的【判据】）。

---

## 2. 分阶段实验设计

### Phase 0 — 环境与数据准备（无实验）

**目的**：让后续所有实验有可运行的地基。

**要做的**：

| 项 | 目标位置 | 来源 | 备注 |
|---|---|---|---|
| DAPO-Math-17k | `train/data/DAPO/dapo-math-17k.parquet` | HF `BytedTsinghua-SIA/DAPO-Math-17k`，注意 `--repo-type dataset` | 300 MB |
| 9 个 benchmark | `data/*/test.parquet` | ✅ 仓库已自带 | 无需下载 |
| 两个底座模型 | `/volume/data/hjiang02/open_source/models/` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`、`nvidia/OpenMath-Nemotron-1.5B` | |
| 两个已发布模型 | 同上 | `hbx/JustRL-DeepSeek-1.5B`、`hbx/JustRL-Nemotron-1.5B` | Phase 1 对齐用 |
| 判分模型 | 同上 | `opencompass/CompassVerifier-3B` | 评测用，3B |

**⚠️ 必须知道的坑（已核实）**：
`dapo-math-17k.parquet` 有 **1,791,700 行，但只有 17,917 道唯一题**，是按块复制了 100 份。
（实测：row[0] == row[17917] == row[35834]，内容哈希完全相同。）

这恰好解释了论文的 4,380 步：
```
len(train_dataloader) = 1,791,700 / 256 = 6,999
6,999 × total_epochs(1) = 4,380 步（但只消费了 62.6% 的数据）
```
所以 `total_epochs=1` 跑出来就是 4,380 步，**脚本不用改**。
但要意识到：4380 步 ≈ 对那 17,917 道唯一题重复采样 **62.6 遍**。
这不是 bug —— 它正是论文"no offline difficulty filtering, no dynamic sampling"能成立的原因（重复采样替代了动态采样）。

---

### Phase 1 — 评测链路自检 ★最高优先级★（不训练）

**目的**：在做任何训练之前，先证明**你的评测是可信的**。
这是整条链的守门人：如果评测不准，后面所有训练结论都是噪声。

**科学性**：这是一个 golden check。作者已发布 JustRL 的权重和公开数字，所以你可以
**在零训练成本下验证整条评测链路**。这是论文里唯一一个"答案已知"的实验。

**做法**：
```bash
cd /volume/data/hjiang02/workspace/JustRL

# 1) 改 gen_vllm.py:27
#    NAME = "hbx/JustRL-DeepSeek-1.5B"   （或本地路径）
# 2) 改 gen_vllm.py:112 —— 论文默认 8 卡，本机只有 4 卡
#    available_workers = [0, 1, 2, 3]
# 3) 改 grade.py:36 —— 必须与 NAME 去掉 org 前缀后一致
#    NAME = "JustRL-DeepSeek-1.5B"
# 4) 必须从仓库根目录运行（gen_vllm.py:14 的 DATA_DIR="data" 是相对路径，
#    README 里写的 `cd evals` 是错的）

PYTHONPATH=evals python evals/gen_vllm.py    # 生成
PYTHONPATH=evals python evals/grade.py       # 判分
```

**产出**：`justrl_eval_outputs/JustRL-DeepSeek-1.5B/grading_results.{cv|rule}.{q|noq}.json`
（文件名随判分口径变化：`cv`/`rule` = 是否启用 CompassVerifier 兜底，`q`/`noq` = 是否把题干传给判分模型；论文基线口径是 `rule.noq`）

**判据（对齐目标）**：

| Benchmark | @N | 论文值 |
|---|---|---|
| AIME24 | 32 | 52.60 |
| AIME25 | 32 | 38.75 |
| AMC23 | 32 | 91.02 |
| MATH-500 | 4 | 91.65 |
| Minerva | 4 | 51.47 |
| Olympiad-Bench | 4 | 67.99 |
| HMMT25 | 32 | 21.98 |
| BRUMO25 | 32 | 52.71 |
| CMIMC25 | 32 | 25.63 |
| **Avg** | | **54.87** |

**如果对不上**，先别训练，按这个顺序排查：
1. `grade.py` 的 `NAME` 是否指向了正确的目录（默认值指向 Nemotron，是个陷阱）
2. 采样参数是否一致：temperature 0.7 / top_p 0.9 / max_tokens 31744
3. CompassVerifier 是否真的加载成功（它在 `grade.py` **import 时**就无条件加载到 GPU 0）
4. N 是否对：AIME/AMC/HMMT/BRUMO/CMIMC=32，MATH-500/Minerva/Olympiad=4

**已知的两个代码缺陷**（会轻微影响绝对值，建议先记录再决定是否修）：
- `grade.py:152` 传给 CompassVerifier 的 question 恒为空字符串 —— 判分模型只看到答案对比，
  看不到题目。`gen_vllm.py` 其实把 `prompt` 写进了 JSONL，可修复。
- `grade.py:162` 的 `avg_output_length` 恒为 0（`length_tokenizer` 从未赋值）。

这两个缺陷作者大概也带着跑出了论文数字，所以**先保持原样对齐**，对齐成功后再作为改进项。

---

### Phase 2 — 基线刻画（不训练，与 Phase 1 共用同一套脚本）

**目的**：测量两个底座模型**训练前**的原始水平。
这是 Table 3/5 里的 "Backbone" 行，**没有它就无法主张任何提升**。

**科学性**：这是实验设计里的 control group（对照组）。论文所有"提升"都是
相对于这两行算出来的，不是相对于零。

**做法**：同 Phase 1，只改 `gen_vllm.py:27` 的 `NAME` 为底座模型路径，重复两轮。

**判据（对齐目标）**：

| Benchmark | DeepSeek-R1-Distill | Nemotron |
|---|---|---|
| AIME24 | 29.90 | 58.75 |
| AIME25 | 22.40 | 48.44 |
| AMC23 | 63.82 | 90.55 |
| MATH-500 | 84.90 | 92.40 |
| Minerva | 34.65 | 26.93 |
| Olympiad-Bench | 45.95 | 71.70 |
| HMMT25 | 13.44 | 30.10 |
| BRUMO25 | 30.94 | 61.67 |
| CMIMC25 | 12.89 | 30.08 |
| **Avg** | **37.65** | **56.74** |

**Phase 1 + 2 完成后**，你就拥有了 Table 3 和 Table 5 的**首行和末行**，
中间的对比方法（DeepScaleR/ProRL-V2/QuestA/BroRL）是引用别人的公开数字，不需要复现。

---

### Phase 3 — E1：弱底座主实验（核心）★算力大头★

**目的**：验证**主张 ③** —— 最简单的配方在弱底座上能打赢复杂方法。

**科学性**：这是论文的主实验。它同时产出 Table 3、Table 4、Fig 1(a)、Fig 2 四份结果。

**做法**：`train/run_training.sh`，需要改的地方：

| 问题 | 处理 |
|---|---|
| `n_gpus_per_node=8, nnodes=4` + `RAY_ADDRESS="11.11.18.2:6379"` 指向已消失的集群 | 改 `n_gpus_per_node=4, nnodes=1`，本地起 ray |
| 硬编码路径 `/home/test/test06/hbx/...`、conda env `hbx_ck` | 换成当前路径/环境 |
| 硬编码明文 `WANDB_API_KEY`（已进 git） | 删除，改用 tensorboard；**该 key 建议去 wandb 后台吊销** |
| `max_actor_ckpt_to_keep: null` + `save_freq=50` → 88 个 ckpt ≈ 1.7 TB | **必须**设 `max_actor_ckpt_to_keep=3` |
| `data.val_batch_size=6312` 但 val 只有 100 题 | 无害（整批跑），是抄 POLARIS 的残留 |

**超参一个字都不能改** —— 论文 Table 2 已与脚本逐条核对一致：
`train_batch=256`、`mini_batch=64`、`lr=1e-6` 恒定、`n=8`、`temp=1.0`、
`clip [0.2, 0.28]`、`clip_ratio_c=10`、无 KL、无 entropy、16k 上下文。

**产出**：
- TensorBoard 曲线 = Fig 1(a) 与 Fig 2
- `trainer.test_freq=50` 自动在 AIME24/25/AMC23 上验证 → Fig 1(a) 的训练中曲线
- 终点的 checkpoint → 喂给 Phase 1 的评测脚本 → Table 3 末行

**判据**：
- 训练曲线应**平滑单调**上升，无崩溃、无平台期（这是论文的核心观察）
- 终点 Avg 应接近 **54.87%**
- AIME24 应从 ~29.90 升到 ~52.60

**⚠️ 一个论文内部的不一致，别追错目标**：
Fig 1(a) 标题写 "from 28% to 58% over 4,000 steps"，但 Table 3 报的是 `52.60`。
58% 很可能是曲线的峰值/平滑值，52.60 才是最终的 avg@32。**以 52.60 为准。**

**算力预期**：这是最大的开销。论文用 32×A800-80GB 跑了约 15 天
（≈480 A800-GPU-天）。在 4×H100 上粗估约 **1.5–2 个月**。
建议先跑 50 步 smoke test（`trainer.total_training_steps=50`）验证不 OOM。

---

### Phase 4 — 训练动力学分析（Fig 2，E1 的副产品，不额外训练）

**目的**：验证**主张 ⑤** —— 回答"为什么稳定"。

**科学性**：这一步把论文从"我结果好"提升到"我知道为什么好"。
它是**对 E1 的观测**，不是新实验 —— 但需要**主动去看**三条曲线，
并且要有明确的"病理判据"。

**做法**：从 E1 的 TensorBoard 里提取三条曲线。

**判据（论文报告的特征）**：

| 指标 | 论文观察 | 病理（若出现则不成立） |
|---|---|---|
| Policy Entropy | 在 1.0–1.6 间振荡，后期 1.2–1.4 | 系统性上升（探索崩塌）或下降（过早收敛） |
| Mean Reward | 从 ~-0.6 升到 ~+0.4，有噪声但趋势明确 | 长时间平台期 / 突然下跌 |
| Mean Response Length | 从 ~8000 token 自然压缩到 4000–5000，**无长度惩罚** | 持续膨胀（length explosion） |

**注意**：`math_dapo` verifier 返回的是 **+1.0 / -1.0**，所以论文的 -0.6 起点、
+0.4 终点是在 [-1,1] 区间内的，与代码一致。不要把 reward 曲线误读成 [0,1]。

---

### Phase 5 — E2：强底座迁移

**目的**：验证**主张 ④** —— 同一套超参能原封不动迁移。

**科学性**：这是论文最有力的一个论证。单模型调好不能说明什么，
**换一个起点、零调参仍然成立**，才排除了"针对某个模型碰巧调对了"的可能。

**做法**：**复制 E1 的完整命令，只改 `ACTOR_MODEL_PATH` 指向 OpenMath-Nemotron-1.5B。**
其余一字不改 —— 这正是实验本身的设计。

**产出**：Table 5、Table 6、Fig 1(b)。

**判据**：
- Avg 应接近 **64.32%**（QuestA 是 63.81%）
- 同样应观察到平滑曲线（Fig 1(b)）
- 训练 **3,440 步**

**算力**：约 1–1.5 个月（4×H100）。

---

### Phase 6 — E3/E4：消融实验

**目的**：验证**主张 ⑥** —— "标准技巧"在本文设定下**有害**。

**科学性**：这是论文的**反直觉主张**，也是最需要严格对照的部分。
两个消融都是**单变量**改动，其余与 E1 完全一致，训练 3,000+ 步。

**E3 — 加 overlong penalty**：
脚本里已有这三行，只需把 `enable` 改为 `True`：
```
+reward_model.reward_kwargs.overlong_buffer_cfg.enable=True
+reward_model.reward_kwargs.overlong_buffer_cfg.len=4096
+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
```
（`len=4096` 即"最后 4k token"惩罚，对应论文 §4.4 的描述。）

**E4 — 再加 robust verifier**：
在 E3 基础上，把验证器从 `verl/utils/reward_score/math_dapo.py`（DAPO 无 SymPy 字符串匹配）
换成 DeepScaleR 的 verifier。
**⚠️ 这部分代码不在本仓库** —— `math_dapo.py` 自带的是**基线** verifier，
DeepScaleR 的实现需要从 `agentica-project/deepscaler` 移植，这是额外的工程量。

**判据**：

| 实验 | AIME24 预期 | Entropy 预期 |
|---|---|---|
| E1（基线） | ~55% | 1.2–1.4（健康振荡） |
| E3（+长度惩罚） | ~50%，约 2000 步后分化 | **0.5–0.6（探索崩塌）** |
| E4（+长度惩罚+鲁棒验证器） | ~45% | 崩溃 |

**关键观察点是 entropy，不是分数** —— 论文的解释是长度惩罚
"collapses exploration"，所以 entropy 曲线是**因果证据**，分数只是结果。

---

## 3. 执行顺序与依赖关系

```
Phase 0 环境数据
    ↓
Phase 1 评测自检（对齐已发布模型）  ←── 守门人，不通过不许往下走
    ↓
Phase 2 基线刻画（两个底座模型）
    ↓
    ├──→ Phase 3  E1 弱底座 ──→ Phase 4  动力学分析（副产品）
    │                              ↓
    │                          Phase 5  E2 强底座（只换模型路径）
    │
    └──→ Phase 6  E3 / E4 消融（依赖 E1 作对照）
```

**Phase 1 是硬性门禁**：它是唯一一个"答案已知"的实验。
如果连已发布权重都评不出 54.87%，说明评测链路有问题，
此时继续训练只会产生无法解释的结果。

---

## 4. 算力与降级策略

**论文规模**：每个模型 32×A800-80GB × 约 15 天 ≈ 480 A800-GPU-天。

**本机实测**：4×H100 80GB（GPU0 已被其他进程占用 26GB），192 vCPU / 2TB RAM。

| 实验 | 论文步数 | 4×H100 粗估 |
|---|---|---|
| E1 | 4,380 | 1.5–2 个月 |
| E2 | 3,440 | 1–1.5 个月 |
| E3 | 3,000+ | ~1 个月 |
| E4 | 3,000+ | ~1 个月 |

（粗估依据：论文 token 预算 ≈ `4380 × 256 prompt × 8 rollout × ~5k tokens ≈ 4.5×10¹⁰` 生成 token；
H100 对此类 1.5B 模型的吞吐约为 A800 的 2 倍。误差 ±2 倍，但量级确定。）

### 建议的三档复现策略

**Tier 1 — 验证论证链（约 1 周，推荐先做）**
- Phase 1 + Phase 2（纯评测，几小时级）
- E1 跑到 **1,000 步**就停
- **为什么够**：论文 Fig 2 的三个关键特征（entropy 稳定在 1.2–1.4、
  长度从 8000 压到 4500、reward 单调上升）**都在前 1000 步内就已经出现**。
  Fig 2 原文即写 "naturally compresses to 4,000-5,000 tokens **by step 1,000**"。
  目标是**验证机制**，不是刷到 54.87。

**Tier 2 — 复现核心数字（约 2 个月）**
- E1 全量 4,380 步 → Table 3 末行 + Fig 1(a) + Fig 2

**Tier 3 — 完整复现（约 5–7 个月）**
- E1 + E2 + E3 + E4 全量 → 6 张表 + 3 张图全覆盖

**不推荐**做的：试图复现 Table 4/6 的算力数字。它们是估算，
复现它们需要先复现 ProRL-V2/BroRL/QuestA 并测量其真实过滤率 —— 那是另一个量级的工程。

---

## 5. 关键风险清单

| 风险 | 影响 | 应对 |
|---|---|---|
| **/volume/data 已写满** | 所有下载/checkpoint 失败 | 见第 6 节，需先解决 |
| 4 卡 vs 论文的 32 卡 | 吞吐与负载分布不同，超参未变但 batch 语义可能偏移 | 保持 `train_batch_size=256` 与 `n=8` 不变；仅调并行度 |
| rollout 显存 | 单步 2048 条序列 × 16k 上下文 | 已估算接近 80GB KV 上限；OOM 时只能调 `gpu_memory_utilization` |
| `save_freq=50` × 无保留上限 | 88 个 ckpt ≈ 1.7 TB | 必须设 `max_actor_ckpt_to_keep` |
| E4 需移植 DeepScaleR verifier | 额外工程量 | 可先只做 E3，E4 单独排期 |
| 论文 Fig 1(a) 的 58% vs Table 3 的 52.60 | 目标不一致 | 以 Table 3 的 **52.60** 为对齐基准 |

---

## 6. 当前阻塞项

1. **磁盘写满**：`/volume/data` 剩余 **60 MB**（368T 已用满）。实测写入 10GB 失败（0 字节）。
   下载与 checkpoint 都无法进行。
2. **HF 数据集下载的 401**：已定位为**调用参数错误**（漏了 `--repo-type dataset`），
   不是鉴权问题 —— 匿名下载实测正常。修正后重试即可。
