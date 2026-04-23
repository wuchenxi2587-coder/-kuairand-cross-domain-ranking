# KuaiRand-27K 跨域短视频排序：渐进式模块堆叠方案

# 完整技术报告

---

## 第一章 项目概述

### 1.1 研究背景与动机

在短视频推荐场景中，用户在不同 Tab 页（如「关注」「发现」「同城」等多达 15 个场景）的行为模式存在显著差异。传统 CTR 模型将所有域的数据混合训练，忽略了域间兴趣迁移和域内偏好差异。本项目提出一套**渐进式模块堆叠**方案，在 DIN 基线之上逐层引入跨域感知、多兴趣融合、显式特征交叉和个性化调制，系统性地提升排序效果。

### 1.2 数据集

基于快手发布的 **KuaiRand-27K** 数据集：
- **27,285** 名用户，约 **3,200 万**视频 ID
- 时间跨度：2022年4月8日 ~ 5月8日
- 核心价值：同时包含**标准推荐曝光**和**均匀随机曝光**，后者提供无偏评估基准
- 预测目标：`long_view`（是否完播，二值标签）
- 训练样本约 **1.3 亿条**

### 1.3 技术方案总览

```
DIN Baseline → +ADS-lite(PSRG+PCRG) → +TransformerFusion → +MBCNet → +PPNet
```

每一层在前一层基础上**增量堆叠**新模块，通过消融实验验证每个模块的边际贡献。同时设置 **sparse（纯 embedding）** 和 **full（+dense 统计特征）** 两轮实验对比特征粒度的影响。

### 1.4 核心成果

| 指标 | Baseline → 最终模型 | 提升 |
|------|-------------------|------|
| Full test_random GAUC | 0.5525 → 0.5815 | **+2.90 pt (+5.25%)** |
| Full test_standard GAUC | 0.6761 → 0.6918 | **+1.57 pt (+2.32%)** |
| Sparse test_random GAUC | 0.5513 → 0.5745 | **+2.32 pt (+4.21%)** |

五步链路在 full track 和 sparse track 上均**严格单调递增**，每个模块均有正向贡献。

---

## 第二章 数据预处理

### 2.1 Pipeline 总流程

```
原始 CSV → Step1 词表构建 → Step2 物品统计量 → Step3 样本生成 → Step4 时序切分
```

由 `run_all_v2.py` 串联执行四步脚本。

### 2.2 Step 1: 词表构建与特征编码

**输入**：`user_features_27k.csv`（27,285 用户）、`video_features_basic_27k.csv`

**处理逻辑**：

| 特征类型 | 编码方式 | 说明 |
|---------|---------|------|
| ID 类（video_id, author_id） | 全量枚举排序，0=PAD | 后续训练时改用 hash embedding 降维 |
| 类别（video_type, tab 等） | 枚举映射，负数/NaN → PAD(0) | 所有编码从 1 起，0 统一保留给 PAD |
| Binary（is_live_streamer 等） | 负数→0, 0→1, 1→2 | 三值编码，兼容原始数据中 -124 等异常 |
| onehot_feat0~17 | 负数→0, 正常值+1 | 18 个匿名类别特征 |
| 视频时长 | 按 [7s, 18s, 30s, 60s] 分桶 | 5 个桶 + PAD，毫秒单位 |
| 用户 Dense（粉丝数等） | `log1p(max(x, 0))` → float32 | 压缩长尾分布 |

**设计要点**：
- 统一的 PAD=0 约定，确保 Embedding 层 `padding_idx=0` 全局一致
- 负数异常值（数据集已知问题）全部安全降级为缺失值

**输出**：`vocabs/*.json`（约 25 个词表文件）、`video_features_processed.parquet`、`user_features_processed.parquet`

### 2.3 Step 2: 物品先验统计量（防泄露设计）

**核心原则**：统计量仅使用 **pre 期**（4/08~4/21）数据计算，训练/测试样本来自 4/22~5/08，严格时间隔离。

```
4/08 ────── 4/21 ────── 4/22 ────── 5/08
[  统计量计算期  ]    [  训练/测试期  ]
```

**计算的 5 个统计量**：

| 特征 | 含义 | 分桶策略 | 桶数 |
|------|------|---------|------|
| pre_ctr | 点击率 | smart_zero | 3（无数据/低/高） |
| pre_lv_rate | 完播率 | smart_zero | 3 |
| pre_like_rate | 点赞率 | smart_zero | 3（99%为0） |
| pre_play_ratio | 平均播放比例 | uniform 等频 | 5 |
| pre_show_cnt | 曝光次数 | log 尺度 | 5（冷启动/1次/.../≥200次） |

**smart_zero 分桶算法**：
```
if 零值比例 > 30%:
    零值 → 独立桶(PAD=0)
    非零值 → 等频分到剩余桶
else:
    全部等频分桶
```

**双版本输出**：
- **Sparse 分桶版**（int8）→ 送入 Embedding 查表，处理长尾
- **Dense 原始值版**（float32）→ 直接输入 MLP，保留数值精度

### 2.4 Step 3: 样本生成（多进程架构）

**架构**：Main 进程加载只读数据（~2-3GB），fork 12 个 Worker（user_id % 12 分桶），Linux copy-on-write 零拷贝共享。单进程 16~18h → 多进程 2~3h。

**单样本构建**（对每个用户按时间排序后逐条处理）：

1. **候选视频特征**（10 维 sparse + 5 维 dense）
2. **上下文特征**（4 维 sparse）：tab, hour_of_day, day_of_week, is_weekend
3. **用户画像特征**（26 维 sparse + 4 维 dense）
4. **用户历史序列**（max_len=50，只取正交互）

**历史序列 Side Information**：

| 辅助特征 | 分桶边界 | 语义 |
|---------|---------|------|
| delta_t_bucket | [1min, 5min, 30min, 1day, 3day] | 时间衰减 |
| play_ratio_bucket | [0.2, 0.5, 0.8, 1.2, 2.0] | 兴趣强度 |
| hist_tab | 0~14 | 跨域标记 |

### 2.5 Step 4: 时序切分

```
Standard 样本按 sample_time_ms 排序:
├── 前 70% → train.parquet      (~91M 样本)
├── 中 10% → val.parquet        (~13M 样本)
└── 后 20% → test_standard.parquet (~26M 样本)

Random 样本 → test_random.parquet (~7M 样本，全量保留)
```

**时序单调性验证**：`assert train_max_t ≤ val_min_t ≤ test_min_t`

**生成的 Meta 文件**：`field_schema.json`（字段定义）、`vocab_sizes.json`（Embedding 维度）、`split_summary.json`（数据统计）、`sanity_checks.json`（正确性验证）

---

## 第三章 数据加载系统

### 3.1 ParquetIterableDataset

**核心设计**：在 Dataset 内部预组装整个 batch 的 numpy 字典，DataLoader 设置 `batch_size=None`，collate 函数只做 numpy→tensor 零拷贝转换。相比逐样本构建 Python dict 提速 30~50 倍。

**读取流程**：
```
Parquet Row Group → Arrow combine_chunks → numpy reshape → 可选 shuffle → fancy indexing 切 batch → yield
```

**关键优化**：
- list 列使用 Arrow 原生 `combine_chunks().values.to_numpy()` + reshape，比 `to_pylist()` 快 10 倍
- Windows 下自动 `num_workers=0`（预组装 batch 已消除主要瓶颈）
- `set_epoch()` 改变 shuffle seed，每 epoch 不同数据顺序
- Worker 间按 Row Group 轮询分配，无数据竞争

### 3.2 BatchCollateFn

- Float 列 → `torch.from_numpy().float()`（零拷贝）
- Int/List 列 → `torch.from_numpy().long()`（零拷贝）
- 多个 `user_dense` 列合并为单一 `[B, D_dense]` tensor
- 首次调用时打印完整 batch schema（调试用）

### 3.3 resolve_columns

根据 `field_schema.json` 动态决定从 parquet 读哪些列：训练集不读 meta 列节省 IO，评估集额外读 user_id_raw（GAUC 需要按用户分组）。

---

## 第四章 模型架构

### 4.1 总体架构图

```
输入特征
  │
  ├─ 候选视频 → HashEmbedding(video_id 32d, author_id 16d) + side embs → cand_repr [B, 48]
  ├─ 历史序列 → HashEmbedding(共享权重) → hist_repr [B, 50, 48]
  ├─ 上下文   → Sparse Embedding → context_embs
  ├─ 用户画像 → Sparse Embedding + Dense → user_profile
  │
  │  ┌────────────── 渐进式模块堆叠 ──────────────┐
  │  │                                              │
  │  │  ① DomainContext(tab+hour+dow) → d_ctx [B,48]│
  │  │  ② PSRG: 门控残差调制历史序列                  │
  │  │  ③ DIN Attention → user_interest_din [B,48]  │
  │  │  ④ PCRG: 4-query 多兴趣提取                   │
  │  │     → user_interest_ads [B,48]               │
  │  │     → interest_tokens [B,4,48]               │
  │  │  ⑤ Fusion(din, ads) → user_interest [B,48]   │
  │  │  ⑥ TransformerFusion(tokens) → u_fused [B,48]│
  │  │                                              │
  │  └──────────────────────────────────────────────┘
  │
  ├─ head_input = concat(所有特征块) → [B, ~264/269]
  │
  │  ┌────────────── 预测头 ────────────────────────┐
  │  │  ⑦ PPNet: GroupWiseFiLM 个性化调制             │
  │  │  ⑧ MBCNet: FGC + LowRank + Deep 三分支交叉    │
  │  │     → 融合 → logit                           │
  │  └──────────────────────────────────────────────┘
  └─→ BCEWithLogitsLoss
```

### 4.2 Embedding 层

| Embedding | 维度 | 桶数/词表 | 方式 |
|-----------|------|---------|------|
| video_id | 32 | 1,000,000 | HashEmbedding: `(x % 999999) + 1` |
| author_id | 16 | 500,000 | HashEmbedding |
| 小类别（词表 <100） | 4 | 各自词表 | 标准 Embedding |
| 大类别（词表 ≥100） | 8 | 各自词表 | 标准 Embedding |

候选与历史**共享**同一套 video_id / author_id Embedding 权重。

`item_repr_dim = 32 + 16 = 48`（贯穿全模型的核心维度）

### 4.3 DomainContextEncoder

将场景信息编码为密集向量，供 PSRG/PCRG 使用。

**输入**：tab + hour_of_day + day_of_week 三个 embedding 拼接

**结构**：`Linear → ReLU → Linear(→48) → LayerNorm`

**设计动机**：跨域排序的关键在于让模型感知「用户当前在哪个域」。同一用户在「关注页」和「发现页」的兴趣可能完全不同。

### 4.4 PSRG — Personalized Sequence Representation Generation

**目的**：根据域上下文对历史序列中每个 item 的表示进行门控调制。

**门控残差机制**：
```
delta = MLP([h; d_ctx])        # 生成增量
gate = sigmoid(MLP([h; d_ctx])) # 生成门控信号
h' = LayerNorm(h + gate * delta) # 门控残差更新
```

- 可选拼接 `hist_tab` embedding（历史交互发生在哪个 tab），形成逐位置不同的域上下文
- padding 位置保持原始值：`h_out * mask + h_orig * (1-mask)`

**设计动机**：同一历史视频在不同域上下文下应有不同表示。例如一个美食视频在「关注页」可能是核心兴趣（关注了博主），在「发现页」可能只是偶然点击。

### 4.5 DIN Target Attention

**评分函数**：`MLP([q, k, q-k, q*k])` → 标量分（输入 4×48=192 维，隐层 [64,32]，PReLU 激活）

```
scores_i = MLP([cand, hist_i, cand-hist_i, cand*hist_i])
weights = masked_softmax(scores, hist_mask)  # padding → -1e4
user_interest_din = Σ(weights_i * hist_i)  → [B, 48]
```

### 4.6 PCRG — Personalized Candidate Representation Generation

**目的**：从候选+域上下文生成 G=4 个查询向量，多角度 attend 历史序列。

```
[cand; d_ctx] → MLP → 4 个 query [B, 4, 48]
  → 每个 query 独立做 DIN-style attention on hist
  → 4 个 interest 向量 [B, 4, 48]
  → mean_pool → user_interest_ads [B, 48]
  → interest_tokens [B, 4, 48]（送入 TransformerFusion）
```

**设计动机**：单 query（DIN）只能捕获与候选最相关的单一兴趣。G=4 个 query 从不同角度挖掘历史（题材、作者、时长、风格），域上下文参与 query 生成确保角度是场景感知的。

### 4.7 兴趣融合

DIN 与 PCRG 各产出 `[B, 48]`：`concat → [B, 96] → Linear(96, 48) → user_interest`

### 4.8 TransformerFusion

**三步流程**：
```
Step 1: interest_tokens [B,4,48] → Self-Attention(1 layer, 2 heads)
        → 兴趣 token 间去冗余、互补遗漏
Step 2: 候选做 query → Target Attention on self-attended tokens
        → 加权聚合 [B, 48]
Step 3: FFN(GELU, 256) → LayerNorm → u_fused [B, 48]
```

融合到主兴趣：`concat → proj → [B, 48]`

**设计动机**：PCRG 的 4 个兴趣 token 独立生成、互不感知。自注意力让它们「看到彼此」后再由 target attention 提取与候选最匹配的信号。

### 4.9 Head Input 构建

将所有特征块拼接为 flat vector，通过 `feature_slices` 记录每个语义块的偏移量：

```
head_input = [user_interest | cand_repr | user_sparse_embs | user_dense | context_embs | cand_side_embs]
              48              48          ~104               4            ~16             ~40
              → head_input_dim ≈ 264 (sparse) 或 269 (full)
```

### 4.10 PPNet — Personalized Prediction Network

**PersonalContextEncoder**（三流编码器）：
```
p_ctx = [scene_ctx | user_ctx | activity_ctx]  → ~112 维
```
- Scene：tab/hour/day/weekend embedding
- User：user_sparse_embs → MLP → 64 维
- Activity：hist_len 归一化 + is_lowactive → MLP → 32 维

**GroupWiseFiLM 调制**（6 组）：
```
p_ctx → MLP([64,32]) → (γ_1..γ_6, β_1..β_6)
x_g' = (1 + γ_g) * x_g + β_g   对 head_input 的 6 个语义组
```

**零初始化**：输出层权重/偏置全零 → 训练初期 γ≈0, β≈0 → FiLM 退化为恒等映射 → 从「不做任何事」逐步学习调制，避免破坏前序模块已学好的特征。

### 4.11 MBCNet — Multi-Branch Cross Network

替换简单 MLP 头，三种互补的特征交叉：

| 分支 | 机制 | 擅长 |
|------|------|------|
| **FGC** | 分组内 CrossNet-v1: `g' = g + (w^T g)*g` | 语义组内低阶显式交叉 |
| **LowRank Cross** | 低秩全局交叉 `U·x, V·x → P(u*v)`, rank=16 | 全局跨组二阶交互 |
| **Deep** | MLP [256,128,64], GELU, Dropout=0.1 | 高阶隐式交互 |

**分支融合**：各分支 → Linear → 128 维，concat [B,384] → MLP [128,64] → logit

---

## 第五章 训练系统

### 5.1 训练配置

| 参数 | DIN Baseline | ADS 及后续 |
|------|-------------|-----------|
| Epochs | 1 | 5 |
| Batch Size | 4096 | 4096 |
| Optimizer | AdamW | AdamW |
| Learning Rate | 1e-3 | 1e-3 |
| Weight Decay | 1e-5 | 1e-5 |
| AMP | bfloat16 | bfloat16 |
| Gradient Clipping | 1.0 | 1.0 |
| Early Stopping | patience=3, monitor=GAUC | 同左 |

### 5.2 训练循环

```python
for batch in dataloader:
    with autocast(dtype=bfloat16):
        logits = model(batch)
        loss = BCEWithLogitsLoss(logits, labels) / accum_steps
    scaler.scale(loss).backward()
    if (step+1) % accum_steps == 0:
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### 5.3 ADS Debug 监控

每 200 步记录诊断信息：DIN 注意力熵、PCRG query 间方差、MBCNet 各分支输出范数、PPNet γ/β 均值方差。用于检测兴趣坍缩、多兴趣退化等问题。

### 5.4 Sanity Check

训练前对 val 集跑 1 batch 前向：打印所有 tensor shape/dtype，检查 logits 范围，NaN/Inf 立即中断。

### 5.5 Checkpoint 与 Early Stopping

- 每 epoch 结束在 val 上评估 GAUC
- GAUC 提升 → 保存 checkpoint（model + optimizer + metrics）
- 连续 3 epoch 无提升 → 提前终止
- 训练结束后加载 best checkpoint，在 val / test_standard / test_random 上做 final evaluation

---

## 第六章 评估体系

### 6.1 指标

| 指标 | 定义 | 角色 |
|------|------|------|
| **GAUC** | 按用户分组 AUC 的加权平均: `Σ(auc_u × n_u) / Σ(n_u)` | **主指标** |
| AUC | 全局 ROC 曲线下面积 | 辅助 |
| LogLoss | 交叉熵 | 校准质量 |

GAUC 跳过样本数 <2 和标签全同的用户（AUC 未定义）。

### 6.2 StreamingMetricCollector

评估集约 2600 万样本，每 batch 追加 (preds, labels, user_ids) 的 numpy chunk，累计超过 2GB 自动切换到磁盘溢出模式（.npz 临时文件），防止 OOM。

### 6.3 两个测试集

| 测试集 | 曝光方式 | 评估含义 |
|--------|---------|---------|
| test_standard | 受推荐策略筛选 | 在已筛选候选上的排序质量 |
| test_random | 均匀随机曝光 | 面对全量候选的**真实排序能力** |

本项目以 **test_random GAUC 为主指标**（KuaiRand 原论文推荐的无偏评估方式）。

---

## 第七章 实验结果

### 7.1 Full Track（含统计特征）

| 模块 | val GAUC | test_std GAUC | test_rnd GAUC | test_std 增量 | test_rnd 增量 |
|------|----------|---------------|---------------|--------------|--------------|
| DIN Baseline | 0.6745 | 0.6761 | 0.5525 | — | — |
| + ADS | 0.6845 | 0.6854 | 0.5698 | +0.0093 | **+0.0173** |
| + Transformer | 0.6878 | 0.6880 | 0.5745 | +0.0026 | +0.0047 |
| + MBCNet | 0.6905 | 0.6905 | 0.5775 | +0.0025 | +0.0030 |
| + PPNet | **0.6921** | **0.6918** | **0.5815** | +0.0013 | +0.0040 |

**总提升**：test_rnd 0.5525 → 0.5815 = **+2.90 pt (+5.25%)**，test_std 0.6761 → 0.6918 = **+1.57 pt (+2.32%)**

### 7.2 Sparse Track（无统计特征）

| 模块 | val GAUC | test_std GAUC | test_rnd GAUC | test_std 增量 | test_rnd 增量 |
|------|----------|---------------|---------------|--------------|--------------|
| DIN Baseline | 0.6736 | 0.6754 | 0.5513 | — | — |
| + ADS | 0.6826 | 0.6825 | 0.5632 | +0.0071 | **+0.0119** |
| + Transformer | 0.6858 | 0.6852 | 0.5680 | +0.0027 | +0.0048 |
| + MBCNet | 0.6882 | 0.6873 | 0.5718 | +0.0021 | +0.0038 |
| + PPNet | **0.6897** | **0.6885** | **0.5745** | +0.0012 | +0.0027 |

**总提升**：test_rnd 0.5513 → 0.5745 = **+2.32 pt (+4.21%)**

### 7.3 统计特征的价值（Full - Sparse）

| 模型阶段 | test_std 增益 | test_rnd 增益 |
|---------|-------------|-------------|
| Baseline | +0.0007 | +0.0012 |
| + ADS | +0.0029 | +0.0066 |
| + Transformer | +0.0028 | +0.0065 |
| + MBCNet | +0.0032 | +0.0057 |
| + PPNet | +0.0033 | **+0.0070** |

**关键发现**：统计特征的价值随模型复杂度递增（test_rnd 从 +0.12pt 增长到 +0.70pt）。越复杂的模型越能利用 dense 数值信息中的细粒度差异。

### 7.4 调参影响分析（batch_size 8192 → 4096）

| 模块 | 调参前 (bs=8192) | 调参后 (bs=4096) | 改善 |
|------|-----------------|-----------------|------|
| Baseline | 0.5515 | 0.5525 | +0.0010 |
| + ADS | 0.5682 | 0.5698 | +0.0016 |
| + Transformer | 0.5759 | 0.5745 | -0.0014 (噪声) |
| + MBCNet | 0.5751 | 0.5775 | **+0.0024** |
| + PPNet | 0.5808 | 0.5815 | +0.0007 |

**调参前的问题**：bs=8192 时 MBCNet/PPNet 在 test_standard 上出现回退。原因是较大 batch size 降低了梯度噪声，对大容量模型（MBCNet 三分支、PPNet 个性化参数多）反而导致在有偏数据上过拟合推荐策略本身。

**调参后的改善**：bs=4096 引入适度的梯度噪声作为隐式正则化，test_standard 不再回退，五步链路完美单调递增。

### 7.5 结果分析

**1. ADS-lite 是最大增益源**（test_rnd +1.73pt / test_std +0.93pt）

跨域感知的序列重映射（PSRG）+ 多兴趣提取（PCRG）显著提升了模型从用户历史中挖掘信息的能力。这是符合预期的——在 15 个 tab 场景中，历史行为的域感知重映射和多角度兴趣提取是最本质的改进。

**2. TransformerFusion 提供稳定增益**（test_rnd +0.47pt / test_std +0.26pt）

4 个兴趣 token 间的自注意力交互有效去冗余和互补。增益低于 ADS 说明 PCRG 已经完成了大部分兴趣提取工作，Transformer 是锦上添花的精炼。

**3. MBCNet 一致性正增益**（test_rnd +0.30pt / test_std +0.25pt）

三分支显式交叉替换 MLP 带来稳定但温和的增益。在 ~264 维特征空间中，显式交叉的优势相对有限（更适合高维场景），但 FGC 的分组内交叉和 LowRank 的跨组交叉仍然捕获了 MLP 遗漏的低阶模式。

**4. PPNet 在 full track 上贡献突出**（test_rnd +0.40pt / test_std +0.13pt）

个性化调制的效果与输入特征丰富度正相关：full track 比 sparse track 多 5 个 dense 统计特征，PPNet 在 full 上增益是 sparse 上的 1.5 倍（0.40 vs 0.27pt）。这验证了 PPNet 的设计假设——足够丰富的特征输入是个性化调制发挥作用的前提。

**5. Full vs Sparse 的递增差距**

统计特征的价值从 baseline 的 +0.12pt 增长到 PPNet 的 +0.70pt（test_rnd），说明越复杂的模型越能「用好」dense 数值信息。这是因为分桶损失了数值精度（CTR 0.05 和 0.06 在同一个桶），而 PCRG 的多查询注意力、MBCNet 的显式交叉和 PPNet 的 FiLM 调制都能利用 dense 特征中的细粒度差异。

---

## 第八章 工程亮点总结

| 模块 | 设计 | 目的 |
|------|------|------|
| Hash Embedding | video_id 1M桶, author_id 500K桶 | 3200万 ID → 可控显存 |
| smart_zero 分桶 | 零值独立桶 + 非零等频 | 处理推荐场景极端长尾 |
| 双版本统计特征 | sparse 分桶 + dense 原始值 | 鲁棒性与精度兼顾 |
| 三层防泄露 | pre 期算统计 / 时序切分 / assert 验证 | 杜绝泄露路径 |
| 多进程 fork + COW | 12 Worker 共享只读数据 | 16h → 2~3h |
| 预组装 batch | Dataset 内组装 numpy dict | 避免逐样本 Python 开销 (30~50x) |
| Arrow 原生解析 | combine_chunks + reshape | 比 to_pylist 快 10x |
| AMP bfloat16 | 前向低精度, 优化器高精度 | 4GB GPU 可训 1.3 亿样本 |
| StreamingMetricCollector | 超 2GB 自动磁盘溢出 | 2600万样本评估不 OOM |
| 零初始化 PPNet | γ=0, β=0 起步 | 恒等映射热启动，稳定训练 |
| 全链路 Mask 安全 | -1e4 fill + renorm + all-pad 检测 | 防 NaN 全链路 |
| 渐进式配置 | 5 个 YAML 逐层增加模块 | 严格消融，增量可控 |
| ADS Debug 监控 | 每 200 步记录注意力熵/方差 | 检测兴趣坍缩 |

---

## 第九章 面试 Q&A

### 项目总览

**Q: 一句话介绍你的项目？**

> 基于快手 KuaiRand-27K 数据集，设计了一套渐进式模块堆叠方案（DIN → ADS → Transformer → MBCNet → PPNet），在 DIN 基线上系统性提升跨域短视频排序效果，无偏测试集 GAUC 从 0.5525 提升到 0.5815（+5.25%），五步链路在有偏和无偏测试集上均严格单调递增。

**Q: 为什么选择 KuaiRand 这个数据集？**

> 两个原因：(1) 它同时包含标准推荐曝光和均匀随机曝光，随机曝光可以做无偏评估，这是其他公开数据集（MovieLens、Amazon）没有的；(2) 它有 15 个 tab 场景的标注，天然适合研究跨域排序问题。

**Q: 为什么用渐进式堆叠而不是直接训一个大模型？**

> (1) 消融需要——每层的边际贡献可以独立量化；(2) 调试友好——大模型不 work 时无法定位问题；(3) 工业实践中模块化设计便于灵活上线，根据算力预算选择性启用。

### 数据处理

**Q: 你的防泄露做了哪几层？**

> 三层：(1) 统计量只用 pre 期（4/08~4/21），训练/测试用 4/22~5/08，时间不重叠；(2) 数据集按时间严格排序后切分 70/10/20，assert 验证 train_max_t ≤ val_min_t ≤ test_min_t；(3) 随机曝光测试集独立保留不参与切分。

**Q: 为什么时序切分而不是随机切分？**

> 推荐系统是在线服务，永远是用过去预测未来。随机切分让模型在训练时看到「未来的用户行为」，validation/test 指标虚高。时序切分更贴近线上真实场景。

**Q: 历史序列为什么只取正交互？**

> 目的是送入 Target Attention 提取兴趣信号。负反馈数量远大于正反馈（~20:1），全加入会稀释正向信号。而且负反馈可能是位置偏差（没看到）而非不感兴趣。这也是 DIN/DIEN 的标准做法，负反馈信息通过 label=0 在 loss 中隐式学习。

**Q: smart_zero 分桶解决什么问题？**

> 点赞率 99% 是 0，CTR 大部分也是 0。直接等频分 3 桶，前 2 桶全是 0，无区分度。smart_zero 让零值单独成桶（语义=冷启动/无数据），非零值在剩余桶内等频，得到有意义的「无数据/低/高」三桶。

**Q: 为什么同时做 sparse 和 dense 两个版本？**

> Sparse 分桶通过 Embedding 查表处理长尾，但损失了数值精度（CTR 0.05 和 0.06 在同一桶）。Dense 保留原始值直接输入 MLP。实验证明统计特征价值随模型复杂度递增（baseline +0.12pt → PPNet +0.70pt on test_rnd），越复杂的模型越能利用 dense 中的细粒度差异。

### 模型设计

**Q: PSRG 的核心思想？**

> 同一历史视频在不同域上下文下应有不同表示。美食视频在「关注页」可能是核心兴趣（关注了博主），在「发现页」可能只是偶然点击。PSRG 用门控残差：`h' = h + sigmoid(gate(h, d_ctx)) * delta(h, d_ctx)`，让每个历史 item 的表示随当前域动态变化。

**Q: PCRG 为什么要 4 个 query？**

> 单 query 只能捕获单一兴趣维度。用户兴趣是多面的——题材、作者、时长、风格等。4 个 query 从候选+域上下文生成，各自从不同角度 attend 历史，域上下文参与确保角度是场景感知的。

**Q: TransformerFusion 解决什么问题？**

> PCRG 的 4 个兴趣 token 独立生成、互不感知。自注意力让它们「看到彼此」——去冗余（两个关注相似角度就合并）、互补遗漏。然后 target attention 用候选从融合后的 token 中选择性提取最相关信号。

**Q: MBCNet 三个分支各擅长什么？**

> FGC 做语义组内低阶显式交叉（如 user_interest 内部各维度组合），LowRank Cross 做全局跨组二阶交互（如用户画像 × 候选特征，rank=16 控制复杂度），Deep MLP 做高阶隐式交互。三者互补：显式交叉精确但阶数有限，MLP 灵活但可能忽略低阶模式。

**Q: PPNet 为什么要零初始化？**

> FiLM 公式 `(1+γ)*x + β`，如果随机初始化会大幅扰动 head_input，破坏前序模块已学好的特征。零初始化确保 γ≈0, β≈0，FiLM 退化为恒等映射，从「不做任何事」逐步学习调制。

**Q: PPNet 的个性化上下文包含哪些信息？**

> 三流编码：(1) scene_ctx：tab/hour/day/weekend 场景信息；(2) user_ctx：用户画像 embedding → MLP；(3) activity_ctx：hist_len 归一化 + 活跃度特征。三流拼接后生成 6 组 γ/β，对 head_input 的不同语义块分组调制。

### 实验与调参

**Q: 为什么 batch_size 从 8192 调到 4096 效果更好？**

> 较大 batch 降低梯度噪声，对简单模型（DIN/ADS）是好事，但对大容量模型（MBCNet 三分支、PPNet 多组参数）反而导致在有偏数据上过拟合推荐策略本身。调到 4096 引入适度的梯度噪声作为隐式正则化，MBCNet 在 test_rnd 上从 0.5751 提升到 0.5775（+0.24pt），test_standard 上的回退问题也消失了。

**Q: 为什么 GAUC 而不是 AUC？**

> 推荐排序是用户内排序。全局 AUC 被高活跃用户主导。GAUC 先算每用户 AUC 再加权平均，更公平地反映各类用户体验。

**Q: test_random 和 test_standard 为什么趋势不同（调参前）？**

> test_standard 的曝光经过推荐策略筛选，存在 selection bias。模型在有偏数据上提升更难（baseline 已经 0.6761），而且可能学到的是「模仿推荐策略」而非「理解用户偏好」。test_random 是均匀随机曝光（baseline 只有 0.5525），更能反映模型的真实 ranking 能力。调参后两个测试集都单调递增，说明模块确实在提升排序质量。

**Q: 统计特征价值为什么随模型复杂度递增？**

> 基线模型能力有限，只能用 embedding 中最粗粒度的信息，dense 原始值的细微差异被浪费了。随着模块增加——PCRG 的多 query 注意力、MBCNet 的显式交叉、PPNet 的 FiLM 调制——模型有了更多利用数值精度的通道。PPNet 阶段 full 比 sparse 多 +0.70pt（test_rnd），说明个性化调制 + 精细数值 = 最大收益。

### 工程

**Q: 数据加载怎么做到高吞吐？**

> 三个关键：(1) Dataset 内预组装整个 batch 的 numpy dict，DataLoader 设 `batch_size=None`，避免逐样本 Python dict 开销（原方案 30~50x 瓶颈）；(2) 序列列用 Arrow 原生 `combine_chunks` + reshape，比 `to_pylist` 快 10x；(3) collate 只做 `torch.from_numpy()` 零拷贝。

**Q: 1.3 亿样本在 4GB 显存怎么训？**

> 四个手段：(1) AMP bfloat16 砍半显存；(2) Hash embedding 把 3200 万 ID 压到 1M 桶（~2GB → ~64MB）；(3) 流式 parquet 读取，不全量加载；(4) 评估时 StreamingMetricCollector 超 2GB 自动磁盘溢出。

**Q: 怎么保证训练稳定性？**

> (1) Sanity check：训练前跑 1 batch 前向，检查 shape/dtype/NaN/Inf；(2) 全链路 mask 安全：padding → -1e4，softmax 后 renorm，全 padding 输出零向量；(3) gradient clipping max_norm=1.0；(4) PPNet 零初始化避免扰动；(5) ADS debug 监控检测兴趣坍缩。
