# KuaiRand 跨域多场景精排推荐

基于 KuaiRand-27K 数据集的多场景精排实验仓库。在 DIN baseline 上，逐步叠加 ADS 跨域多兴趣建模、Transformer 兴趣融合、MBCNet 多分支交叉网络、PPNet 个性化调制，形成完整的模型升级链路。

## 核心贡献

- **任务**：快手短视频精排，处理多 tab / 多场景下的跨域信息共享和用户多兴趣建模
- **标签**：`label_long_view`（完播预测），输入包含 50 长度行为序列、候选 item、场景特征、用户画像特征
- **升级路线**：`DIN → ADS (PSRG+PCRG) → TransformerFusion → MBCNet → PPNet`
- **最终结果**：test_random GAUC 从 0.5525 提升至 0.5815（**+5.25%**），五步链路在 full track 和 sparse track 均单调递增

## 实验结果

### Full Track（含物品统计特征）

| 阶段 | 核心改动 | test_std GAUC | test_rnd GAUC | test_rnd 增量 |
|------|----------|--------------|--------------|--------------|
| DIN baseline | 单 query DIN + HashEmbedding + 50 长度历史 | 0.6761 | 0.5525 | — |
| +ADS | PSRG 动态历史重映射 + PCRG 多 query 兴趣建模 | 0.6854 | 0.5698 | +0.0173 |
| +Transformer | Self-Attention + Target-Attention DNN + FFN | 0.6880 | 0.5745 | +0.0047 |
| +MBCNet | 分组交叉 + 低秩交叉 + Deep 分支 | 0.6905 | 0.5775 | +0.0030 |
| +PPNet | 场景 / 用户 / 活跃度条件化 FiLM 调制 | 0.6918 | 0.5815 | +0.0040 |

### Sparse Track（仅 embedding 特征）

| 阶段 | test_std GAUC | test_rnd GAUC | test_rnd 增量 |
|------|--------------|--------------|--------------|
| DIN baseline | 0.6754 | 0.5513 | — |
| +ADS | 0.6825 | 0.5632 | +0.0119 |
| +Transformer | 0.6852 | 0.5680 | +0.0048 |
| +MBCNet | 0.6873 | 0.5718 | +0.0038 |
| +PPNet | 0.6885 | 0.5745 | +0.0027 |

> test_random 基于随机曝光日志，无分发偏置，更能反映模型真实排序能力。

## 项目概览

| 维度 | 说明 |
|------|------|
| 用户规模 | 27,285 个用户 |
| 行为规模 | standard log 约 3.2 亿条曝光，random log 约 118 万条 |
| 训练样本 | train 1.3 亿 / val 1860 万 / test_standard 3720 万 / test_random 118 万 |
| 序列长度 | `max_hist_len=50`，正反馈筛选（long_view=1 或 is_like=1）|
| 候选空间 | `video_id` vocab 3200 万，`author_id` vocab 880 万，默认走 HashEmbedding |
| 场景信息 | tab、hour_of_day、day_of_week、is_weekend |

## 方法演进

### 0. DIN Baseline

标准 DIN attention，视频和作者 embedding 在历史序列与候选侧共享。用户 sparse/dense 特征、场景特征和候选特征一起进 DNN head。

- 配置：`configs/train_din_mem16gb.yaml`
- 模型：`src/models/din.py`
- 训练入口：`src/main_train_din.py`

### 1. ADS：PSRG + PCRG

解决单 query 对整段历史做一次聚合的局限性。`DomainContextEncoder` 编码场景信息为 `d_ctx`，`PSRGLite` 做域条件历史重映射，`PCRGLite` 生成多个 query 做多兴趣 attention。

- 配置：`configs/train_din_psrg_pcrg_mem16gb.yaml`
- 代码：`src/models/modules/domain_context.py`、`psrg.py`、`pcrg.py`
- 测试：`tests/test_din_psrg_pcrg_shapes.py`

### 2. Transformer 兴趣融合

PCRG 多兴趣 token 之间默认彼此独立。加入 Self-Attention 做兴趣内部上下文建模，`TargetAttentionDNN` 强化候选与兴趣的相关性，额外 FFN 提取高阶兴趣表征。

- 配置：`configs/train_din_psrg_pcrg_transformer.yaml`
- 代码：`src/models/modules/transformer_fusion.py`、`target_attention_dnn.py`
- 测试：`src/tests/test_transformer_fusion_shapes.py`

### 3. MBCNet

替换 concat → MLP 的 head，引入 FGC 分支（特征分组交叉）、Low-rank Cross 分支（低秩显式交叉）、Deep 分支（隐式高阶组合），按语义分组切分特征块。

- 配置：`configs/train_din_psrg_pcrg_transformer_mbcnet.yaml`
- 代码：`src/models/modules/mbcnet.py`、`feature_slices.py`
- 测试：`src/tests/test_mbcnet_shapes.py`

### 4. PPNet

对 head 输入做 group-wise FiLM 调制，`PersonalContextEncoder` 将场景时间、用户 dense、`hist_len`、活跃度代理等编码为 `p_ctx`，实现轻量条件化个性化。

- 配置：`configs/train_din_psrg_pcrg_transformer_mbcnet_ppnet.yaml`
- 代码：`src/models/modules/personal_context.py`、`ppnet.py`
- 测试：`src/tests/test_ppnet_shapes.py`

## 仓库结构

```text
.
├── configs/                          # 训练配置（6 个 YAML，对应升级链路每一步）
├── preprocess/                       # 数据预处理流水线
│   ├── run_all_v2.py                # 一键运行入口
│   ├── step1_build_vocabs_v2.py     # 构建词表和处理基础特征
│   ├── step2_compute_stats.py       # 计算物品统计量（仅用 pre 期数据）
│   ├── step3_generate_samples_v2.py # 生成样本（含行为序列构建）
│   ├── step4_split_and_save_v2.py   # 切分 + 生成 meta
│   └── feature_config.yaml          # 特征配置
├── src/
│   ├── main_train_din.py            # 训练入口
│   ├── models/
│   │   ├── din.py                   # DIN 主模型（含全部 variant 组装逻辑）
│   │   └── modules/                 # PSRG / PCRG / Transformer / MBCNet / PPNet
│   ├── datasets/                    # ParquetIterableDataset / collate
│   ├── trainers/                    # 训练 / 验证 / 评估循环
│   ├── metrics/                     # AUC / LogLoss / GAUC
│   ├── analysis/                    # 实验汇总 / tab 统计
│   └── utils/                       # seed / checkpoint
├── scripts/
│   ├── smoke_test_din.py            # 冒烟测试
│   ├── run_ads_ablation.py          # ADS 消融实验
│   ├── bench_train_throughput.py    # 数据加载吞吐 benchmark
│   └── run_eda.py                   # 探索性数据分析
├── kuairand/eda/                    # 原始日志分析工具
├── tests/                           # PSRG / PCRG 集成测试
├── src/tests/                       # Transformer / MBCNet / PPNet shape 测试
├── reports/                         # EDA 报告 / 面试讲解材料
├── output/                          # 训练数据、词表、实验产物
└── pyproject.toml / requirements.txt
```

## 数据处理流程

### 预处理（从原始 CSV 到训练 parquet）

预处理脚本位于 `preprocess/`，完整流程分 4 步：

```bash
cd preprocess
python run_all_v2.py
```

| 步骤 | 脚本 | 说明 |
|------|------|------|
| Step 1 | `step1_build_vocabs_v2.py` | 读取用户/视频特征 CSV，构建词表（0 保留为 PAD），处理 sparse/dense 特征 |
| Step 2 | `step2_compute_stats.py` | **仅用 pre 期数据**（4/08~4/21）计算 CTR、完播率等统计量，避免数据泄漏 |
| Step 3 | `step3_generate_samples_v2.py` | 处理实验期日志（4/22~5/08），构建候选+用户+上下文+历史序列样本 |
| Step 4 | `step4_split_and_save_v2.py` | 按用户时间切分 train/val/test，生成 field_schema 和 meta 文件 |

详见 `preprocess/README.md`。

### 训练数据流

1. `src/main_train_din.py` 根据 YAML 读取 `processed / meta / vocabs`
2. `ParquetIterableDataset` 按 row group 流式读取，直接产出预组装 batch
3. `collate.py` 做 numpy → torch.Tensor 转换
4. `din.py` 根据 variant / head / ppnet 配置组装模型
5. `train_din.py` 跑训练、验证和最终评估
6. `metrics.py` 计算 AUC / LogLoss / GAUC（支持超阈值自动落盘）
7. `checkpoint.py` 保存 config_snapshot / checkpoint_best / final_metrics

## 快速开始

### 1. 安装

```bash
pip install -r requirements.txt
pip install torch scikit-learn
```

### 2. 数据准备

将 KuaiRand-27K 原始数据放到 `data/KuaiRand-27K/`：

```text
data/KuaiRand-27K/
├── log_standard_4_08_to_4_21_27k_part1.csv
├── log_standard_4_08_to_4_21_27k_part2.csv
├── log_standard_4_22_to_5_08_27k_part1.csv
├── log_standard_4_22_to_5_08_27k_part2.csv
├── log_random_4_22_to_5_08_27k.csv
├── user_features_27k.csv
└── video_features_basic_27k.csv
```

运行预处理：

```bash
cd preprocess && python run_all_v2.py
```

### 3. 冒烟测试

```bash
python scripts/smoke_test_din.py
```

### 4. Debug 训练（验证链路）

```bash
# baseline
python -m src.main_train_din --config configs/train_din_mem16gb.yaml --debug_rows 512 --epochs 1 --run_dir output/exp_runs/debug_din

# 全栈
python -m src.main_train_din --config configs/train_din_psrg_pcrg_transformer_mbcnet_ppnet.yaml --debug_rows 256 --epochs 1 --run_dir output/exp_runs/debug_fullstack
```

### 5. 正式训练

```bash
# baseline
python -m src.main_train_din --config configs/train_din_mem16gb.yaml --run_dir output/exp_runs/din_baseline

# ADS
python -m src.main_train_din --config configs/train_din_psrg_pcrg_mem16gb.yaml --run_dir output/exp_runs/ads

# Transformer
python -m src.main_train_din --config configs/train_din_psrg_pcrg_transformer.yaml --run_dir output/exp_runs/transformer

# MBCNet
python -m src.main_train_din --config configs/train_din_psrg_pcrg_transformer_mbcnet.yaml --run_dir output/exp_runs/mbcnet

# PPNet (全栈)
python -m src.main_train_din --config configs/train_din_psrg_pcrg_transformer_mbcnet_ppnet.yaml --run_dir output/exp_runs/ppnet
```

### 6. 汇总结果

```bash
python -m src.analysis.summarize_experiments --runs_root output/exp_runs --output_dir output/analysis/exp_summary
```

### 7. 运行测试

```bash
pytest -q tests src/tests
```

## 工程亮点

- **无偏评测**：standard / random 日志分开处理，test_random 单独保留作为无分发偏置的评测集
- **流式读取**：ParquetIterableDataset 按 row group 预组装 batch，避免逐样本 Python 循环
- **显存友好**：大词表默认 HashEmbedding，评估阶段支持超阈值自动落盘
- **配置驱动**：variant / num_queries / transformer / head / ppnet 全部通过 YAML + CLI 控制
- **实验可追溯**：每个 run 自动保存 config_snapshot / checkpoint_best / final_metrics
- **防泄漏设计**：物品统计量仅用 pre 期数据计算，训练/测试按用户+时间切分
- **测试覆盖**：覆盖全 padding、mask fallback、query 维度不一致、MBCNet 分组兜底、PPNet 广播调制等边界情况

## 当前限制

- `pyproject.toml` 中 `kuairand-din-build` entry 指向的模块不存在，预处理请直接使用 `preprocess/run_all_v2.py`
- 训练依赖声明不完整，`torch` 和 `scikit-learn` 需手动安装
- `video_features_statistic` 相关特征未接入主训练链路（时间泄漏风险未完成严格对照）
- PPNet 的 branch gate 路径已在代码和测试中打通，但主配置仍为 `apply_to=head_input`
