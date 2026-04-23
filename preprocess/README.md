# KuaiRand-27K 预处理脚本

完整的数据预处理流程，生成符合训练框架要求的 parquet 和 meta 文件。

## 输出结构

```
output/
├── processed/
│   ├── train.parquet              # 训练集 (~1.3亿样本)
│   ├── val.parquet                # 验证集 (~1800万样本)
│   ├── test_standard.parquet      # 标准测试集 (~3700万样本)
│   └── test_random.parquet        # 随机测试集 (~118万样本)
├── meta/
│   ├── field_schema.json          # 字段schema定义
│   ├── vocab_sizes.json           # 各字段词表大小
│   ├── feature_vocab_manifest.json # 词表文件映射
│   ├── split_summary.json         # 切分统计信息
│   └── sanity_checks.json         # 数据检查结果
└── vocabs/
    ├── video_id.json              # 视频ID词表
    ├── author_id.json             # 作者ID词表
    ├── video_type.json            # 视频类型词表
    ├── upload_type.json           # 上传类型词表
    ├── music_type.json            # 音乐类型词表
    ├── tab.json                   # 场景词表
    └── ...                        # 其他词表
```

## 使用方式

### 方式1：一键运行（推荐）

```bash
cd preprocess
python run_all.py
```

### 方式2：分步运行

```bash
cd preprocess
python step1_build_vocabs.py      # 构建词表和处理基础特征
python step2_compute_stats.py     # 计算物品统计量
python step3_generate_samples.py  # 生成样本（最耗时）
python step4_split_and_save.py    # 切分和保存
```

## 预处理步骤说明

### Step 1: 构建词表和基础特征

- 读取 `user_features_27k.csv` 和 `video_features_basic_27k.csv`
- 为所有 sparse 特征构建词表（值从1开始，0为PAD）
- 处理视频特征：编码ID、video_type、upload_type、music_type、duration分桶
- 处理用户特征：
  - Sparse: 分桶版本（如 `follow_user_num_range`）
  - Dense: 原始值log1p（如 `log1p_follow_user_num`）
- 输出：`video_features_processed.parquet`, `user_features_processed.parquet`, `vocabs/*.json`

### Step 2: 计算物品统计量

- 仅使用 `log_standard_4_08_to_4_21`（pre期，无泄露）
- 计算每个视频的：
  - 展示数、CTR、long_view率、like率、播放比例
  - 等频分桶（5桶）
  - log1p压缩的展示数
- 输出：`item_statistics.parquet`

### Step 3: 生成样本

- 处理 `log_standard_4_22_to_5_08` 和 `log_random_4_22_to_5_08`
- 为每个交互事件生成样本：
  - 候选视频特征（join video_basic + item_stats）
  - 用户特征（sparse + dense）
  - 上下文特征（tab, hour, day, is_weekend）
  - 历史序列（定长50，含 mask、delta_t_bucket、play_ratio_bucket、tab）
- 正反馈筛选：long_view=1 或 is_like=1 才入历史池
- 分块处理以控制内存
- 输出：`samples_standard.parquet`, `samples_random.parquet`

### Step 4: 切分和生成Meta

- 按用户切分 train/val/test（80/10/10），避免泄露
- random样本单独保存为 test_random
- 生成所有meta文件：
  - `field_schema.json`: 字段定义，匹配 `train_din_mem16gb.yaml`
  - `vocab_sizes.json`: 各字段词表大小
  - `feature_vocab_manifest.json`: 词表文件映射
  - `split_summary.json`: 切分统计
  - `sanity_checks.json`: 数据检查（标签分布、用户重叠、序列长度等）

## 数据特征说明

### 标签
- `label_long_view`: 主标签（0/1）

### 用户ID
- `user_id_raw`: 用户原始ID（用于GAUC计算）

### 历史序列（定长50）
- `hist_video_id`: list[int64]，历史视频ID（vocab索引）
- `hist_author_id`: list[int64]，历史作者ID（vocab索引）
- `hist_mask`: list[int8]，1=有效，0=padding
- `hist_len`: int8，实际历史长度
- `hist_delta_t_bucket`: list[int64]，时间差分桶
- `hist_play_ratio_bucket`: list[int64]，播放比例分桶
- `hist_tab`: list[int64]，历史场景

### 候选Item特征（Sparse）
- `cand_video_id`: 视频ID
- `cand_author_id`: 作者ID
- `cand_video_type`: 视频类型（NORMAL/AD）
- `cand_upload_type`: 上传类型
- `cand_music_type`: 音乐类型
- `cand_video_duration_bucket`: 时长分桶
- `cand_pre_ctr_bucket`: pre期CTR分桶
- `cand_pre_lv_rate_bucket`: pre期long_view率分桶
- `cand_pre_like_rate_bucket`: pre期like率分桶

### 候选Item特征（Dense）
- `cand_pre_log_show_cnt`: log1p(pre期展示数)

### 上下文特征
- `tab`: 场景标签
- `hour_of_day`: 小时（0-23 +1）
- `day_of_week`: 星期（0-6 +1）
- `is_weekend`: 是否周末（1=工作日，2=周末）

### 用户Sparse特征（分桶版本）
- `user_active_degree`: 活跃等级
- `is_lowactive_period`: 是否低活跃期
- `is_live_streamer`: 是否主播
- `is_video_author`: 是否视频作者
- `follow_user_num_range`: 关注数分桶
- `fans_user_num_range`: 粉丝数分桶
- `friend_user_num_range`: 好友数分桶
- `register_days_range`: 注册天数分桶
- `onehot_feat0` ~ `onehot_feat17`: 匿名one-hot特征

### 用户Dense特征（原始值log1p）
- `log1p_follow_user_num`: log1p(关注数)
- `log1p_fans_user_num`: log1p(粉丝数)
- `log1p_friend_user_num`: log1p(好友数)
- `log1p_register_days`: log1p(注册天数)

### Meta列
- `user_id_raw`: 用户ID
- `meta_is_rand`: 是否随机曝光（0=standard, 1=random）
- `sample_time_ms`: 样本时间戳

## 词表规则

- 所有 sparse 特征的值 **+1** 后入词表
- **0 保留为 PAD token**
- 示例：原始值 0 → 编码 1，原始值 1 → 编码 2，PAD → 0

## 注意事项

1. **内存需求**：Step 3 需要较多内存，已做分块处理，建议 32GB+ RAM
2. **时间预估**：完整运行约需 2-4 小时（取决于硬件）
3. **显存优化**：video_id 和 author_id 词表很大，训练时建议使用 HashEmbedding
4. **冷启动**：未见过的视频/作者在统计特征和ID编码上均为0（PAD）

## 与训练框架的对接

生成的文件可直接被以下配置使用：
- `configs/train_din_mem16gb.yaml`
- `configs/train_din_psrg_pcrg_mem16gb.yaml`
- `configs/train_din_psrg_pcrg_transformer.yaml`
- 等等

运行训练：
```bash
python -m src.main_train_din --config configs/train_din_mem16gb.yaml
```
