# 全链路详细操作手册

> 本文档精确到每个张量的 shape 变化、每个矩阵乘法和激活函数，
> 覆盖从原始 batch 输入到最终 logit 输出的全部计算步骤。
> 以最完整的 `+PPNet` 配置 (full track, batch_size=4096) 为例。

---

## 0. 输入 batch 结构

DataLoader 输出一个 `Dict[str, Tensor]`，关键字段：

```
batch = {
    # ── 候选视频 ──
    "cand_video_id":             [4096]          int64     # hash 前的编码 ID
    "cand_author_id":            [4096]          int64
    "cand_video_type":           [4096]          int64     # 0=PAD, 1=NORMAL, 2=AD
    "cand_upload_type":          [4096]          int64
    "cand_video_duration_bucket":[4096]          int64     # 1~5
    "cand_pre_ctr_bucket":       [4096]          int64     # 0~2 (smart_zero 3桶)
    "cand_pre_lv_rate_bucket":   [4096]          int64
    "cand_pre_like_rate_bucket": [4096]          int64
    "cand_pre_play_ratio_bucket":[4096]          int64     # 1~5 (uniform 5桶)
    "cand_pre_show_bucket":      [4096]          int64     # 0~5 (log尺度)
    "cand_pre_ctr":              [4096]          float32   # dense 原始值
    "cand_pre_lv_rate":          [4096]          float32
    "cand_pre_like_rate":        [4096]          float32
    "cand_pre_play_ratio":       [4096]          float32
    "cand_pre_show_log":         [4096]          float32

    # ── 历史序列 ──
    "hist_video_id":             [4096, 50]      int64     # 已 pad 到 50
    "hist_author_id":            [4096, 50]      int64
    "hist_mask":                 [4096, 50]      int64     # 1=有效, 0=padding
    "hist_len":                  [4096]          int64     # 实际序列长度
    "hist_delta_t_bucket":       [4096, 50]      int64     # 1~6
    "hist_play_ratio_bucket":    [4096, 50]      int64     # 1~6
    "hist_tab":                  [4096, 50]      int64     # 0~15

    # ── 上下文 ──
    "tab":                       [4096]          int64     # 1~15
    "hour_of_day":               [4096]          int64     # 1~24
    "day_of_week":               [4096]          int64     # 1~7
    "is_weekend":                [4096]          int64     # 1=否, 2=是

    # ── 用户画像 sparse ──
    "user_active_degree":        [4096]          int64
    "is_lowactive_period":       [4096]          int64     # 0/1/2
    "is_live_streamer":          [4096]          int64
    "is_video_author":           [4096]          int64
    "follow_user_num_range":     [4096]          int64
    "fans_user_num_range":       [4096]          int64
    "friend_user_num_range":     [4096]          int64
    "register_days_range":       [4096]          int64
    "onehot_feat0"~"onehot_feat17": [4096]      int64     # 18 个

    # ── 用户画像 dense ──
    "user_dense":                [4096, 4]       float32   # log1p(follow/fans/friend/register)

    # ── 元数据（评估时才用）──
    "user_id_raw":               [4096]          int64
    "label_long_view":           [4096]          int64     # 0 或 1
}
```

### 0.1 字段详细说明

#### 候选视频特征

| 字段名 | Shape | 类型 | 含义 | 取值说明 |
|--------|-------|------|------|---------|
| `cand_video_id` | [4096] | int64 | 候选视频的编码 ID | Step1 全量枚举排序后的 ID，进入模型后经 HashEmbedding 映射到 1M 桶再查 32 维 embedding |
| `cand_author_id` | [4096] | int64 | 候选视频作者的编码 ID | 同上，HashEmbedding 映射到 500K 桶，查 16 维 embedding |
| `cand_video_type` | [4096] | int64 | 视频类型 | 0=PAD(缺失/未知), 1=NORMAL(普通视频), 2=AD(广告)。区分广告和普通视频的用户点击模式差异 |
| `cand_upload_type` | [4096] | int64 | 上传方式 | 如直接上传、转发等，Step1 从原始数据枚举映射 |
| `cand_video_duration_bucket` | [4096] | int64 | 视频时长分桶 | 按 [7s, 18s, 30s, 60s] 毫秒边界分 5 桶: 1=极短(<7s), 2=短(7-18s), 3=中(18-30s), 4=长(30-60s), 5=超长(>60s), 0=PAD |
| `cand_pre_ctr_bucket` | [4096] | int64 | 先验点击率分桶 (sparse) | smart_zero 3桶: 0=无数据/冷启动(CTR=0的视频), 1=低CTR, 2=高CTR。仅用 pre 期(4/08-4/21)数据计算，防泄露 |
| `cand_pre_lv_rate_bucket` | [4096] | int64 | 先验完播率分桶 (sparse) | smart_zero 3桶: 0=无完播数据, 1=低完播率, 2=高完播率 |
| `cand_pre_like_rate_bucket` | [4096] | int64 | 先验点赞率分桶 (sparse) | smart_zero 3桶: 0=无点赞(99%的视频), 1=低点赞率, 2=高点赞率 |
| `cand_pre_play_ratio_bucket` | [4096] | int64 | 先验播放比例分桶 (sparse) | uniform 等频 5桶: 1~5，播放比例=播放时长/视频时长，分布相对均匀无需 smart_zero |
| `cand_pre_show_bucket` | [4096] | int64 | 先验曝光次数分桶 (sparse) | log 尺度 5桶: 0=冷启动, 1=1次, 2=2-9次, 3=10-49次, 4=50-199次, 5=≥200次 |
| `cand_pre_ctr` | [4096] | float32 | 先验点击率原始值 (dense) | 0.0~1.0 的连续值，保留了 0.032 和 0.035 之间的细微差异，与分桶版互补 |
| `cand_pre_lv_rate` | [4096] | float32 | 先验完播率原始值 (dense) | 0.0~1.0 |
| `cand_pre_like_rate` | [4096] | float32 | 先验点赞率原始值 (dense) | 0.0~1.0，绝大多数为 0 |
| `cand_pre_play_ratio` | [4096] | float32 | 先验平均播放比例原始值 (dense) | 0.0~3.0，clip 上限 3.0 |
| `cand_pre_show_log` | [4096] | float32 | 先验曝光次数的 log1p (dense) | log1p(曝光数)，用 log 压缩长尾（有的视频曝光百万次） |

> **sparse 版 vs dense 版**：sparse 分桶进 Embedding 查表得到可学习向量（对极端值鲁棒），dense 原始值直接输入 MLP（保留数值精度）。实验证明两者互补，模型越复杂越能利用 dense 的细粒度差异（PPNet 阶段 full 比 sparse 多涨 +0.70pt）。

#### 历史序列特征

| 字段名 | Shape | 类型 | 含义 | 取值说明 |
|--------|-------|------|------|---------|
| `hist_video_id` | [4096, 50] | int64 | 历史正交互视频 ID 序列 | 用户最近 50 条正交互（click=1 或 long_view=1）的视频，不足 50 用 0 填充。与 cand_video_id 共享同一套 HashEmbedding 权重 |
| `hist_author_id` | [4096, 50] | int64 | 历史正交互作者 ID 序列 | 与 hist_video_id 对应的作者 ID，共享 cand_author_id 的 Embedding |
| `hist_mask` | [4096, 50] | int64 | 历史序列有效位掩码 | 1=该位置有真实历史交互, 0=padding。用于 attention 中屏蔽无效位置（填 -1e4 后 softmax） |
| `hist_len` | [4096] | int64 | 实际历史长度 | 0~50，新用户可能为 0（无历史），此时 DIN attention 输出零向量 |
| `hist_delta_t_bucket` | [4096, 50] | int64 | 时间间隔分桶 | 每条历史交互距上一条的时间差，按 [1min, 5min, 30min, 1day, 3day] 分桶: 1=<1min(连续刷), 2=1-5min, 3=5-30min, 4=30min-1day, 5=1-3day, 6=>3day。捕捉兴趣时效性 |
| `hist_play_ratio_bucket` | [4096, 50] | int64 | 播放完成度分桶 | 每条历史的播放时长/视频时长，按 [0.2, 0.5, 0.8, 1.2, 2.0] 分桶: 1=只看了<20%(划走), 2=看了一半, 3=看了大部分, 4=完整看完, 5=看了1.2-2倍(可能重放), 6=看了>2倍(反复观看)。区分兴趣强度 |
| `hist_tab` | [4096, 50] | int64 | 历史交互发生的 Tab 页 | 0~15，标记该次交互发生在哪个场景（关注/发现/同城等）。被 PSRG 用来做跨域门控——同 tab 的历史更相关 |

> **为什么只取正交互？** 负反馈（曝光未点击）数量是正反馈的 ~20 倍，全加入会稀释兴趣信号。而且负反馈可能是位置偏差（用户没看到），不代表不感兴趣。

#### 上下文特征

| 字段名 | Shape | 类型 | 含义 | 取值说明 |
|--------|-------|------|------|---------|
| `tab` | [4096] | int64 | 当前曝光的 Tab 页 | 1~15，快手的 15 个场景（关注/发现/同城/...）。跨域排序的核心域标识，驱动 DomainContext → PSRG/PCRG 的域感知行为 |
| `hour_of_day` | [4096] | int64 | 曝光发生的小时 | 1~24（原始 0~23 加 1，0 保留给 PAD）。捕捉时间偏好：深夜用户可能偏好不同内容 |
| `day_of_week` | [4096] | int64 | 曝光发生的星期几 | 1~7（周一=1，周日=7，0 保留给 PAD）。工作日 vs 周末行为模式不同 |
| `is_weekend` | [4096] | int64 | 是否周末 | 1=工作日, 2=周末（0 保留给 PAD）。注意编码是 +1 后的值，不是直接的 0/1 |

> 这 4 个特征拼接后送入 DomainContextEncoder 编码为 d_ctx [B, 48]，也直接拼入 head_input 的 context_embs 块。

#### 用户画像特征 (sparse)

| 字段名 | Shape | 类型 | 含义 | 取值说明 |
|--------|-------|------|------|---------|
| `user_active_degree` | [4096] | int64 | 用户活跃度等级 | 0=PAD, 1=high_active, 2=full_active, 3=middle_active, 4=UNKNOWN |
| `is_lowactive_period` | [4096] | int64 | 是否处于低活跃期 | 0=PAD(数据异常/缺失), 1=否, 2=是。原始数据中负值(-124等)映射为 0 |
| `is_live_streamer` | [4096] | int64 | 是否直播主播 | 0=PAD, 1=否, 2=是。主播身份可能影响其观看偏好（更关注竞品/学习） |
| `is_video_author` | [4096] | int64 | 是否视频创作者 | 0=PAD, 1=否, 2=是。创作者可能更关注同类型内容 |
| `follow_user_num_range` | [4096] | int64 | 关注数分段 | Step1 从原始 CSV 中枚举映射，如 "0", "[1,10)", "[10,50)" 等 |
| `fans_user_num_range` | [4096] | int64 | 粉丝数分段 | 同上，反映用户的社交影响力 |
| `friend_user_num_range` | [4096] | int64 | 好友数分段 | 同上 |
| `register_days_range` | [4096] | int64 | 注册天数分段 | 同上，区分新用户和老用户 |
| `onehot_feat0`~`onehot_feat17` | [4096] | int64 | 18 个匿名类别特征 | 快手脱敏后的用户标签，具体语义未公开。编码方式: 原始值+1，负值→0(PAD) |

> 这 26 个 sparse 特征各自通过 Embedding(词表大小, 4 或 8) 查表后 concat，构成 user_sparse_embs [B, ~104]。

#### 用户画像特征 (dense)

| 字段名 | Shape | 类型 | 含义 | 取值说明 |
|--------|-------|------|------|---------|
| `user_dense` | [4096, 4] | float32 | 4 个用户连续值特征 | 由 BatchCollateFn 将 4 列合并而成 |

4 个维度分别是：

| 维度 | 原始字段 | 含义 | 变换 |
|------|---------|------|------|
| dim 0 | `log1p_follow_user_num` | 关注数 | log1p(max(原始值, 0))，压缩长尾 |
| dim 1 | `log1p_fans_user_num` | 粉丝数 | 同上。大 V 可能百万粉丝，log1p 后约 14 |
| dim 2 | `log1p_friend_user_num` | 好友数 | 同上 |
| dim 3 | `log1p_register_days` | 注册天数 | 同上。区分新老用户 |

> 与 sparse 版的 `*_range` 特征互补：sparse 版按段分桶（"[1,10)"），dense 版保留精确数值。

#### 元数据（不参与模型计算）

| 字段名 | Shape | 类型 | 含义 | 取值说明 |
|--------|-------|------|------|---------|
| `user_id_raw` | [4096] | int64 | 原始用户 ID | 不送入模型，仅用于 GAUC 计算时按用户分组。训练时不读取（节省 IO） |
| `label_long_view` | [4096] | int64 | 是否完播标签 | 0=未完播, 1=完播。模型的预测目标，用于 BCEWithLogitsLoss |

---

## 1. Embedding 查表

> **设计思路**：Embedding 层是整个模型的"地基"。这里的核心决策有三个：
> 1. **Hash 取模代替全量词表** — 原始 video_id 有 3200 万种，如果做全量 Embedding(32M, 32)，光这一层就需要 ~4GB 显存。HashEmbedding 将 ID 取模映射到 100 万桶，参数量压缩 32 倍，代价是少量哈希冲突（实测 GAUC 几乎无损）。
> 2. **候选与历史共享权重** — cand_video_id 和 hist_video_id 查的是同一张 Embedding 表。这样"候选 embedding"和"历史 embedding"天然在同一个语义空间，DIN attention 的内积打分才有意义。
> 3. **Side Info 用残差加入** — 历史序列的时间间隔、播放比例、tab 三个特征通过 `Embedding → Linear → 残差相加` 注入，而不是 concat。残差保持了原始 item repr 的主导地位，side info 只做微调，训练更稳定。

### 1.1 候选视频 Item Representation

```
操作 1: HashEmbedding(cand_video_id)
  输入: cand_video_id [4096]
  计算: hashed = (x % 999999) + 1, PAD位=0
       emb = Embedding(1000000, 32, padding_idx=0)
  输出: cand_vid_emb [4096, 32]

操作 2: HashEmbedding(cand_author_id)
  输入: cand_author_id [4096]
  计算: hashed = (x % 499999) + 1
       emb = Embedding(500000, 16, padding_idx=0)
  输出: cand_aid_emb [4096, 16]

操作 3: concat
  输入: cand_vid_emb [4096, 32] + cand_aid_emb [4096, 16]
  输出: cand_item_repr [4096, 48]    ← 核心维度 D=48
```

> **操作详解**：
>
> | 操作 | 做了什么 | 为什么这样设计 |
> |------|---------|---------------|
> | 操作 1: HashEmbedding(video_id) | 将 3200 万 video_id 取模映射到 100 万桶，查 32 维 embedding | 桶数 100 万是经验值：太小冲突多，太大显存爆。32 维是 video 的主 embedding 维度 |
> | 操作 2: HashEmbedding(author_id) | 同理，50 万桶，查 16 维 embedding | author 的基数（~10 万）比 video 小，50 万桶足够；16 维是因为 author 信息量比 video 少 |
> | 操作 3: concat | 将 video_emb 和 author_emb 拼接 | 不用相加是因为 video 和 author 是不同语义（"什么内容" vs "谁拍的"），拼接保留两者独立表达 |
>
> **D=48 的由来**：32(video) + 16(author) = 48。这个维度贯穿整个模型——DIN attention、PSRG、PCRG、TransformerFusion 的 d_model 都基于 48。

### 1.2 历史序列 Item Representation

```
操作 4: HashEmbedding(hist_video_id)  ← 与操作1共享权重
  输入: hist_video_id [4096, 50]
  输出: hist_vid_emb [4096, 50, 32]

操作 5: HashEmbedding(hist_author_id) ← 与操作2共享权重
  输入: hist_author_id [4096, 50]
  输出: hist_aid_emb [4096, 50, 16]

操作 6: concat
  输入: hist_vid_emb [4096, 50, 32] + hist_aid_emb [4096, 50, 16]
  输出: hist_repr_base [4096, 50, 48]
```

### 1.3 历史 Side Information 残差加入

```
操作 7: Embedding(hist_delta_t_bucket)
  Embedding(7, 4, padding_idx=0) → [4096, 50, 4]
  Linear(4, 48) 投影到 item 维度 → [4096, 50, 48]
  hist_repr = hist_repr_base + projected  ← 残差相加

操作 8: Embedding(hist_play_ratio_bucket)
  Embedding(7, 4, padding_idx=0) → [4096, 50, 4]
  Linear(4, 48) → [4096, 50, 48]
  hist_repr = hist_repr + projected

操作 9: Embedding(hist_tab)
  Embedding(16, 4, padding_idx=0) → [4096, 50, 4]
  Linear(4, 48) → [4096, 50, 48]
  hist_repr = hist_repr + projected

最终: hist_item_repr [4096, 50, 48]
```

> **Side Info 残差机制详解**：
>
> | Side Info 特征 | Embedding 维度 | 投影 | 语义 |
> |---------------|---------------|------|------|
> | `hist_delta_t_bucket` | Embedding(7, 4) | Linear(4→48) | 这条历史距上一条多久？<1min=连续刷，>3day=很久前的兴趣 |
> | `hist_play_ratio_bucket` | Embedding(7, 4) | Linear(4→48) | 用户看了多少？<20%=划走(弱兴趣)，>200%=反复看(强兴趣) |
> | `hist_tab` | Embedding(16, 4) | Linear(4→48) | 这条历史在哪个 tab 发生？同域 vs 跨域的兴趣迁移依据 |
>
> **为什么用残差相加而不是 concat？**
> - concat 会把 48 维变成 48+48+48+48=192 维，后续所有模块的维度都要跟着变，参数量暴增
> - 残差相加保持 D=48 不变，side info 只做"微调"——比如 delta_t 编码让最近的交互 embedding 略有偏移，但不改变 item 的主语义
> - 类比 Transformer 的位置编码：也是加法注入，不改变主维度

### 1.4 其他 Sparse Embedding 查表

```
操作 10: 候选侧 side embedding（后续 head_input 用）
  cand_video_type:           Embedding(3, 4)   → [4096, 4]
  cand_upload_type:          Embedding(V, 4)   → [4096, 4]
  cand_video_duration_bucket:Embedding(6, 4)   → [4096, 4]
  cand_pre_ctr_bucket:       Embedding(4, 4)   → [4096, 4]
  cand_pre_lv_rate_bucket:   Embedding(4, 4)   → [4096, 4]
  cand_pre_like_rate_bucket: Embedding(4, 4)   → [4096, 4]
  cand_pre_play_ratio_bucket:Embedding(6, 4)   → [4096, 4]
  cand_pre_show_bucket:      Embedding(6, 4)   → [4096, 4]
  concat → cand_side_embs [4096, ~32]

操作 11: 上下文 embedding
  tab:          Embedding(16, 4)  → [4096, 4]
  hour_of_day:  Embedding(25, 4)  → [4096, 4]
  day_of_week:  Embedding(8, 4)   → [4096, 4]
  is_weekend:   Embedding(3, 4)   → [4096, 4]
  concat → context_embs [4096, 16]

操作 12: 用户画像 sparse embedding
  user_active_degree:    Embedding(5, 4)     → [4096, 4]
  is_lowactive_period:   Embedding(3, 4)     → [4096, 4]
  is_live_streamer:      Embedding(3, 4)     → [4096, 4]
  is_video_author:       Embedding(3, 4)     → [4096, 4]
  follow_user_num_range: Embedding(V, 4)     → [4096, 4]
  fans_user_num_range:   Embedding(V, 4)     → [4096, 4]
  friend_user_num_range: Embedding(V, 4)     → [4096, 4]
  register_days_range:   Embedding(V, 4)     → [4096, 4]
  onehot_feat0~17:       各 Embedding(V, 4~8) → [4096, 4~8] × 18
  concat → user_sparse_embs [4096, ~104]
```

> **Embedding 维度选择策略**：
>
> | 特征类别 | 典型词表大小 | Embedding 维度 | 设计理由 |
> |---------|------------|---------------|---------|
> | video_id | 100 万 (Hash) | 32 | 高基数 ID，需要足够维度表达 |
> | author_id | 50 万 (Hash) | 16 | 中等基数，信息量比 video 少 |
> | 候选 side 特征 (type/duration/bucket) | 3~6 | 4 | 低基数类别，4 维足够区分 |
> | 上下文特征 (tab/hour/dow) | 3~25 | 4 | 同上，且总数只有 4 个特征 |
> | 用户画像特征 | 3~几十 | 4~8 | 低基数用 4 维，onehot_feat 词表稍大用 8 维 |
> | 历史 side info | 7~16 | 4 | 辅助信息，4 维 + Linear 投影到 48 |
>
> **经验规则**：Embedding 维度 ≈ min(词表大小的 4 次方根, 合理上限)。基数 <10 的特征用 4 维足够，基数 >1 万的用 16~32 维。

---

## 2. DomainContextEncoder — 域上下文编码

> **设计思路**：域上下文 d_ctx 是跨域排序的核心信号。它回答一个问题："当前用户处于什么推荐场景？"
> - **为什么只用 tab + hour + day_of_week？** 这三个特征完整描述了"域"——tab 决定内容分发策略（关注页 vs 发现页行为截然不同），hour+dow 捕捉时间模式（深夜刷短视频 vs 工作日午休）。is_weekend 是 dow 的冗余，不再重复加入。
> - **为什么要过 MLP？** 直接拼接 embedding 只有 12 维，信息太稀疏。MLP 把 12 维投影到 48 维（与 item repr 同维），使得 d_ctx 能直接与 item embedding 做加法/拼接等操作。
> - **为什么输出维度 = 48？** 与 item_repr 同维是刻意设计——PSRG 中需要 `[hist_item_repr, d_ctx_seq]` 拼接，PCRG 中需要 `[cand_item_repr, d_ctx]` 拼接，维度一致让后续计算更统一。

```
操作 13: 收集域上下文 embedding
  tab_emb [4096, 4] + hour_emb [4096, 4] + dow_emb [4096, 4]
  concat → ctx_concat [4096, 12]

操作 14: MLP 编码
  Linear(12, 64) → ReLU → Linear(64, 48) → LayerNorm(48)
  输出: d_ctx [4096, 48]
```

> **操作详解**：
>
> | 操作 | 做了什么 | 为什么这样设计 |
> |------|---------|---------------|
> | 操作 13: concat 3 个 embedding | 将 tab/hour/dow 的 embedding 拼接为 12 维向量 | 3 个离散特征各 4 维，拼接后保留各自独立语义 |
> | 操作 14: MLP 编码 | 12 → 64 → 48，两层 MLP 加 LayerNorm | 第一层 12→64 做升维（让 ReLU 有足够空间学非线性组合），第二层 64→48 压回目标维度。LayerNorm 稳定下游模块的输入分布 |
>
> **d_ctx 的下游使用**：
> - **PSRG**：与历史序列逐位置拼接，生成门控信号（判断每条历史在当前域是否相关）
> - **PCRG**：与候选拼接，生成 4 个域感知 query（让不同域关注不同兴趣维度）
> - d_ctx 不直接进 head_input（那里用的是原始 context_embs [B,16]）

---

## 3. PSRG — 域感知历史重映射

> **核心问题**：用户在"关注页"点赞的搞笑视频，在"发现页"推荐时是否还有用？
> PSRG (Personalized Sequence Re-mapping Gate) 的答案是：**有用，但要打折**。它为每条历史交互生成一个 [0,1] 的门控值，衡量"这条历史在当前域下有多相关"。
>
> **与 ADS 原文的区别**：
> - ADS 的 SRG 为每个域生成完整的 MLP 权重矩阵（HyperNetwork），参数量巨大
> - PSRG-lite 改为门控残差：`h' = h + sigmoid(gate) * delta`，不生成权重矩阵，只生成门控信号和增量
> - 效果：参数量从 O(D²) 降到 O(D)，训练更稳定，4GB GPU 能跑
>
> **直觉理解**：
> - gate ≈ 1, delta 大 → 这条历史被大幅调整（跨域迁移）
> - gate ≈ 0 → 这条历史保持原样（同域直接用）
> - LayerNorm 保证调整后的表示不会偏离太远

### 3.1 构建逐位置域上下文 d_ctx_seq

```
操作 15: 当前 tab embedding 广播
  tab_emb [4096, 4] → unsqueeze(1) → expand → [4096, 50, 4]

操作 16: 历史 tab embedding
  hist_tab_emb [4096, 50, 4]  (已在操作9中查表)

操作 17: 拼接
  [current_tab_emb, hist_tab_emb] → d_ctx_seq [4096, 50, 8]
```

### 3.2 门控残差计算

```
操作 18: 拼接历史与域上下文
  [hist_item_repr, d_ctx_seq] → merged [4096, 50, 56]
  reshape → [204800, 56]   (B*L 合并便于 MLP 计算)

操作 19: 生成门控信号
  gate_mlp: Linear(56, 64) → ReLU → Linear(64, 48)
  sigmoid → gate [204800, 48]
  reshape → [4096, 50, 48]

操作 20: 生成增量
  delta_mlp: Linear(56, 64) → ReLU → Linear(64, 48)
  → delta [204800, 48]
  reshape → [4096, 50, 48]

操作 21: 门控残差更新
  hist_out = hist_item_repr + gate * delta
  hist_out = LayerNorm(hist_out)

操作 22: Mask 安全
  mask_f = hist_mask.unsqueeze(-1) [4096, 50, 1]
  hist_for_attn = hist_out * mask_f + hist_item_repr * (1 - mask_f)
  → padding 位保持原始值

最终: hist_for_attn [4096, 50, 48]
```

> **逐操作详解**：
>
> | 操作 | 做了什么 | 为什么这样设计 |
> |------|---------|---------------|
> | 操作 15-17: 构建 d_ctx_seq | 将当前 tab embedding 广播到 50 个位置，与每个位置的历史 tab embedding 拼接为 [B,50,8] | 每个位置的域上下文不同——位置 i 的上下文是"当前 tab=3，历史位置 i 的 tab=1"，这个 pair 决定了跨域距离 |
> | 操作 18: 拼接 | hist_item_repr [B,50,48] 与 d_ctx_seq [B,50,8] 拼接为 [B,50,56] | MLP 需要同时看到"item 是什么"和"域关系是什么"才能做出准确的门控决策 |
> | 操作 19: gate_mlp → sigmoid | 56 → 64 → 48，sigmoid 输出 [0,1] | sigmoid 保证门控值在 [0,1]，语义为"这条历史有多少信息需要跨域调整" |
> | 操作 20: delta_mlp | 56 → 64 → 48，无激活约束 | delta 是"调整量"，可正可负——比如某条历史在当前域下应该被增强（正）或抑制（负） |
> | 操作 21: gate * delta + 残差 | h' = h + gate * delta, 再 LayerNorm | 核心公式。gate=0 时 h'=h（不调整），gate=1 时 h'=h+delta（完全调整）。LayerNorm 防止调整后 norm 飘移 |
> | 操作 22: mask 安全 | padding 位保持原始值 | 对 padding 位做门控没有意义（梯度浪费），直接跳过保持原始零向量 |
>
> **具体例子**：用户当前在「发现页(tab=3)」，历史中一个在「关注页(tab=1)」的点击搞笑视频：
> - d_ctx_seq[i] = concat[tab3_emb, tab1_emb] = [8维]，编码了"发现页←关注页"的跨域关系
> - gate_mlp 学到跨域pair (3,1) 的gate≈0.6（中等调整：搞笑兴趣部分可迁移）
> - delta_mlp 输出增量，可能放大"搞笑"维度、抑制"关注页特有"维度
> - 最终 h'保留了搞笑兴趣但去除了"因为关注了该作者才点击"的偏置

---

## 4. DIN Attention — 基线兴趣提取

> **设计思路**：DIN (Deep Interest Network) 是推荐系统中最经典的序列建模方法。核心思想是：**不是所有历史都同等重要，用候选 item 去查询哪些历史最相关**。
>
> 与 PCRG 的多兴趣提取不同，DIN 只产出一个兴趣向量——代表"与当前候选最相关的用户兴趣"。两者后续会 concat + 投影融合。
>
> **注意力四元组 [q, k, q-k, q*k] 的含义**：
> - `q` (候选): "我是什么内容"
> - `k` (历史): "用户看过什么"
> - `q-k` (差异): "候选和历史有多不同"——差异小说明相似
> - `q*k` (交互): "候选和历史在哪些维度同时激活"——逐元素乘捕捉对齐关系
> - 四者拼接后过 MLP，让模型自动学习最佳的相似度函数，比 cosine 或内积更灵活

```
操作 23: 构建注意力输入
  query = cand_item_repr [4096, 48]
  query_expand = query.unsqueeze(1).expand(-1, 50, -1) → [4096, 50, 48]

  att_input = concat[
    query_expand,                     # q     [4096, 50, 48]
    hist_for_attn,                    # k     [4096, 50, 48]
    query_expand - hist_for_attn,     # q-k   [4096, 50, 48]
    query_expand * hist_for_attn,     # q*k   [4096, 50, 48]
  ] → [4096, 50, 192]

操作 24: MLP 打分
  Linear(192, 64) → PReLU → Linear(64, 32) → PReLU → Linear(32, 1)
  → att_scores [4096, 50]

操作 25: Masked Softmax
  att_scores[padding] = -1e4
  probs = softmax(att_scores, dim=-1)
  probs = probs * mask         # 重新清零 padding
  probs = probs / sum(probs)   # 重新归一化

操作 26: 加权求和
  user_interest_din = bmm(probs.unsqueeze(1), hist_for_attn).squeeze(1)
  → [4096, 48]

全 padding 安全: 如果某样本历史全为空，user_interest_din = 零向量
```

> **逐操作详解**：
>
> | 操作 | 做了什么 | 为什么这样设计 |
> |------|---------|---------------|
> | 操作 23: 构建四元组 | q/k/q-k/q*k 拼接为 192 维 | 经典 DIN 设计。4×48=192，给 MLP 足够的"证据"来判断候选与每条历史的相关性 |
> | 操作 24: MLP 打分 | 192→64→32→1，PReLU 激活 | 3 层 MLP 足够学非线性相似度。PReLU 比 ReLU 好——允许负值通过（某些"不相似"的信号也有用） |
> | 操作 25: Masked Softmax | padding 位填 -1e4 → softmax → 重新 mask → 重新归一化 | 三步保护：①填 -1e4 让 softmax 后接近 0；②乘 mask 彻底清零（防浮点精度残留）；③重新归一化确保有效位置权重和=1 |
> | 操作 26: 加权求和 | bmm(probs, hist) | 标准注意力聚合：权重高的历史贡献大，权重低的几乎忽略。结果是一个 48 维向量，代表"当前候选视角下的用户兴趣" |
>
> **全 padding 安全**：新用户（hist_len=0）的 probs 全为 0/NaN，代码检测后直接输出零向量。下游模块对零向量是鲁棒的——Fuse1 的 Linear 会学到"零向量输入 → 给 PCRG 分支更多权重"。

---

## 5. PCRG — 多兴趣提取

> **核心问题**：DIN 只产出一个兴趣向量，但用户的兴趣是多面的——同一个用户可能同时喜欢搞笑、美食、科技。PCRG (Personalized Cross-domain Re-mapping Gate) 通过生成 **G=4 个不同的 query**，从同一个历史序列中提取 4 个不同的兴趣向量。
>
> **与 ADS 原文的区别**：
> - ADS 的 CRG 用复杂的聚合机制（如胶囊网络）融合多兴趣
> - PCRG-lite 简化为 mean_pool——4 个兴趣向量直接取平均
> - 但关键创新保留了：**query 生成时融入 d_ctx**，使得不同域生成不同的 query，从而提取域相关的兴趣子集
>
> **为什么 G=4？**
> - G 太小（如 2）：无法覆盖多面兴趣
> - G 太大（如 16）：每个 query 关注的历史太少，attention 分布过于尖锐，容易过拟合
> - G=4 是在 KuaiRand 上的最优值（config 中可调）

### 5.1 生成 4 个 Query

```
操作 27: 拼接候选与域上下文
  q_input = concat[cand_item_repr, d_ctx] → [4096, 96]

操作 28: MLP 生成多 query
  Linear(96, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 4*48=192)
  reshape → queries [4096, 4, 48]
```

### 5.2 每个 Query 独立做 DIN-style Attention

```
操作 29: 构建 4-query × 50-hist 的注意力输入
  q_expand = queries.unsqueeze(2).expand(-1, -1, 50, -1)    # [4096, 4, 50, 48]
  k_expand = hist_for_attn.unsqueeze(1).expand(-1, 4, -1, -1) # [4096, 4, 50, 48]

  pair = concat[q_expand, k_expand, q_expand-k_expand, q_expand*k_expand]
  → [4096, 4, 50, 192]

操作 30: MLP 打分
  Linear(192, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 1)
  → scores [4096, 4, 50]

操作 31: Masked Softmax（每个 query 独立）
  mask_3d = hist_mask.unsqueeze(1).expand(-1, 4, -1)  # [4096, 4, 50]
  attn = masked_softmax(scores, mask_3d)               # [4096, 4, 50]

操作 32: 加权聚合
  z = einsum("bgl,bld->bgd", attn, hist_for_attn)     # [4096, 4, 48]
  → 4 个兴趣向量 interest_tokens
```

### 5.3 聚合 4 个兴趣

```
操作 33: Mean Pool
  user_interest_ads = z.mean(dim=1)  → [4096, 48]
```

> **逐操作详解**：
>
> | 操作 | 做了什么 | 为什么这样设计 |
> |------|---------|---------------|
> | 操作 27: 拼接 cand+d_ctx | 候选 repr [B,48] 与域上下文 [B,48] 拼接为 [B,96] | 让 query 生成同时感知"候选是什么"和"当前在哪个域"。同一个候选在不同 tab 会生成不同的 query |
> | 操作 28: MLP → 4 queries | 96→64→32→192, reshape 为 [B,4,48] | 一个 MLP 同时输出 4 个 query（共享底层参数），reshape 后每个 query 48 维，与 item repr 同维 |
> | 操作 29-30: 4-query DIN attention | 每个 query 独立对 50 条历史做 DIN-style attention | 与 DIN 相同的四元组+MLP 打分，但有 4 个独立的 query，每个 query"关注"不同的历史子集 |
> | 操作 31: Masked Softmax | 每个 query 独立做 masked softmax | 确保每个兴趣向量的 attention 权重和=1，且 padding 位不参与 |
> | 操作 32: 加权聚合 → 4 tokens | einsum 聚合为 [B,4,48] | 4 个兴趣向量，每个代表用户的一个兴趣面（如"搞笑""美食""科技""运动"） |
> | 操作 33: mean_pool → 1 向量 | 4 个兴趣取平均 → [B,48] | 简化版聚合。虽然丢失了多兴趣结构，但后续 TransformerFusion 会用 interest_tokens 补回来 |
>
> **关键输出**（三路分发）：
- `user_interest_ads [4096, 48]` → 与 DIN 融合（Fuse1）
- `interest_tokens [4096, 4, 48]` → 送入 TransformerFusion（保留多兴趣结构）
- `interest_mask [4096, 4]` 全 1 → Transformer 的 padding mask（4 个 token 都有效）

---

## 6. 兴趣融合 1: DIN + PCRG

> **设计思路**：DIN 提取的是"与候选最相关的单一兴趣"，PCRG 提取的是"域感知的多面兴趣均值"。两者互补：
> - DIN 擅长精准匹配（"这个候选最像你看过的哪个"）
> - PCRG 擅长全局覆盖（"你在当前域的整体兴趣画像"）
>
> **为什么用 concat+Linear 而不是相加？** 相加假设两者同等重要，但实际上：
> - 对历史丰富的活跃用户，DIN 更准（有足够历史做精确匹配）
> - 对历史稀疏的新用户，PCRG 的域感知 query 更有用（即使历史少，域上下文能补信息）
> - Linear(96→48) 让模型自动学习加权策略

```
操作 34: concat + 投影
  concat[user_interest_din, user_interest_ads] → [4096, 96]
  Linear(96, 48) → user_interest [4096, 48]
```

> | 操作 | 做了什么 | 为什么这样设计 |
> |------|---------|---------------|
> | 操作 34: concat+Linear | 两个 48 维向量拼接后用线性层压回 48 维 | 维度不膨胀（保持 D=48），同时通过学习权重矩阵实现自适应融合。Linear 权重中前 48 列学"DIN 的贡献"，后 48 列学"PCRG 的贡献" |

---

## 7. TransformerFusion — 多兴趣精炼

> **核心问题**：PCRG 的 4 个 interest_tokens 是独立生成的——token 1 不知道 token 2 捕获了什么。这导致：
> 1. 兴趣可能重叠（4 个 token 都关注了"搞笑"，没人关注"美食"）
> 2. 缺乏相对重要性（不知道哪个兴趣对当前候选更重要）
>
> **三步解决方案**：
> - **Step 1 Self-Attention**：让 4 个 token 互相看到彼此，自动去重和分化（"你已经关注搞笑了，我去关注美食"）
> - **Step 2 Target Attention**：用候选 item 作为 query，从 4 个增强后的 token 中挑出最相关的（"这个美食视频 → 美食兴趣 token 权重最高"）
> - **Step 3 Output FFN**：再做一层非线性变换，提取更高阶的兴趣组合
>
> **为什么只对 4 个 token 做 Transformer 而不是原始 50 条历史？**
> - Self-Attention 复杂度 O(N²)：N=50 → 2500 次计算，N=4 → 16 次计算
> - 4GB GPU 限制下，N=50 的 Transformer 显存可能不够
> - 4 个 token 是 PCRG 的"浓缩兴趣"，信息密度远高于原始历史

### 7.1 Token 投影（如果维度不匹配）

```
操作 35: token_proj (Identity，因为 48==48)
  x = interest_tokens [4096, 4, 48]  → 不变

操作 36: query_proj (Identity)
  q = cand_item_repr [4096, 48]  → 不变
```

### 7.2 Step 1: Self-Attention

```
操作 37: MultiheadAttention(d_model=48, n_heads=2, batch_first=True)
  每个 head: d_k = 48/2 = 24

  Q = W_Q · x → [4096, 4, 48]  (split into 2 heads: [4096, 2, 4, 24])
  K = W_K · x → 同上
  V = W_V · x → 同上

  attn_scores = Q @ K^T / sqrt(24) → [4096, 2, 4, 4]
  attn_weights = softmax(scores)    → [4096, 2, 4, 4]
  attn_out = attn_weights @ V       → [4096, 2, 4, 24]
  concat heads → [4096, 4, 48]
  W_out → [4096, 4, 48]

操作 38: Residual + LayerNorm
  x = LayerNorm(x + Dropout(attn_out))  → [4096, 4, 48]

操作 39: FFN
  Linear(48, 96) → GELU → Dropout → Linear(96, 48)
  → ffn_out [4096, 4, 48]

操作 40: Residual + LayerNorm
  x = LayerNorm(x + Dropout(ffn_out))  → [4096, 4, 48]

操作 41: Mask 安全
  valid_mask = (~padding_mask).unsqueeze(-1)  # interest_mask 全1，不影响
  x = x * valid_mask  → [4096, 4, 48]
```

### 7.3 Step 2: Target Attention

```
操作 42: DIN-style Target Attention
  query = cand_item_repr [4096, 48]
  keys  = self-attended tokens [4096, 4, 48]
  mask  = interest_mask [4096, 4] 全 1

  q_expand = query.unsqueeze(1).expand(-1, 4, -1) → [4096, 4, 48]
  att_input = concat[q, k, q-k, q*k] → [4096, 4, 192]

  Linear(192, 64) → PReLU → Linear(64, 32) → PReLU → Linear(32, 1)
  → scores [4096, 4]
  weights = softmax(scores) → [4096, 4]

  u_target = sum(weights * tokens) → [4096, 48]
```

### 7.4 Step 3: Output FFN

```
操作 43: OutputFFN
  Linear(48, 256) → GELU → Dropout(0.1) → Linear(256, 48) → LayerNorm(48)
  → u_fused [4096, 48]
```

> **TransformerFusion 逐操作详解**：
>
> | 操作 | 做了什么 | 为什么这样设计 |
> |------|---------|---------------|
> | 操作 35-36: 投影 | token/query 维度已经是 48，Identity 跳过 | d_model=48 与 input_dim 一致时不需要投影，节省一次矩阵乘 |
> | 操作 37: MultiheadAttention | 2 头，每头 d_k=24，4×4 的注意力矩阵 | 2 头让模型从两个"视角"看 token 间关系。4×4 矩阵 = 每个 token 对其他 3 个 token 的注意力权重 |
> | 操作 38: Residual+LN | x = LN(x + dropout(attn_out)) | 标准 Pre-LN Transformer。残差保证 self-attn 不破坏原始 token 信息，LN 稳定训练 |
> | 操作 39-40: FFN+Residual+LN | 48→96→48 的两层 FFN | FFN 的升维比 2x（48→96）是轻量设计，标准 Transformer 用 4x。GELU 比 ReLU 平滑，适合小模型 |
> | 操作 41: Mask 安全 | 乘 valid_mask 清零 padding | interest_tokens 全 1 mask，这步实际不影响，但代码防御性编写 |
> | 操作 42: Target Attention | DIN-style，候选对 4 个 token 打分 | 与 DIN 相同的四元组机制，但 N=4 而不是 50。输出是加权和——当前候选最相关的兴趣占主导 |
> | 操作 43: OutputFFN | 48→256→48，GELU | 额外一层 FFN 做更高阶兴趣抽取。升维比 5.3x（48→256）让模型有更多空间学组合特征 |
>
> **Self-Attention 的实际效果**：训练后观察 attention map，通常会看到"分化"模式——每个 token 的 self-attention 权重集中在与自己不同的 token 上，说明模型在学"差异互补"而不是"相似增强"。

---

## 8. 兴趣融合 2: 主兴趣 + Transformer 输出

> **设计思路**：Fuse1 的输出是 DIN+PCRG 的朴素融合，Fuse2 再把 TransformerFusion 的精炼输出融合进来。
> 两次融合的架构完全相同（concat+Linear），但语义不同：
> - Fuse1：融合"精准匹配兴趣" + "域感知多面兴趣"
> - Fuse2：融合"朴素兴趣" + "Transformer 精炼后的结构化兴趣"
>
> 为什么分两步而不是三路一起融合？**渐进式融合** 更稳定——每次只加入一个新信号，Linear 更容易学到有意义的权重。

```
操作 44: concat + 投影
  concat[user_interest, u_fused] → [4096, 96]
  Linear(96, 48) → user_interest [4096, 48]    ← 最终兴趣表示
```

> | 操作 | 做了什么 | 为什么这样设计 |
> |------|---------|---------------|
> | 操作 44: concat+Linear | 与 Fuse1 完全相同的结构：96→48 | 保持一致性。最终 user_interest [B,48] 是经过 DIN、PCRG、TransformerFusion 三重提炼的兴趣表示，信息密度最高 |
>
> **user_interest 的旅程总结**：
> 1. DIN → 单一精准匹配兴趣 [B,48]
> 2. PCRG → 域感知多面兴趣均值 [B,48]
> 3. Fuse1 → 两者融合 [B,48]
> 4. TransformerFusion → 对 4 个 token 做自注意力+目标注意力 [B,48]
> 5. Fuse2 → 最终兴趣 [B,48]，送入 head_input

---

## 9. PersonalContextEncoder — PPNet 前置

> **核心问题**：PPNet 需要一个"条件向量"p_ctx 来生成个性化的 FiLM 参数（γ/β）。p_ctx 应该编码"这个样本属于什么类型的用户，在什么场景下"。
>
> **三流设计的原因**：
> - **场景流 (scene_ctx)**：tab/hour/dow/weekend → 16 维。决定"现在处于什么推荐环境"。直通不过 MLP，因为 embedding 本身已经有语义。
> - **用户流 (user_ctx)**：user_sparse_embs + user_dense → MLP → 64 维。决定"这个用户是什么类型的人"。过 MLP 压缩是因为原始 ~108 维太稀疏、太高维。
> - **活跃度流 (activity_ctx)**：hist_len + is_lowactive + user_active_degree → MLP → 32 维。决定"这个用户活跃不活跃"。单独建流是因为活跃度对 PPNet 的调制逻辑有特殊意义（冷启动用户需要不同的特征加权）。
>
> **为什么不直接把所有特征拼接过一个大 MLP？** 三流分别压缩后再拼接，比全量拼接更好：
> 1. 避免高维 sparse 特征淹没低维 dense 特征
> 2. 每个流有独立的压缩比（场景不压缩、用户 108→64、活跃度 9→32）
> 3. 更好的可解释性——可以分别看三个流的贡献

### 9.1 场景上下文（直通）

```
操作 45: context_embs [4096, 16]
  来自操作 11 的 tab/hour/dow/weekend embedding concat
  → scene_ctx [4096, 16]
```

### 9.2 用户上下文

```
操作 46: 拼接用户 sparse + dense
  user_sparse_embs [4096, ~104] + user_dense [4096, 4]
  concat → [4096, ~108]

操作 47: MLP 投影
  Linear(108, 64) → ReLU → Linear(64, 64)
  → user_ctx [4096, 64]
```

### 9.3 活跃度上下文

```
操作 48: 活跃度特征
  hist_len_norm = hist_len / 50.0 → [4096, 1]
  is_lowactive_emb: Embedding(3, 4) → [4096, 4]
  user_active_degree_emb: Embedding(5, 4) → [4096, 4]
  concat → [4096, 9]

操作 49: MLP 投影
  Linear(9, 32) → ReLU → Linear(32, 32)
  → activity_ctx [4096, 32]
```

### 9.4 拼接

```
操作 50: concat + LayerNorm
  p_ctx = concat[scene_ctx, user_ctx, activity_ctx]
        = concat[[4096,16], [4096,64], [4096,32]]
  → p_ctx [4096, 112]
  LayerNorm(112) → p_ctx [4096, 112]
```

> **逐操作详解**：
>
> | 操作 | 做了什么 | 为什么这样设计 |
> |------|---------|---------------|
> | 操作 45: scene_ctx 直通 | context_embs [B,16] 直接作为场景部分 | 4 个场景 embedding 各 4 维，已经足够紧凑，不需要再压缩 |
> | 操作 46-47: user MLP | sparse_embs [B,~104] + dense [B,4] 拼接后 MLP(108→64→64) | 108 维包含 26 个 sparse 特征的 embedding，信息冗余大。MLP 压缩到 64 维，保留核心用户画像 |
> | 操作 48-49: activity MLP | hist_len_norm + is_lowactive_emb + active_degree_emb → MLP(9→32→32) | hist_len/50.0 归一化到 [0,1]，与 embedding 拼接后过 MLP。32 维足够编码活跃度等级 |
> | 操作 50: concat+LayerNorm | 三流拼接 16+64+32=112，LayerNorm 归一化 | LayerNorm 关键——确保 p_ctx 的 scale 一致，避免 PPNet 的 γ/β 生成器输入分布不稳定 |
>
> **p_ctx [B,112] 的语义解读**：
> - 前 16 维：场景（"发现页、周五晚上 10 点"）
> - 中间 64 维：用户画像（"活跃女性用户，关注 50+人，注册 3 年"）
> - 后 32 维：活跃度（"历史 45 条，高活跃期"）
> - PPNet 读取这个 112 维向量来决定"对这个特定用户在这个特定场景下，应该加重哪些特征"

---

## 10. Head Input 构建

> **设计思路**：Head Input 是模型"信息汇聚点"——把之前所有模块的输出拼接成一个 flat 向量，作为预测头的输入。
>
> **7 个语义块的安排顺序**不是随意的：
> 1. `user_interest` 排第一位——最重要的信号（经过 DIN+PCRG+TransformerFusion 三重提炼）
> 2. `cand_repr` 排第二——候选本身的表达
> 3. `user_profile` 排第三——用户画像，信息量大但与候选关系较远
> 4. `user_dense` 排第四——数值型用户特征
> 5. `context_embs` 排第五——场景上下文
> 6. `candidate_side_embs` 排第六——候选侧额外属性（时长、分桶等）
> 7. `cand_dense` 排最后——候选的连续值统计特征（仅 full track 有）
>
> **feature_slices 的作用**：记录每个语义块的偏移量，供 PPNet 的 GroupWiseFiLM 和 MBCNet 的 FGC 按组切分。

```
操作 51: 拼接所有特征块

  parts 的精确顺序和维度:
  ┌──────────────────────────┬─────────┬────────────┐
  │ 语义块                    │ 维度     │ 偏移量      │
  ├──────────────────────────┼─────────┼────────────┤
  │ user_interest             │ 48      │ [0, 48)    │
  │ cand_repr                 │ 48      │ [48, 96)   │
  │ user_profile_sparse_embs  │ ~104    │ [96, 200)  │
  │ user_dense                │ 4       │ [200, 204) │
  │ context_embs              │ 16      │ [204, 220) │
  │ candidate_side_embs       │ ~32     │ [220, 252) │
  │ cand_dense (full track)   │ 5       │ [252, 257) │
  └──────────────────────────┴─────────┴────────────┘

  head_input = torch.cat(parts, dim=-1) → [4096, ~264 or ~269]

  feature_slices 记录: {
    "user_interest": (0, 48),
    "cand_repr": (48, 96),
    "user_profile_sparse_embs": (96, 200),
    "user_dense": (200, 204),
    "context_embs": (204, 220),
    "candidate_side_embs": (220, 252),
  }
```

> **各语义块详解**：
>
> | 语义块 | 维度 | 来源 | 编码了什么信息 |
> |-------|------|------|--------------|
> | user_interest | 48 | Fuse2 输出 | 经 DIN+PCRG+TransformerFusion 三重提炼的最终兴趣表示。携带了"用户最近关注什么+在当前域最关注什么" |
> | cand_repr | 48 | HashEmb(video)+HashEmb(author) | 候选视频的核心 ID 表达。"这是一个什么视频，谁拍的" |
> | user_profile_sparse_embs | ~104 | 26 个 sparse 特征的 Embedding concat | 用户的离散画像：活跃度/身份/社交/注册时长/18 个匿名标签 |
> | user_dense | 4 | log1p(follow/fans/friend/register) | 用户的连续值画像，与 sparse 版互补 |
> | context_embs | 16 | tab/hour/dow/weekend 的 Embedding concat | 当前场景上下文。注意这里用的是原始 embedding 而非 DomainContextEncoder 的输出 |
> | candidate_side_embs | ~32 | 8 个候选 side 特征的 Embedding concat | 候选的附加属性：视频类型/时长/先验 CTR 分桶等 |
> | cand_dense (full track) | 5 | pre_ctr/lv_rate/like_rate/play_ratio/show_log | 候选的连续值统计，仅 full track 有此列。sparse track 无此块 |
>
> **full track vs sparse track 的维度差异**：full track 有 cand_dense 5 维，总维度 ~269；sparse track 没有，总维度 ~264。

---

## 11. PPNet — 个性化调制

> **核心问题**：传统模型所有用户共享同一套 MLP 权重。但不同用户应该有不同的特征加权策略：
> - 活跃用户历史丰富 → 应该更信任 user_interest 块
> - 冷启动用户历史不足 → 应该更依赖 context + cand_side 块
> - 高端设备用户 → 可能更关注视频质量（cand_side 中的时长/播放比例）
>
> **PPNet 的解决方案：FiLM (Feature-wise Linear Modulation)**
> 公式：`x_g' = (1 + γ_g) * x_g + β_g`
> - γ_g：缩放因子（每组一个标量），控制"这组特征放大还是缩小"
> - β_g：偏移因子（每组一个标量），控制"这组特征整体偏移"
> - 总共只有 12 个参数（6 组 × 2），极其轻量
>
> **零初始化 (Zero-Init) 的妙处**：
> - 训练初期 γ≈0, β≈0，FiLM 退化为恒等映射 x'=x
> - 等于 PPNet "不存在"，不干扰主干网络的正常训练
> - 随着训练推进，γ/β 慢慢学到有意义的值，渐进式引入个性化
> - 这是一种"warm start"技巧，避免新模块一上来就破坏已有的梯度流

### 11.1 生成 6 组 γ/β

```
操作 52: Backbone MLP
  p_ctx [4096, 112]
  Linear(112, 64) → LayerNorm(64) → ReLU → Dropout(0.1)
  Linear(64, 32)  → LayerNorm(32) → ReLU → Dropout(0.1)
  → [4096, 32]

操作 53: 输出投影（零初始化权重!）
  Linear(32, 12)  ← 6 组 × 2 (γ + β)
  → params [4096, 12]
  gamma, beta = chunk(params, 2) → 各 [4096, 6]

训练初期: 因为零初始化, gamma ≈ 0, beta ≈ 0
```

### 11.2 分组 FiLM 调制

```
操作 54: 对 6 个语义组分别调制

  组 0 (user_interest, 0:48):
    x_0' = (1 + gamma[:,0].unsqueeze(-1)) * x[:,0:48] + beta[:,0].unsqueeze(-1)
    → [4096, 48]

  组 1 (cand_repr, 48:96):
    x_1' = (1 + gamma[:,1]) * x[:,48:96] + beta[:,1]
    → [4096, 48]

  组 2 (user_profile, 96:200):
    x_2' = (1 + gamma[:,2]) * x[:,96:200] + beta[:,2]
    → [4096, 104]

  组 3 (user_dense, 200:204):
    x_3' = (1 + gamma[:,3]) * x[:,200:204] + beta[:,3]
    → [4096, 4]

  组 4 (context, 204:220):
    x_4' = (1 + gamma[:,4]) * x[:,204:220] + beta[:,4]
    → [4096, 16]

  组 5 (cand_side, 220:252):
    x_5' = (1 + gamma[:,5]) * x[:,220:252] + beta[:,5]
    → [4096, 32]

  concat → [4096, ~252]
  Dropout(0.1) → LayerNorm(252)

  cand_dense 部分 (252:257) 不在 group_slices 中，通过 cat 拼回

  → head_input [4096, ~264/269]  (调制后)
```

> **逐操作详解**：
>
> | 操作 | 做了什么 | 为什么这样设计 |
> |------|---------|---------------|
> | 操作 52: Backbone MLP | p_ctx [B,112] → 64 → 32，带 LN+ReLU+Dropout | 两层 MLP 逐步压缩：112→64→32。LayerNorm 稳定梯度，Dropout 防止 PPNet 过拟合到特定用户 |
> | 操作 53: 输出投影 | Linear(32→12)，**权重全零初始化** | 12 = 6组×2(γ+β)。零初始化是关键——训练开始时 γ=β=0，FiLM 无效，不破坏主干训练 |
> | 操作 54: 分组 FiLM | 每组：x_g' = (1+γ_g)*x_g + β_g | γ_g 是标量但广播到整组所有维度。例如 γ[0] 作用于 user_interest 的全部 48 维 |
>
> **6 个组的 FiLM 语义解读**：
>
> | 组 | 特征块 | γ > 0 的含义 | γ < 0 的含义 | 典型场景 |
> |---|--------|------------|------------|---------|
> | 0 | user_interest (0:48) | 放大兴趣信号 | 抑制兴趣（不信任历史） | 活跃用户 γ>0，冷启动 γ<0 |
> | 1 | cand_repr (48:96) | 放大候选特征 | 抑制候选特征 | 热门视频可能 γ<0（不需要靠 ID 区分） |
> | 2 | user_profile (96:200) | 放大用户画像 | 抑制画像 | 画像丰富的用户 γ>0 |
> | 3 | user_dense (200:204) | 放大数值特征 | 抑制数值特征 | 大 V 用户（fans 多）γ>0 |
> | 4 | context (204:220) | 放大场景信号 | 抑制场景 | 小众 tab 可能 γ>0（需要更多场景信息辅助） |
> | 5 | cand_side (220:252) | 放大候选属性 | 抑制属性 | 冷启动视频 γ>0（需要靠时长/类型等属性判断） |
>
> **cand_dense 不参与 FiLM**：它不在 feature_slices 中（只有 full track 有），PPNet 调制后通过 cat 拼回去。

---

## 12. MBCNet — 三分支预测头

> **核心问题**：传统单路 MLP 预测头只能学"隐式"特征交叉（靠多层非线性逐步组合），对低阶交叉（如"用户画像×候选类型"）学习效率低。
>
> **MBCNet 的三分支设计**：
> - **FGC (Fine-Grained Cross)**：按语义组做组内显式交叉 → 捕捉"同组特征内部"的低阶交叉（如"关注数×粉丝数"）
> - **LowRank Cross**：全局低秩交叉 → 捕捉"跨组特征之间"的二阶交叉（如"用户兴趣×候选属性"），用 rank=16 压缩避免参数爆炸
> - **Deep MLP**：标准多层感知机 → 捕捉任意阶隐式交叉，作为兜底
>
> **为什么不只用 DCN-V2？** DCN-V2 做全局交叉不区分语义组，"用户画像×用户画像"的交叉没有意义（都是同一个用户的属性，自交叉信息量低）。FGC 按语义组切分后，交叉更精准。
>
> **三分支的互补性**：
> - FGC 善于低阶、局部、可解释的交叉
> - LowRank 善于全局二阶交叉，但受 rank 限制不能学太复杂的模式
> - Deep 善于高阶、全局、不可解释的交叉
> - 三者 concat 后过融合 MLP，模型自动选择最有用的分支输出

### 12.1 FGC 分支（分组内显式交叉）

```
操作 55: 按组切分 head_input
  g_0 = head_input[:, 0:48]     # user_interest
  g_1 = head_input[:, 48:96]    # cand_repr
  g_2 = head_input[:, 96:200]   # user_profile
  g_3 = head_input[:, 200:204]  # user_dense
  g_4 = head_input[:, 204:220]  # context
  g_5 = head_input[:, 220:252]  # cand_side

操作 56: 每组独立做 CrossNet-v1 (1 layer)
  以 g_0 (48维) 为例:
    scale = Linear(48, 1)(g_0) → [4096, 1]     # 标量打分 w^T * g
    cross = scale * g_0        → [4096, 48]     # 广播乘
    g_0' = g_0 + Dropout(cross)                  # 残差
    g_0' = LayerNorm(48)(g_0')  → [4096, 48]

  每组都有独立的 Linear + LayerNorm 参数

操作 57: 拼接所有组
  fgc_out = concat[g_0', g_1', g_2', g_3', g_4', g_5']
  → [4096, ~264]
```

### 12.2 LowRank Cross 分支（全局低秩交叉）

```
操作 58: 第 1 层 LowRankCrossLayer (rank=16)
  输入: x = head_input [4096, ~264]

  u = Linear(264, 16, bias=False)(x)  → [4096, 16]    # U 投影
  v = Linear(264, 16, bias=False)(x)  → [4096, 16]    # V 投影
  cross = Linear(16, 264, bias=True)(u * v) → [4096, 264]  # P(u⊙v)
  x' = x + Dropout(cross)
  x' = LayerNorm(264)(x')  → [4096, 264]

操作 59: 第 2 层 LowRankCrossLayer (同结构)
  输入: x' → 同样的 U,V,P 操作（独立参数）
  → lowrank_out [4096, ~264]
```

### 12.3 Deep 分支（MLP 隐式交叉）

```
操作 60: MLP
  输入: head_input [4096, ~264]
  Linear(264, 256) → GELU → Dropout(0.1)
  Linear(256, 128) → GELU → Dropout(0.1)
  Linear(128, 64)  → GELU → Dropout(0.1)
  → deep_out [4096, 64]
```

### 12.4 分支融合

```
操作 61: 各分支投影到统一维度
  fgc_proj    = Linear(264, 128)(fgc_out)    → [4096, 128]
  lowrank_proj = Linear(264, 128)(lowrank_out) → [4096, 128]
  deep_proj   = Linear(64, 128)(deep_out)    → [4096, 128]

操作 62: Concat + 融合 MLP
  fused = concat[fgc_proj, lowrank_proj, deep_proj] → [4096, 384]
  Dropout(0.1)
  Linear(384, 128) → ReLU → Dropout(0.1)
  Linear(128, 64)  → ReLU → Dropout(0.1)
  Linear(64, 1)
  → logit [4096]
```

> **逐操作详解**：
>
> | 操作 | 做了什么 | 为什么这样设计 |
> |------|---------|---------------|
> | 操作 55-57: FGC | 6 组各自做 CrossNet-v1 (1 layer) | CrossNet-v1 公式：`x' = x + x * (w^T * x) + b`。`w^T*x` 是标量打分（"这组特征的整体激活强度"），乘回 x 实现二阶自交叉。残差+LN 保证稳定 |
> | 操作 58-59: LowRank Cross (2层) | U(264→16) * V(264→16) → P(16→264) | 低秩分解：完整二阶交叉需要 264×264 的矩阵（70K 参数），低秩 rank=16 只需 264×16×3 ≈ 12K 参数。u⊙v 是 Hadamard 积，捕捉两两维度的交互 |
> | 操作 60: Deep MLP | 264→256→128→64，3 层 GELU+Dropout | 逐层压缩（264→256 几乎不压，256→128→64 逐步抽取高阶特征）。GELU 比 ReLU 平滑，对小 batch 更友好 |
> | 操作 61: 各分支投影 | FGC/LowRank 264→128, Deep 64→128 | 统一到 128 维，让三个分支在同一语义空间。不统一维度直接 concat 会让维度大的分支主导 |
> | 操作 62: 融合 MLP → logit | 384→128→64→1 | 最终 MLP 从三分支的 concat 表示中抽取预测信号。输出是 1 个标量 logit（未经 sigmoid），送入 BCEWithLogitsLoss |
>
> **FGC 的 6 组交叉具体含义**：
>
> | 组 | 特征块 | 组内交叉捕捉的交互 |
> |---|--------|-------------------|
> | g0 | user_interest (48维) | 兴趣向量内部维度的二阶组合（如"搞笑兴趣维度 × 美食兴趣维度"） |
> | g1 | cand_repr (48维) | 视频 ID embedding 内部的组合（如"video 特征 × author 特征"） |
> | g2 | user_profile (104维) | 用户画像内部的组合（如"活跃度 × 是否创作者 × 粉丝段"） |
> | g3 | user_dense (4维) | 连续值之间的交互（如"关注数 × 粉丝数"→ 社交影响力指标） |
> | g4 | context (16维) | 场景特征的组合（如"tab × 小时"→ "发现页+深夜"的特定模式） |
> | g5 | cand_side (32维) | 候选属性的组合（如"时长 × CTR 分桶"→ "长视频+高 CTR"的优质内容信号） |

---

## 13. Loss 计算

> **设计思路**：使用 BCEWithLogitsLoss（二分类交叉熵+内置 sigmoid），预测"用户是否会完播这个视频"。
>
> **为什么用 BCEWithLogitsLoss 而不是 BCELoss + sigmoid？**
> - 数值稳定性：BCEWithLogitsLoss 内部用 log-sum-exp 技巧，避免 sigmoid 输出接近 0/1 时 log 溢出
> - 与 AMP (bfloat16) 配合更好：logit 精度损失比 probability 小

```
操作 63: BCEWithLogitsLoss
  labels = batch["label_long_view"].float() → [4096]
  loss = BCEWithLogitsLoss(logit, labels)
  loss = loss / gradient_accum_steps

  反向传播: scaler.scale(loss).backward()
```

> | 操作 | 做了什么 | 为什么这样设计 |
> |------|---------|---------------|
> | label 转 float | int64 → float32 | BCEWithLogitsLoss 要求 target 为浮点数 |
> | BCEWithLogitsLoss | 计算 -[y*log(σ(z)) + (1-y)*log(1-σ(z))] | σ(z) 是 sigmoid，y 是 label。内部用数值稳定的实现 |
> | loss / grad_accum | 除以梯度累积步数 | 等效于更大 batch_size。实际 batch=4096，如果 accum=2 则等效 batch=8192 |
> | scaler.scale(loss).backward() | AMP 混合精度反向传播 | GradScaler 动态调整 loss 缩放倍数，防止 bfloat16 下梯度 underflow |
>
> **完整训练循环**：每个 step 执行 forward → loss → backward → (累积 N 步后) optimizer.step() → scaler.update()。使用 AdamW 优化器，学习率 1e-3，weight_decay=1e-5。

---

## 全链路张量流总结图

```
                    cand_video_id                hist_video_id
                    cand_author_id               hist_author_id
                         │                            │
                    ┌────┴────┐                  ┌────┴────┐
                    │ Hash    │                  │ Hash    │  ← 共享权重
                    │ Embed   │                  │ Embed   │
                    └────┬────┘                  └────┬────┘
                         │                            │
                    cand_item_repr              hist_item_repr
                    [B, 48]                     [B, 50, 48]
                         │                            │
                         │                     + delta_t/play_ratio/tab
                         │                     残差投影加入
                         │                            │
                    ┌────┴────────────────────────────┤
                    │                                  │
              tab+hour+dow                             │
                    │                                  │
              DomainContext                            │
              d_ctx [B,48]                             │
                    │                                  │
                    ├─────────────────┐                │
                    │                 │                │
                    │           PSRG(hist, d_ctx_seq)  │
                    │                 │                │
                    │           hist_for_attn [B,50,48]│
                    │                 │                │
                    │        ┌───────┴────────┐       │
                    │        │                │       │
                    │   DIN Attention     PCRG(4 query)│
                    │   [B,48]           [B,48] + tokens[B,4,48]
                    │        │                │
                    │        └──── Fuse1 ─────┘
                    │              │
                    │         user_interest [B,48]
                    │              │
                    │         TransformerFusion
                    │         (self-attn→target-attn→FFN)
                    │              │
                    │         Fuse2 → user_interest [B,48] (最终)
                    │              │
                    │    ┌─────────┼──────────────────────┐
                    │    │         │                      │
                    │    │    ┌────┴────┐                 │
                    │    │    │ Head    │                 │
                    │    │    │ Input   │ + cand_repr     │
                    │    │    │ Concat  │ + user_sparse   │
                    │    │    │         │ + user_dense    │
                    │    │    │         │ + context       │
                    │    │    │         │ + cand_side     │
                    │    │    │         │ + cand_dense    │
                    │    │    └────┬────┘                 │
                    │    │         │ [B, ~269]            │
                    │    │         │                      │
                    │    │    PersonalContext              │
                    │    │    p_ctx [B,112]  ──── PPNet ──┘
                    │    │         │           FiLM调制
                    │    │         │
                    │    │    head_input (调制后) [B, ~269]
                    │    │         │
                    │    │    ┌────┴─────────────────┐
                    │    │    │                      │
                    │    │   FGC         LowRank    Deep
                    │    │   (组内交叉)   (全局r=16)  (MLP)
                    │    │    │           │          │
                    │    │   proj→128   proj→128   proj→128
                    │    │    │           │          │
                    │    │    └─── concat [B,384] ──┘
                    │    │              │
                    │    │         MLP [128,64] → Linear → logit [B]
                    │    │              │
                    │    │         BCEWithLogitsLoss
```

---

## 各模块与 ADS 原论文的对应关系

```
ADS 原论文                    本项目实现                    优化了什么
──────────                    ────────────                ──────────
SRG (全量参数生成)             → PSRG-lite (门控残差)       轻量化：不生成完整MLP权重
                                                          而是用 gate*delta 残差更新

CRG (多query生成+聚合)        → PCRG-lite (mean_pool)      轻量化：mean_pool替代复杂聚合

--- 以下是本项目的原创优化 ---

(ADS 没有)                   → TransformerFusion           优化 PCRG 输出的多兴趣融合
                               自注意力 + target attention   让独立生成的token互相感知

(ADS 用简单 MLP)             → MBCNet 三分支               优化预测头的特征交叉
                               FGC + LowRank + Deep         显式低阶 + 全局二阶 + 隐式高阶

(ADS 所有用户共享参数)        → PPNet GroupWiseFiLM          优化个性化
                               p_ctx → 分组 γ/β 调制        不同用户动态加权特征组
```

---

## 为什么优化 ADS 而不是选择别的基座模型

### 选 ADS 做基座的原因

| 理由 | 说明 |
|------|------|
| **问题匹配** | ADS 是专门为跨域推荐设计的，SRG/CRG 直接对标 KuaiRand 的 15-tab 跨域场景 |
| **模块化** | SRG 和 CRG 都是独立模块，可以 plug-in 到 DIN 框架，不需要改变整个架构 |
| **轻量** | ADS-lite 的参数增量很小（~2M），适合 4GB GPU 约束 |
| **增益空间清晰** | ADS 解决了「怎么提取兴趣」，但「怎么用好兴趣」还有明确的优化空间 |

### 不选其他模型的原因

| 备选 | 不选的理由 |
|------|----------|
| **DIEN** | GRU 假设等步长序列演化，但 KuaiRand 的正交互间隔极不规律（1分钟到3天），GRU 的时序建模假设不成立。PSRG 的门控残差 + delta_t 分桶更鲁棒 |
| **SIM** | 解决超长历史（10000+）的检索问题，本项目 max_hist=50，不需要两阶段检索 |
| **MIND (Capsule)** | Multi-Interest 用 dynamic routing 提取多兴趣，但 capsule routing 训练不稳定，且不感知域上下文。PCRG 的 query 生成天然融入 d_ctx |
| **STAR** | 为每个域维护独立参数，15 个 tab 需要 15 份参数副本，4GB GPU 装不下。PPNet 的 FiLM 只生成 12 个标量（6 组 γ+β），参数量小 3 个数量级 |
| **PLE** | 解决多任务学习中的跷跷板效应（CTR 和 CVR 冲突），本项目是单任务（long_view），不需要 |
| **DCN-V2** | 全局交叉不区分语义组，「用户画像×用户画像」的交叉无意义。MBCNet 的 FGC 分组交叉更精准 |
| **全 Transformer** | 序列长度只有 50，不需要 Transformer 的长程依赖能力。而且 4GB GPU 跑全 Transformer 显存不够。只在 4 个 interest token 上用 1 层 Transformer 是性价比最高的选择 |

### 优化方向的选择逻辑

```
实验数据:
  ADS (PSRG+PCRG) 本身: test_rnd +1.73pt  → 占总提升的 60%
  后续三个优化模块:      test_rnd +1.17pt  → 占总提升的 40%

结论: ADS 的 SRG+CRG 已经是最优的序列建模方案
      瓶颈不在「怎么提取兴趣」，而在下游三个方向:

      ┌─ 瓶颈1: 多兴趣融合质量差 ─→ TransformerFusion (+0.47pt)
      │    PCRG 的 4 个 token 独立生成、mean_pool 丢失结构信息
      │
      ├─ 瓶颈2: 特征交叉方式单一 ─→ MBCNet (+0.30pt)
      │    单路 MLP 对低阶交叉学习效率低
      │
      └─ 瓶颈3: 缺乏个性化差异 ─→ PPNet (+0.40pt)
           所有用户共享同一套 MLP 权重
```
