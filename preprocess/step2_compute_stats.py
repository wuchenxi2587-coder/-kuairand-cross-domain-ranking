#!/usr/bin/env python3
"""
Step 2: 计算物品冻结统计量（无泄露）
仅用 log_standard_4_08_to_4_21（pre期）数据计算统计量
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = "/111111/newproject/data/KuaiRand-27K"
OUTPUT_DIR = "/111111/newproject/output"

def main():
    logger.info("=" * 60)
    logger.info("Step 2: Computing Item Statistics (Leakage-Free)")
    logger.info("=" * 60)

    # 1. 读取pre期数据（4/08-4/21）
    logger.info("Loading pre-period logs (4/08-4/21)...")
    NEEDED_COLS = ['video_id', 'is_click', 'long_view', 'is_like', 'play_time_ms', 'duration_ms']
    log_pre1 = pd.read_csv(
        f"{DATA_DIR}/log_standard_4_08_to_4_21_27k_part1.csv",
        usecols=NEEDED_COLS
    )
    log_pre2 = pd.read_csv(
        f"{DATA_DIR}/log_standard_4_08_to_4_21_27k_part2.csv",
        usecols=NEEDED_COLS
    )
    log_pre = pd.concat([log_pre1, log_pre2], ignore_index=True)
    del log_pre1, log_pre2

    logger.info(f"Pre-period logs: {len(log_pre)} rows")

    # 2. 计算统计量
    logger.info("Computing per-video statistics...")

    # 播放比例
    log_pre['play_ratio'] = (log_pre['play_time_ms'] / log_pre['duration_ms'].clip(lower=1)).clip(upper=3.0)

    stats = log_pre.groupby('video_id').agg(
        pre_show_cnt=('video_id', 'count'),
        pre_ctr=('is_click', 'mean'),
        pre_lv_rate=('long_view', 'mean'),
        pre_like_rate=('is_like', 'mean'),
        pre_play_ratio=('play_ratio', 'mean'),
    ).reset_index()

    # 3. 方案C：智能分桶 + Dense双版本
    logger.info("Computing features (Smart Bucketing + Dense dual version)...")

    def smart_bucket(values, n_buckets=5, zero_special=True):
        """
        智能分桶：根据数据分布特点动态选择分桶策略

        Args:
            values: 输入数据
            n_buckets: 目标桶数
            zero_special: 是否对0值特殊处理（0单独成桶）
        """
        values = values.fillna(0)

        # 检查0值比例
        zero_ratio = (values == 0).mean()

        # 如果0值超过50%且zero_special=True，0单独一桶
        if zero_special and zero_ratio > 0.3:
            logger.info(f"  Zero ratio {zero_ratio:.2%}, using special 0-bucket strategy")
            buckets = pd.Series(0, index=values.index, dtype=int)  # 0默认PAD
            non_zero = values[values > 0]

            if len(non_zero) > 0:
                # 非0值等频分剩余桶
                ranks = non_zero.rank(method='min', pct=True)
                # 非0值分到 1~(n_buckets-1) 或 1~n_buckets
                actual_buckets = min(n_buckets - 1, 4) if n_buckets > 2 else n_buckets
                non_zero_buckets = np.ceil(ranks * actual_buckets).clip(1, actual_buckets).astype(int)
                buckets.loc[non_zero.index] = non_zero_buckets
            return buckets

        # 常规等频分桶
        unique_vals = values.nunique()
        if unique_vals < n_buckets:
            # 值太少，简单映射
            return pd.Series(1, index=values.index, dtype=int)

        ranks = values.rank(method='min', pct=True)
        return np.ceil(ranks * n_buckets).clip(1, n_buckets).astype(int)

    # 3.1 原始值作为Dense特征（不分桶）
    logger.info("Creating dense features (raw values)...")
    stats['pre_ctr_raw'] = stats['pre_ctr'].astype(np.float32)
    stats['pre_lv_rate_raw'] = stats['pre_lv_rate'].astype(np.float32)
    stats['pre_like_rate_raw'] = stats['pre_like_rate'].astype(np.float32)
    stats['pre_play_ratio_raw'] = stats['pre_play_ratio'].astype(np.float32)
    stats['pre_show_log'] = np.log1p(stats['pre_show_cnt']).astype(np.float32)

    # 3.2 智能分桶作为Sparse特征
    logger.info("Creating sparse bucket features (smart bucketing)...")

    # CTR - 极度长尾，大量0值，3桶（0/低/高）
    logger.info("  Bucketing pre_ctr (extreme long-tail, many zeros)...")
    stats['pre_ctr_bucket'] = smart_bucket(stats['pre_ctr'], n_buckets=3, zero_special=True)

    # 完播率 - 两极化（0或1），3桶
    logger.info("  Bucketing pre_lv_rate (bipolar distribution)...")
    stats['pre_lv_rate_bucket'] = smart_bucket(stats['pre_lv_rate'], n_buckets=3, zero_special=True)

    # 点赞率 - 极度长尾，99%为0，3桶
    logger.info("  Bucketing pre_like_rate (extreme sparse)...")
    stats['pre_like_rate_bucket'] = smart_bucket(stats['pre_like_rate'], n_buckets=3, zero_special=True)

    # 播放比例 - 相对均匀（0-3），5桶
    logger.info("  Bucketing pre_play_ratio (relatively uniform)...")
    stats['pre_play_ratio_bucket'] = smart_bucket(stats['pre_play_ratio'], n_buckets=5, zero_special=False)

    # 曝光数 - 极度长尾，用对数刻度分桶避免大量1次曝光视频挤占同一桶
    logger.info("  Bucketing pre_show_cnt (log-scale for count feature)...")
    log_show = np.log1p(stats['pre_show_cnt'])
    # 新边界：log(3)=1.10, log(11)=2.48, log(51)=3.93, log(201)=5.30
    # 桶语义：0=冷启动, 1=1次, 2=2-9次, 3=10-49次, 4=50-199次, 5=≥200次
    log_bounds = [np.log1p(2), np.log1p(10), np.log1p(50), np.log1p(200)]
    stats['pre_show_bucket'] = (np.digitize(log_show, log_bounds) + 1).astype(np.int8)
    # 0次（如果存在）保持为0(PAD)
    stats.loc[stats['pre_show_cnt'] == 0, 'pre_show_bucket'] = 0

    logger.info("Feature distributions:")
    bucket_cols = [c for c in stats.columns if '_bucket' in c]
    for col in bucket_cols:
        dist = stats[col].value_counts().sort_index()
        logger.info(f"  {col} (n_buckets={dist.shape[0]}):\n{dist}")

    # 5. 转换类型（bucket用int8，dense用float32）
    for col in [c for c in stats.columns if '_bucket' in c]:
        stats[col] = stats[col].astype(np.int8)

    # 修复字段名不一致：step3 读取 'pre_log_show_cnt'，step2 生成的是 'pre_show_log'
    # 两列内容完全相同，都是 log1p(pre_show_cnt)，保留两个名字确保兼容
    stats['pre_log_show_cnt'] = stats['pre_show_log']

    logger.info(f"\nStatistics computed for {len(stats)} videos")
    logger.info(f"Columns: {list(stats.columns)}")

    # 6. 保存
    logger.info("Saving item statistics...")
    stats.to_parquet(f"{OUTPUT_DIR}/item_statistics.parquet", index=False)

    logger.info("Step 2 completed!")
    logger.info(f"Columns: {list(stats.columns)}")

if __name__ == "__main__":
    main()
