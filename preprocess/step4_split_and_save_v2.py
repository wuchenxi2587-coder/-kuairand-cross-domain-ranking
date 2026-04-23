#!/usr/bin/env python3
"""
Step 4: 数据集切分与Meta文件生成 (修正版)
严格匹配 configs/train_din_mem16gb.yaml 和 README 的样本比例
"""

import pandas as pd
import numpy as np
import json
import logging
import os
import gc
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = "output"

# 严格匹配 README 的样本比例：train 70%, val 10%, test 20%
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
# test_ratio = 0.20 (剩余)

def generate_field_schema():
    """生成严格匹配 train_din_mem16gb.yaml 的 field_schema.json"""
    schema = {
        "label_col": "label_long_view",  # config里明确写的
        "user_id_col": "user_id_raw",
        "max_hist_len": 50,
        "hist_seq_cols": ["hist_video_id", "hist_author_id"],
        "hist_mask_col": "hist_mask",
        "hist_len_col": "hist_len",
        "optional_hist_seq_cols": [
            "hist_delta_t_bucket",
            "hist_play_ratio_bucket",
            "hist_tab"
        ],
        "cand_cols": {
            "video_id": "cand_video_id",
            "author_id": "cand_author_id",
            "video_type": "cand_video_type",
            "upload_type": "cand_upload_type",
            "duration_bucket": "cand_video_duration_bucket",
            # 方案C：Sparse分桶特征（长尾特征：CTR/LV/Like等）
            "pre_ctr_bucket": "cand_pre_ctr_bucket",
            "pre_lv_rate_bucket": "cand_pre_lv_rate_bucket",
            "pre_like_rate_bucket": "cand_pre_like_rate_bucket",
            "pre_play_ratio_bucket": "cand_pre_play_ratio_bucket",
            "pre_show_bucket": "cand_pre_show_bucket"
        },
        "context_sparse_cols": [
            "tab",
            "hour_of_day",
            "day_of_week",
            "is_weekend"
        ],
        "user_sparse_cols": [
            "user_active_degree",
            "is_lowactive_period",
            "is_live_streamer",
            "is_video_author",
            "follow_user_num_range",
            "fans_user_num_range",
            "friend_user_num_range",
            "register_days_range",
            "onehot_feat0", "onehot_feat1", "onehot_feat2", "onehot_feat3",
            "onehot_feat4", "onehot_feat5", "onehot_feat6", "onehot_feat7",
            "onehot_feat8", "onehot_feat9", "onehot_feat10", "onehot_feat11",
            "onehot_feat12", "onehot_feat13", "onehot_feat14", "onehot_feat15",
            "onehot_feat16", "onehot_feat17"
        ],
        "user_dense_cols": [
            "log1p_follow_user_num",
            "log1p_fans_user_num",
            "log1p_friend_user_num",
            "log1p_register_days"
        ],
        "cand_dense_cols": [
            # 方案C：Dense原始值特征（均匀分布或需要精细数值）
            "cand_pre_log_show_cnt",  # 原有的log1p曝光数
            "cand_pre_ctr",           # CTR原始值 0-1
            "cand_pre_lv_rate",       # 完播率原始值
            "cand_pre_like_rate",     # 点赞率原始值
            "cand_pre_play_ratio",    # 播放比例原始值
            "cand_pre_show_log"       # log1p(曝光数)
        ],
        "meta_cols": ["user_id_raw", "meta_is_rand", "sample_time_ms", "meta_log_source"]
    }
    return schema

def compute_vocab_sizes():
    """计算各字段的vocab大小（严格匹配config的字段）"""
    vocab_sizes = {}

    # 从词表文件读取
    vocab_files = {
        'video_id': 'video_id.json',
        'author_id': 'author_id.json',
        'upload_type': 'upload_type.json',
        'tab': 'tab.json',
    }

    for name, filename in vocab_files.items():
        try:
            with open(f"{OUTPUT_DIR}/vocabs/{filename}", 'r') as f:
                vocab = json.load(f)
                vocab_sizes[name] = len(vocab)
        except:
            logger.warning(f"Vocab file not found: {filename}")
            vocab_sizes[name] = 100  # 默认值

    # 固定值
    vocab_sizes['video_type'] = 3  # PAD, NORMAL, AD
    vocab_sizes['user_active_degree'] = 5
    vocab_sizes['video_duration_bucket'] = 6
    vocab_sizes['hour_of_day'] = 25
    vocab_sizes['day_of_week'] = 8
    vocab_sizes['is_weekend'] = 3

    # 方案C：Sparse分桶特征 vocab大小（动态读取实际桶数，避免硬编码不一致）
    try:
        _stats = pd.read_parquet(
            f"{OUTPUT_DIR}/item_statistics.parquet",
            columns=['pre_ctr_bucket', 'pre_lv_rate_bucket', 'pre_like_rate_bucket',
                     'pre_play_ratio_bucket', 'pre_show_bucket']
        )
        # +1 因为桶编号从0(PAD)开始，vocab_size = max_bucket_id + 1
        vocab_sizes['pre_ctr_bucket'] = int(_stats['pre_ctr_bucket'].max()) + 1
        vocab_sizes['pre_lv_rate_bucket'] = int(_stats['pre_lv_rate_bucket'].max()) + 1
        vocab_sizes['pre_like_rate_bucket'] = int(_stats['pre_like_rate_bucket'].max()) + 1
        vocab_sizes['pre_play_ratio_bucket'] = int(_stats['pre_play_ratio_bucket'].max()) + 1
        vocab_sizes['pre_show_bucket'] = int(_stats['pre_show_bucket'].max()) + 1
        logger.info(f"动态读取分桶特征 vocab_sizes: pre_ctr={vocab_sizes['pre_ctr_bucket']}, "
                    f"pre_lv={vocab_sizes['pre_lv_rate_bucket']}, "
                    f"pre_like={vocab_sizes['pre_like_rate_bucket']}, "
                    f"pre_play={vocab_sizes['pre_play_ratio_bucket']}, "
                    f"pre_show={vocab_sizes['pre_show_bucket']}")
        del _stats
    except Exception as e:
        logger.warning(f"无法动态读取 item_statistics.parquet，使用默认值: {e}")
        #  fallback 默认值
        vocab_sizes['pre_ctr_bucket'] = 4
        vocab_sizes['pre_lv_rate_bucket'] = 4
        vocab_sizes['pre_like_rate_bucket'] = 4
        vocab_sizes['pre_play_ratio_bucket'] = 6
        vocab_sizes['pre_show_bucket'] = 5

    # 用户分桶特征
    for col in ['follow_user_num_range', 'fans_user_num_range',
                'friend_user_num_range', 'register_days_range']:
        try:
            with open(f"{OUTPUT_DIR}/vocabs/{col}.json", 'r') as f:
                vocab = json.load(f)
                vocab_sizes[col] = len(vocab)
        except:
            vocab_sizes[col] = 10

    # onehot feats
    for i in range(18):
        try:
            with open(f"{OUTPUT_DIR}/vocabs/onehot_feat{i}.json", 'r') as f:
                vocab = json.load(f)
                vocab_sizes[f'onehot_feat{i}'] = len(vocab)
        except:
            vocab_sizes[f'onehot_feat{i}'] = 100

    # binary 用户特征（0=PAD, 1=False, 2=True）
    vocab_sizes['is_lowactive_period'] = 3
    vocab_sizes['is_live_streamer'] = 3
    vocab_sizes['is_video_author'] = 3

    # 历史序列分桶特征（col_to_vocab_name: hist_xxx → xxx）
    vocab_sizes['delta_t_bucket'] = 7   # 5个边界 → 桶1~6 + PAD(0)
    vocab_sizes['play_ratio_bucket'] = 7

    return vocab_sizes

def generate_feature_vocab_manifest():
    """生成 feature_vocab_manifest.json，列出所有需要 vocab 文件的特征"""
    manifest = {
        "video_id": "vocabs/video_id.json",
        "author_id": "vocabs/author_id.json",
        "upload_type": "vocabs/upload_type.json",
        "tab": "vocabs/tab.json",
        "user_active_degree": "vocabs/user_active_degree.json",
        "follow_user_num_range": "vocabs/follow_user_num_range.json",
        "fans_user_num_range": "vocabs/fans_user_num_range.json",
        "friend_user_num_range": "vocabs/friend_user_num_range.json",
        "register_days_range": "vocabs/register_days_range.json",
    }
    for i in range(18):
        manifest[f"onehot_feat{i}"] = f"vocabs/onehot_feat{i}.json"
    return manifest

def perform_sanity_checks(train_df, val_df, test_df, random_df):
    """执行数据检查"""
    checks = {}

    # 标签分布
    for name, df in [('train', train_df), ('val', val_df),
                     ('test_standard', test_df), ('test_random', random_df)]:
        checks[f'{name}_label_dist'] = {
            'total': len(df),
            'positive': int(df['label_long_view'].sum()),
            'ctr': float(df['label_long_view'].mean())
        }

    # 时序单调性验证（按时间切分后，train最大时间 <= val最小时间 <= test最小时间）
    train_max_t = int(train_df['sample_time_ms'].max())
    val_min_t   = int(val_df['sample_time_ms'].min())
    val_max_t   = int(val_df['sample_time_ms'].max())
    test_min_t  = int(test_df['sample_time_ms'].min())

    checks['temporal_split_check'] = {
        'train_max_time_ms': train_max_t,
        'val_min_time_ms': val_min_t,
        'val_max_time_ms': val_max_t,
        'test_min_time_ms': test_min_t,
        'train_before_val': train_max_t <= val_min_t,
        'val_before_test': val_max_t <= test_min_t,
        'status': 'PASSED' if (train_max_t <= val_min_t and val_max_t <= test_min_t) else 'FAILED'
    }

    # 用户分布（时序切分后 train/val/test 用户有重叠，此处仅做参考统计）
    checks['user_coverage'] = {
        'train_unique_users': int(train_df['user_id_raw'].nunique()),
        'val_unique_users': int(val_df['user_id_raw'].nunique()),
        'test_unique_users': int(test_df['user_id_raw'].nunique()),
    }

    # 历史序列检查
    for name, df in [('train', train_df), ('val', val_df)]:
        checks[f'{name}_hist_len'] = {
            'mean': float(df['hist_len'].mean()),
            'max': int(df['hist_len'].max()),
            'min': int(df['hist_len'].min())
        }

    # 列表列长度检查（必须严格等于50）
    list_cols = ['hist_video_id', 'hist_author_id', 'hist_mask']
    for col in list_cols:
        lens = train_df[col].apply(len)
        checks[f'{col}_length_check'] = 'PASSED' if (lens == 50).all() else 'FAILED'

    return checks

def main():
    logger.info("=" * 60)
    logger.info("Step 4: Splitting and Generating Meta Files (V2 - Strict Match)")
    logger.info("=" * 60)

    logger.info("Loading samples (essential cols only)...")
    cols_for_split = ['user_id_raw', 'sample_time_ms', 'label_long_view']
    standard_df = pd.read_parquet(f"{OUTPUT_DIR}/samples_standard.parquet", columns=cols_for_split)
    random_df = pd.read_parquet(f"{OUTPUT_DIR}/samples_random.parquet")

    logger.info(f"Standard samples: {len(standard_df)}")
    logger.info(f"Random samples: {len(random_df)}")

    logger.info("\nSplitting by time (strict temporal 70/10/20)...")
    standard_df = standard_df.sort_values('sample_time_ms').reset_index(drop=True)
    n = len(standard_df)
    n_train = int(n * TRAIN_RATIO)
    n_val_boundary = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_boundary_time = int(standard_df.iloc[n_train]['sample_time_ms'])
    val_boundary_time   = int(standard_df.iloc[n_val_boundary]['sample_time_ms'])
    logger.info(f"  Train boundary time_ms: {train_boundary_time}")
    logger.info(f"  Val   boundary time_ms: {val_boundary_time}")
    del standard_df
    gc.collect()

    os.makedirs(f"{OUTPUT_DIR}/processed", exist_ok=True)

    def read_split_and_save(filters, out_path, split_name):
        logger.info(f"Reading {split_name} split...")
        import pyarrow.parquet as pq
        import pyarrow as pa

        pf = pq.ParquetFile(f"{OUTPUT_DIR}/samples_standard.parquet")
        writer = None
        total = 0
        stats = None
        all_min_t, all_max_t = [], []
        n_users_set = set()
        lv_sum = 0

        for batch in pf.iter_batches(batch_size=500000):
            table = pa.Table.from_batches([batch])
            df = table.to_pandas()

            # 应用过滤条件
            for col, op, val in filters:
                if op == '<':
                    df = df[df[col] < val]
                elif op == '>=':
                    df = df[df[col] >= val]

            if len(df) == 0:
                del df, table
                continue

            df = df.sort_values('sample_time_ms').reset_index(drop=True)

            all_min_t.append(int(df['sample_time_ms'].min()))
            all_max_t.append(int(df['sample_time_ms'].max()))
            n_users_set.update(df['user_id_raw'].unique().tolist())
            lv_sum += int(df['label_long_view'].sum())
            total += len(df)

            t = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, t.schema)
            writer.write_table(t)

            del df, table, t
            gc.collect()

        if writer:
            writer.close()

        stats = {
            'n_samples': total,
            'n_users': len(n_users_set),
            'min_t': min(all_min_t) if all_min_t else 0,
            'max_t': max(all_max_t) if all_max_t else 0,
            'lv_positive': lv_sum,
            'lv_ctr': lv_sum / total if total > 0 else 0.0,
        }

        # hist长度检查（只对train）
        if split_name == 'train':
            logger.info("  Checking hist lengths (sampling first batch)...")
            pf2 = pq.ParquetFile(out_path)
            sample_batch = next(pf2.iter_batches(batch_size=10000))
            df_check = pa.Table.from_batches([sample_batch]).to_pandas()
            for col in ['hist_video_id', 'hist_author_id', 'hist_mask']:
                lens = df_check[col].apply(len)
                stats[f'{col}_length_check'] = 'PASSED' if (lens == 50).all() else 'FAILED'
            stats['hist_len_mean'] = float(df_check['hist_len'].mean())
            stats['hist_len_max'] = int(df_check['hist_len'].max())
            stats['hist_len_min'] = int(df_check['hist_len'].min())
            del df_check

        logger.info(f"  {split_name}: {total:,} samples written")
        return stats

    train_stats = read_split_and_save(
        filters=[('sample_time_ms', '<', train_boundary_time)],
        out_path=f"{OUTPUT_DIR}/processed/train.parquet",
        split_name='train'
    )
    val_stats = read_split_and_save(
        filters=[('sample_time_ms', '>=', train_boundary_time),
                 ('sample_time_ms', '<',  val_boundary_time)],
        out_path=f"{OUTPUT_DIR}/processed/val.parquet",
        split_name='val'
    )
    test_stats = read_split_and_save(
        filters=[('sample_time_ms', '>=', val_boundary_time)],
        out_path=f"{OUTPUT_DIR}/processed/test_standard.parquet",
        split_name='test_standard'
    )

    logger.info("Saving test_random...")
    random_df.to_parquet(f"{OUTPUT_DIR}/processed/test_random.parquet", index=False)
    random_stats = {
        'n_samples': len(random_df),
        'n_users':   int(random_df['user_id_raw'].nunique()),
        'min_t':     int(random_df['sample_time_ms'].min()),
        'max_t':     int(random_df['sample_time_ms'].max()),
        'lv_positive': int(random_df['label_long_view'].sum()),
        'lv_ctr':    float(random_df['label_long_view'].mean()),
    }
    del random_df
    gc.collect()

    logger.info(f"\nFinal counts:")
    logger.info(f"Train:  {train_stats['n_samples']:,}")
    logger.info(f"Val:    {val_stats['n_samples']:,}")
    logger.info(f"Test:   {test_stats['n_samples']:,}")
    logger.info(f"Random: {random_stats['n_samples']:,}")

    assert train_stats['max_t'] <= val_stats['min_t'],   "时序切分错误: train_max > val_min"
    assert val_stats['max_t']   <= test_stats['min_t'],  "时序切分错误: val_max > test_min"
    logger.info("Temporal split check: PASSED")

    logger.info("\nGenerating field_schema.json...")
    field_schema = generate_field_schema()
    vocab_sizes = compute_vocab_sizes()
    field_schema["vocab_sizes"] = vocab_sizes
    os.makedirs(f"{OUTPUT_DIR}/meta", exist_ok=True)
    with open(f"{OUTPUT_DIR}/meta/field_schema.json", 'w') as f:
        json.dump(field_schema, f, indent=2)

    with open(f"{OUTPUT_DIR}/meta/vocab_sizes.json", 'w') as f:
        json.dump(vocab_sizes, f, indent=2)

    manifest = generate_feature_vocab_manifest()
    with open(f"{OUTPUT_DIR}/meta/feature_vocab_manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    split_summary = {
        "train":         {"n_users": train_stats['n_users'],  "n_samples": train_stats['n_samples'],  "date_range": {"start": train_stats['min_t'],  "end": train_stats['max_t']}},
        "val":           {"n_users": val_stats['n_users'],    "n_samples": val_stats['n_samples'],    "date_range": {"start": val_stats['min_t'],    "end": val_stats['max_t']}},
        "test_standard": {"n_users": test_stats['n_users'],   "n_samples": test_stats['n_samples'],   "date_range": {"start": test_stats['min_t'],   "end": test_stats['max_t']}},
        "test_random":   {"n_users": random_stats['n_users'], "n_samples": random_stats['n_samples'], "date_range": {"start": random_stats['min_t'], "end": random_stats['max_t']}},
        "generated_at":  datetime.now().isoformat()
    }
    with open(f"{OUTPUT_DIR}/meta/split_summary.json", 'w') as f:
        json.dump(split_summary, f, indent=2)

    checks = {
        'train_label_dist':         {'total': train_stats['n_samples'],  'positive': train_stats['lv_positive'],  'ctr': train_stats['lv_ctr']},
        'val_label_dist':           {'total': val_stats['n_samples'],    'positive': val_stats['lv_positive'],    'ctr': val_stats['lv_ctr']},
        'test_standard_label_dist': {'total': test_stats['n_samples'],   'positive': test_stats['lv_positive'],   'ctr': test_stats['lv_ctr']},
        'test_random_label_dist':   {'total': random_stats['n_samples'], 'positive': random_stats['lv_positive'], 'ctr': random_stats['lv_ctr']},
        'temporal_split_check': {
            'train_max_time_ms': train_stats['max_t'], 'val_min_time_ms': val_stats['min_t'],
            'val_max_time_ms':   val_stats['max_t'],   'test_min_time_ms': test_stats['min_t'],
            'train_before_val':  train_stats['max_t'] <= val_stats['min_t'],
            'val_before_test':   val_stats['max_t']   <= test_stats['min_t'],
            'status': 'PASSED'
        },
        'user_coverage': {'train_unique_users': train_stats['n_users'], 'val_unique_users': val_stats['n_users'], 'test_unique_users': test_stats['n_users']},
        'train_hist_len': {'mean': train_stats.get('hist_len_mean'), 'max': train_stats.get('hist_len_max'), 'min': train_stats.get('hist_len_min')},
        'hist_video_id_length_check': train_stats.get('hist_video_id_length_check', 'N/A'),
        'hist_author_id_length_check': train_stats.get('hist_author_id_length_check', 'N/A'),
        'hist_mask_length_check': train_stats.get('hist_mask_length_check', 'N/A'),
    }
    with open(f"{OUTPUT_DIR}/meta/sanity_checks.json", 'w') as f:
        json.dump(checks, f, indent=2)

    logger.info(f"\nSanity Check Results:")
    logger.info(f"Train:  {train_stats['n_samples']:,} (CTR: {train_stats['lv_ctr']:.4f})")
    logger.info(f"Val:    {val_stats['n_samples']:,} (CTR: {val_stats['lv_ctr']:.4f})")
    logger.info(f"Test:   {test_stats['n_samples']:,} (CTR: {test_stats['lv_ctr']:.4f})")
    logger.info(f"Random: {random_stats['n_samples']:,} (CTR: {random_stats['lv_ctr']:.4f})")
    logger.info(f"Hist mask check: {train_stats.get('hist_mask_length_check', 'N/A')}")

    logger.info("\n" + "=" * 60)
    logger.info("Step 4 completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()