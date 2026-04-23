#!/usr/bin/env python3
"""
Step 1: 构建词表和基础特征 (修正版)
处理异常值（如 -124 的 binary 特征）
"""

import pandas as pd
import numpy as np
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = "/111111/newproject/data/KuaiRand-27K"
OUTPUT_DIR = "/111111/newproject/output"

def build_category_vocab(unique_vals, name):
    """构建类别词表，处理各种异常值"""
    vocab = {'__PAD__': 0}
    # 过滤 nan 和负数（视为异常）
    clean_vals = [v for v in unique_vals if pd.notna(v) and (not isinstance(v, (int, float)) or v >= 0)]
    for i, v in enumerate(sorted(set(clean_vals)), 1):
        vocab[str(v)] = i
    logger.info(f"{name}: {len(vocab)} tokens")
    return vocab

def main():
    logger.info("=" * 60)
    logger.info("Step 1: Building Vocabs and Processing Features (V2)")
    logger.info("=" * 60)

    # 读取数据
    logger.info("Loading data...")
    user_df = pd.read_csv(f"{DATA_DIR}/user_features_27k.csv")
    video_df = pd.read_csv(f"{DATA_DIR}/video_features_basic_27k.csv")

    # tab 取值范围在数据集文档中明确为 [0, 14]，无需扫描文件
    tabs = set(range(15))

    # 构建词表
    logger.info("Building vocabs...")
    vocabs = {}

    # ID 类
    vocabs['video_id'] = {int(v): i+1 for i, v in enumerate(sorted(video_df['video_id'].unique()))}
    vocabs['video_id']['__PAD__'] = 0
    vocabs['author_id'] = {int(v): i+1 for i, v in enumerate(sorted(video_df['author_id'].unique()))}
    vocabs['author_id']['__PAD__'] = 0

    # 类别
    vocabs['video_type'] = {'__PAD__': 0, 'NORMAL': 1, 'AD': 2}
    vocabs['user_active_degree'] = {'__PAD__': 0, 'high_active': 1, 'full_active': 2, 'middle_active': 3, 'UNKNOWN': 4}
    vocabs['tab'] = build_category_vocab(list(tabs), 'tab')

    # 用户分桶特征（range 版本）
    for col in ['follow_user_num_range', 'fans_user_num_range',
                'friend_user_num_range', 'register_days_range']:
        vocabs[col] = build_category_vocab(user_df[col].unique(), col)

    # onehot feats: 值+1，负数视为 PAD (0)
    for i in range(18):
        max_val = int(user_df[f'onehot_feat{i}'].max())
        max_val = max(max_val, 0)  # 确保非负
        vocab = {'__PAD__': 0}
        for v in range(max_val + 1):
            vocab[v] = v + 1
        vocabs[f'onehot_feat{i}'] = vocab

    # 处理视频特征
    logger.info("Processing video features...")
    video_features = video_df.copy()
    video_features['video_id_enc'] = video_features['video_id'].map(vocabs['video_id']).fillna(0).astype(np.int32)
    video_features['author_id_enc'] = video_features['author_id'].map(vocabs['author_id']).fillna(0).astype(np.int32)
    video_features['video_type_enc'] = video_features['video_type'].map(vocabs['video_type']).fillna(0).astype(np.int16)

    # upload_type 和 music_type
    vocabs['upload_type'] = build_category_vocab(video_features['upload_type'].unique(), 'upload_type')
    vocabs['music_type'] = build_category_vocab(video_features['music_type'].unique(), 'music_type')
    video_features['upload_type_enc'] = video_features['upload_type'].map(vocabs['upload_type']).fillna(0).astype(np.int16)
    video_features['music_type_enc'] = video_features['music_type'].map(vocabs['music_type']).fillna(0).astype(np.int16)

    # duration 分桶（毫秒）
    # 先将 NaN 填为 0（NaN digitize 会产生错误桶），NaN → PAD=0
    duration_bounds = [7000, 18000, 30000, 60000]
    duration_values = pd.to_numeric(video_features['video_duration'], errors='coerce').fillna(0)
    video_features['duration_bucket'] = np.where(
        duration_values <= 0,
        0,  # NaN 或无效值 → PAD
        np.digitize(duration_values, duration_bounds).astype(np.int8) + 1
    )
    video_features['duration_bucket'] = video_features['duration_bucket'].astype(np.int8)

    # 保存词表
    logger.info("Saving vocabs...")
    os.makedirs(f"{OUTPUT_DIR}/vocabs", exist_ok=True)
    for name, vocab in vocabs.items():
        with open(f"{OUTPUT_DIR}/vocabs/{name}.json", 'w') as f:
            json.dump(vocab, f, indent=2)

    video_features.to_parquet(f"{OUTPUT_DIR}/video_features_processed.parquet", index=False)

    # 处理用户特征（关键修正：处理负数）
    logger.info("Processing user features with negative value handling...")
    user_features = user_df.copy()

    # Sparse: 分桶版本（range 后缀）
    for col in ['follow_user_num_range', 'fans_user_num_range',
                'friend_user_num_range', 'register_days_range']:
        user_features[f'{col}_enc'] = user_features[col].map(vocabs[col]).fillna(0).astype(np.int8)

    user_features['user_active_degree_enc'] = user_features['user_active_degree'].map(
        vocabs['user_active_degree']).fillna(4).astype(np.int16)

    # Binary 特征：负数视为 PAD (0)，0->1, 1->2
    for col in ['is_lowactive_period', 'is_live_streamer', 'is_video_author']:
        def encode_binary(x):
            if pd.isna(x) or x < 0:
                return 0  # PAD
            return int(x) + 1
        user_features[f'{col}_enc'] = user_features[col].apply(encode_binary).astype(np.int8)

    # onehot: 负数视为 0，然后 +1
    for i in range(18):
        user_features[f'onehot_feat{i}_enc'] = user_features[f'onehot_feat{i}'].apply(
            lambda x: 0 if pd.isna(x) or x < 0 else int(x) + 1).astype(np.int16)

    # Dense: log1p，负数视为 0
    for col in ['follow_user_num', 'fans_user_num', 'friend_user_num', 'register_days']:
        user_features[f'log1p_{col}'] = np.log1p(
            user_features[col].apply(lambda x: max(x, 0) if pd.notna(x) else 0)
        ).astype(np.float32)

    user_features.to_parquet(f"{OUTPUT_DIR}/user_features_processed.parquet", index=False)

    logger.info("Step 1 completed!")
    logger.info(f"Video features: {len(video_features)}")
    logger.info(f"User features: {len(user_features)}")

if __name__ == "__main__":
    main()
