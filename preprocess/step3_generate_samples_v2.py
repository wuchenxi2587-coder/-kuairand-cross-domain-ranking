#!/usr/bin/env python3
"""
Step 3: 生成训练样本 (多进程版)
user_id % N_WORKERS 分桶，Linux fork copy-on-write 共享只读数据
预计运行时间：2~3小时（vs 单进程 16~18小时）
"""

import pandas as pd
import numpy as np
import json
import logging
import gc
import os
import shutil
import multiprocessing as mp
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = "data/KuaiRand-27K"
OUTPUT_DIR = "output"
MAX_HIST_LEN = 50
N_WORKERS = 12  # 16核机器，留4核给IO和OS

DELTA_T_BOUNDS = [60000, 300000, 1800000, 86400000, 259200000]
PLAY_RATIO_BOUNDS = [0.2, 0.5, 0.8, 1.2, 2.0]

# 只读全局数据：main进程加载，fork后子进程copy-on-write共享，不会复制12份
_VIDEO_DF = None
_USER_DF = None
_ITEM_STATS = None
_TAB_VOCAB = None


def load_processed_data():
    logger.info("Loading processed data...")
    video_df = pd.read_parquet(f"{OUTPUT_DIR}/video_features_processed.parquet")
    video_df.set_index('video_id', inplace=True)
    user_df = pd.read_parquet(f"{OUTPUT_DIR}/user_features_processed.parquet")
    user_df.set_index('user_id', inplace=True)
    item_stats = pd.read_parquet(f"{OUTPUT_DIR}/item_statistics.parquet")
    item_stats.set_index('video_id', inplace=True)
    with open(f"{OUTPUT_DIR}/vocabs/tab.json", 'r') as f:
        tab_vocab = json.load(f)
    return video_df, user_df, item_stats, tab_vocab


def process_user_logs(user_logs, user_id, user_info, history=None):
    """处理单个用户日志，使用全局只读数据（无需传参，fork后共享）"""
    samples = []
    if history is None:
        history = []

    user_logs = user_logs.sort_values('time_ms')

    for _, row in user_logs.iterrows():
        video_id = int(row['video_id'])

        sample = {
            'label_long_view': int(row['long_view']),
            'user_id_raw': int(user_id),
            'sample_time_ms': int(row['time_ms']),
            'meta_is_rand': int(row.get('is_rand', 0)),
            'meta_log_source': str(row.get('log_source', 'UNKNOWN')),
        }

        if video_id in _VIDEO_DF.index:
            vinfo = _VIDEO_DF.loc[video_id]
            sample['cand_video_id'] = int(vinfo['video_id_enc'])
            sample['cand_author_id'] = int(vinfo['author_id_enc'])
            sample['cand_video_type'] = int(vinfo['video_type_enc'])
            sample['cand_upload_type'] = int(vinfo['upload_type_enc'])
            sample['cand_video_duration_bucket'] = int(vinfo['duration_bucket'])
        else:
            sample.update({
                'cand_video_id': 0, 'cand_author_id': 0,
                'cand_video_type': 0, 'cand_upload_type': 0,
                'cand_video_duration_bucket': 0,
            })

        if video_id in _ITEM_STATS.index:
            stat = _ITEM_STATS.loc[video_id]
            sample['cand_pre_ctr_bucket'] = int(stat.get('pre_ctr_bucket', 0))
            sample['cand_pre_lv_rate_bucket'] = int(stat.get('pre_lv_rate_bucket', 0))
            sample['cand_pre_like_rate_bucket'] = int(stat.get('pre_like_rate_bucket', 0))
            sample['cand_pre_play_ratio_bucket'] = int(stat.get('pre_play_ratio_bucket', 0))
            sample['cand_pre_show_bucket'] = int(stat.get('pre_show_bucket', 0))
            sample['cand_pre_ctr'] = float(stat.get('pre_ctr_raw', 0.0))
            sample['cand_pre_lv_rate'] = float(stat.get('pre_lv_rate_raw', 0.0))
            sample['cand_pre_like_rate'] = float(stat.get('pre_like_rate_raw', 0.0))
            sample['cand_pre_play_ratio'] = float(stat.get('pre_play_ratio_raw', 0.0))
            sample['cand_pre_show_log'] = float(stat.get('pre_show_log', 0.0))
            sample['cand_pre_log_show_cnt'] = float(stat.get('pre_log_show_cnt', 0.0))
        else:
            sample.update({
                'cand_pre_ctr_bucket': 0, 'cand_pre_lv_rate_bucket': 0,
                'cand_pre_like_rate_bucket': 0, 'cand_pre_play_ratio_bucket': 0,
                'cand_pre_show_bucket': 0,
                'cand_pre_ctr': 0.0, 'cand_pre_lv_rate': 0.0,
                'cand_pre_like_rate': 0.0, 'cand_pre_play_ratio': 0.0,
                'cand_pre_show_log': 0.0, 'cand_pre_log_show_cnt': 0.0,
            })

        tab_val = str(int(row['tab'])) if pd.notna(row['tab']) else 'UNKNOWN'
        sample['tab'] = _TAB_VOCAB.get(tab_val, 0)
        sample['hour_of_day'] = int(row['_hour_of_day'])
        sample['day_of_week'] = int(row['_day_of_week'])
        sample['is_weekend'] = int(row['_is_weekend'])

        sample['user_active_degree'] = int(user_info.get('user_active_degree_enc', 4))
        sample['is_lowactive_period'] = int(user_info.get('is_lowactive_period_enc', 0))
        sample['is_live_streamer'] = int(user_info.get('is_live_streamer_enc', 0))
        sample['is_video_author'] = int(user_info.get('is_video_author_enc', 0))
        sample['follow_user_num_range'] = int(user_info.get('follow_user_num_range_enc', 0))
        sample['fans_user_num_range'] = int(user_info.get('fans_user_num_range_enc', 0))
        sample['friend_user_num_range'] = int(user_info.get('friend_user_num_range_enc', 0))
        sample['register_days_range'] = int(user_info.get('register_days_range_enc', 0))

        for i in range(18):
            sample[f'onehot_feat{i}'] = int(user_info.get(f'onehot_feat{i}_enc', 1))

        sample['log1p_follow_user_num'] = float(user_info.get('log1p_follow_user_num', 0.0))
        sample['log1p_fans_user_num'] = float(user_info.get('log1p_fans_user_num', 0.0))
        sample['log1p_friend_user_num'] = float(user_info.get('log1p_friend_user_num', 0.0))
        sample['log1p_register_days'] = float(user_info.get('log1p_register_days', 0.0))

        hist = history[-MAX_HIST_LEN:] if len(history) > MAX_HIST_LEN else history
        actual_len = len(hist)
        sample['hist_video_id'] = [h.get('vid', 0) for h in hist] + [0] * (MAX_HIST_LEN - actual_len)
        sample['hist_author_id'] = [h.get('aid', 0) for h in hist] + [0] * (MAX_HIST_LEN - actual_len)
        sample['hist_mask'] = [1] * actual_len + [0] * (MAX_HIST_LEN - actual_len)
        sample['hist_len'] = actual_len
        sample['hist_delta_t_bucket'] = [h.get('dt', 0) for h in hist] + [0] * (MAX_HIST_LEN - actual_len)
        sample['hist_play_ratio_bucket'] = [h.get('pr', 0) for h in hist] + [0] * (MAX_HIST_LEN - actual_len)
        sample['hist_tab'] = [h.get('tab', 0) for h in hist] + [0] * (MAX_HIST_LEN - actual_len)

        samples.append(sample)

        if row['is_click'] == 1 or row['long_view'] == 1:
            curr_time = row['time_ms']
            prev_time = history[-1]['time'] if history else curr_time
            delta_t = curr_time - prev_time
            play_ratio = min(row['play_time_ms'] / max(row['duration_ms'], 1), 3.0)
            history.append({
                'vid': sample['cand_video_id'],
                'aid': sample['cand_author_id'],
                'time': curr_time,
                'dt': int(np.digitize(delta_t, DELTA_T_BOUNDS)) + 1 if delta_t > 0 else 1,
                'pr': int(np.digitize(play_ratio, PLAY_RATIO_BOUNDS)) + 1,
                'tab': sample['tab'],
            })

    return samples, history


def process_phase(worker_id, n_workers, log_files, is_rand, out_path):
    """处理一个phase的所有日志文件（standard或random）"""
    local_history = {}
    writer_ref = [None]  # list包装模拟引用
    buffer = []
    total = 0

    for log_file in log_files:
        for chunk in pd.read_csv(log_file, chunksize=500000):
            # 只处理本worker负责的用户
            chunk = chunk[chunk['user_id'] % n_workers == worker_id].copy()
            if len(chunk) == 0:
                continue

            chunk['is_rand'] = is_rand
            chunk['log_source'] = 'random' if is_rand else 'standard'
            chunk['_dow'] = pd.to_datetime(chunk['date'].astype(str), format='%Y%m%d').dt.dayofweek
            chunk['_hour_of_day'] = (chunk['hourmin'] // 100 + 1).astype(np.int8)
            chunk['_day_of_week'] = (chunk['_dow'] + 1).astype(np.int8)
            chunk['_is_weekend'] = ((chunk['_dow'] >= 5).astype(np.int8) + 1)

            for user_id, user_logs in chunk.groupby('user_id'):
                if user_id not in _USER_DF.index:
                    continue
                user_info = _USER_DF.loc[user_id]
                prior = local_history.get(user_id, [])
                samples, updated = process_user_logs(user_logs, user_id, user_info, history=prior)
                local_history[user_id] = updated
                buffer.extend(samples)

            if len(buffer) >= 50000:
                df = pd.DataFrame(buffer)
                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer_ref[0] is None:
                    writer_ref[0] = pq.ParquetWriter(out_path, table.schema)
                writer_ref[0].write_table(table)
                total += len(buffer)
                buffer = []
                del df, table
                gc.collect()

    if buffer:
        df = pd.DataFrame(buffer)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer_ref[0] is None:
            writer_ref[0] = pq.ParquetWriter(out_path, table.schema)
        writer_ref[0].write_table(table)
        total += len(buffer)

    if writer_ref[0]:
        writer_ref[0].close()

    return total


def worker_fn(args):
    """Worker入口：standard（part1+part2连续历史）→ random（独立历史）"""
    worker_id, n_workers = args

    std_total = process_phase(
        worker_id, n_workers,
        log_files=[
            f"{DATA_DIR}/log_standard_4_22_to_5_08_27k_part1.csv",
            f"{DATA_DIR}/log_standard_4_22_to_5_08_27k_part2.csv",
        ],
        is_rand=0,
        out_path=f"{OUTPUT_DIR}/temp/standard_{worker_id:02d}.parquet"
    )

    rnd_total = process_phase(
        worker_id, n_workers,
        log_files=[f"{DATA_DIR}/log_random_4_22_to_5_08_27k.csv"],
        is_rand=1,
        out_path=f"{OUTPUT_DIR}/temp/random_{worker_id:02d}.parquet"
    )

    print(f"[W{worker_id:02d}] std={std_total:,}  rnd={rnd_total,:}", flush=True)
    return worker_id, std_total, rnd_total


def merge_parquets(temp_paths, out_path, label):
    logger.info(f"Merging {label}...")
    writer = None
    total = 0
    for path in temp_paths:
        if not os.path.exists(path):
            logger.warning(f"Missing: {path}")
            continue
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=500000):
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)
            total += len(table)
    if writer:
        writer.close()
    logger.info(f"  {label} total: {total:,}")
    return total


def main():
    global _VIDEO_DF, _USER_DF, _ITEM_STATS, _TAB_VOCAB

    logger.info("=" * 60)
    logger.info(f"Step 3 (Multiprocess, N_WORKERS={N_WORKERS})")
    logger.info("=" * 60)

    # fork前加载只读数据，子进程copy-on-write共享，不会有12份拷贝
    _VIDEO_DF, _USER_DF, _ITEM_STATS, _TAB_VOCAB = load_processed_data()
    logger.info("Read-only data loaded. Forking workers...")

    os.makedirs(f"{OUTPUT_DIR}/temp", exist_ok=True)

    # Linux fork：子进程继承父进程内存，只读数据零拷贝
    ctx = mp.get_context('fork')
    with ctx.Pool(N_WORKERS) as pool:
        results = pool.map(worker_fn, [(i, N_WORKERS) for i in range(N_WORKERS)])

    logger.info("All workers completed. Merging output...")

    std_paths = [f"{OUTPUT_DIR}/temp/standard_{i:02d}.parquet" for i in range(N_WORKERS)]
    rnd_paths = [f"{OUTPUT_DIR}/temp/random_{i:02d}.parquet" for i in range(N_WORKERS)]

    std_total = merge_parquets(std_paths, f"{OUTPUT_DIR}/samples_standard.parquet", "standard")
    rnd_total = merge_parquets(rnd_paths, f"{OUTPUT_DIR}/samples_random.parquet", "random")

    shutil.rmtree(f"{OUTPUT_DIR}/temp")
    logger.info("Temp files cleaned.")

    logger.info(f"\nStandard samples: {std_total:,}")
    logger.info(f"Random samples:   {rnd_total:,}")
    logger.info("\nStep 3 completed!")


if __name__ == "__main__":
    main()
