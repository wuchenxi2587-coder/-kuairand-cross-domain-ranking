#!/usr/bin/env python3
"""
KuaiRand-27K 预处理主入口 (修正版)
严格匹配项目架构和 configs/train_din_mem16gb.yaml
"""

import subprocess
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_step(script_name):
    """运行单个步骤脚本"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {script_name}")
    logger.info('='*60)

    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        logger.error(f"{script_name} failed with code {result.returncode}")
        sys.exit(1)

    logger.info(f"{script_name} completed successfully")

def main():
    """按顺序运行所有步骤"""
    steps = [
        'step1_build_vocabs_v2.py',        # Step 1: 构建词表（处理负值异常）
        'step2_compute_stats.py',          # Step 2: 计算统计量（等频分桶）
        'step3_generate_samples_v2.py',    # Step 3: 生成样本（严格匹配config）
        'step4_split_and_save_v2.py'       # Step 4: 切分和生成meta（70/10/20比例）
    ]

    logger.info("="*60)
    logger.info("KuaiRand-27K Preprocessing Pipeline (V2 - Strict Config Match)")
    logger.info("="*60)

    for step in steps:
        run_step(step)

    logger.info("\n" + "="*60)
    logger.info("All steps completed successfully!")
    logger.info("="*60)
    logger.info("\nFinal output structure:")
    logger.info("  output/processed/train.parquet")
    logger.info("  output/processed/val.parquet")
    logger.info("  output/processed/test_standard.parquet")
    logger.info("  output/processed/test_random.parquet")
    logger.info("  output/meta/field_schema.json")
    logger.info("  output/meta/vocab_sizes.json")
    logger.info("  output/meta/feature_vocab_manifest.json")
    logger.info("  output/meta/split_summary.json")
    logger.info("  output/meta/sanity_checks.json")
    logger.info("  output/vocabs/*.json")
    logger.info("\nYou can now run training with:")
    logger.info("  python -m src.main_train_din --config configs/train_din_mem16gb.yaml")

if __name__ == "__main__":
    main()
